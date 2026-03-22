"""
This script runs Ray Tune hyperparameter optimization for a 2D Advection PI-DeepONet using HyperNOs and DeepXDE.
The problem setup is based on DeepXDE/examples/operator/advection_aligned_pideeponet_2d.py, 
which treats the 1D time-dependent problem as a 2D spatial problem.
"""

import os

# Ensure DeepXDE uses PyTorch backend
os.environ["DDE_BACKEND"] = "pytorch"

import numpy as np
import torch
from torch.utils.data import DataLoader
import deepxde as dde
from ray import tune

# Import HyperNOs components
from hypernos.tune import tune_hyperparameters
from hypernos.loss_fun import LprelLoss
from hypernos.datasets import deeponet_collate_fn


# 1. Define the DeepXDE PDE problem (2D spatial mapping of 1D advection)
def solve_advection_2d():
    # PDE: u_y + u_x = 0 (where y is time)
    def pde_fn(x, y):
        dy_x = dde.grad.jacobian(y, x, j=0)
        dy_y = dde.grad.jacobian(y, x, j=1)
        return dy_y + dy_x

    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    def func_ic(x, v):
        return v

    def boundary(x, on_boundary):
        return on_boundary and np.isclose(x[1], 0)

    ic = dde.icbc.DirichletBC(geom, func_ic, boundary)

    pde = dde.data.PDE(geom, pde_fn, ic, num_domain=200, num_boundary=200)

    # Function space
    func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)

    # Data
    eval_pts = np.linspace(0, 1, num=50)[:, None]
    pde_op = dde.data.PDEOperatorCartesianProd(
        pde,
        func_space,
        eval_pts,
        1000,
        function_variables=[0],
        num_test=100,
        batch_size=32,
    )
    return pde_op


# 2. Custom Dataset class for Cartesian Product DeepONet
class PDECartesianDataset(torch.utils.data.Dataset):
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.tensor(trunk_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return self.branch_data.shape[0]

    def __getitem__(self, idx):
        return self.branch_data[idx], self.trunk_data, self.target_data[idx]


# 3. Dataset Builder for HyperNOs
def dataset_builder(config):
    pde_op = solve_advection_2d()

    # Generate training and test data
    X_train, y_train, _ = pde_op.train_next_batch(config["training_samples"])
    X_test, y_test, _ = pde_op.test()

    # Handle potentially missing targets in pure physics-informed setup.
    # In this HPO demo, we use ones to avoid division by zero.
    if y_train is None:
        y_train = np.ones((X_train[0].shape[0], X_train[1].shape[0], 1))
    if y_test is None:
        y_test = np.ones((X_test[0].shape[0], X_test[1].shape[0], 1))

    if y_train.shape[-1] == 1:
        y_train = y_train.squeeze(-1)
    if y_test.shape[-1] == 1:
        y_test = y_test.squeeze(-1)

    class PDEData:
        def __init__(self, X_train, y_train, X_test, y_test, batch_size):
            self.train_loader = DataLoader(
                PDECartesianDataset(X_train[0], X_train[1], y_train),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=deeponet_collate_fn,
            )
            self.val_loader = DataLoader(
                PDECartesianDataset(X_test[0], X_test[1], y_test),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=deeponet_collate_fn,
            )
            self.test_loader = self.val_loader

    return PDEData(X_train, y_train, X_test, y_test, config["batch_size"])


# 4. Model Builder for HyperNOs
def model_builder(config):
    num_eval_points = 50
    p = config["network_width"]

    # Feature transform increases input dimension for trunk to 5
    layer_sizes_branch = (
        [num_eval_points] + [config["network_width"]] * config["branch_depth"] + [p]
    )
    layer_sizes_trunk = (
        [5] + [config["network_width"]] * config["trunk_depth"] + [p]
    )

    model = dde.nn.DeepONetCartesianProd(
        layer_sizes_branch,
        layer_sizes_trunk,
        "tanh",
        "Glorot normal",
    )

    # Periodic feature transform: x -> [cos(x), sin(x), cos(2x), sin(2x), y]
    def periodic(x):
        xt, yt = x[:, :1], x[:, 1:]
        xt = xt * 2 * np.pi
        return torch.cat(
            [torch.cos(xt), torch.sin(xt), torch.cos(2 * xt), torch.sin(2 * xt), yt], 1
        )

    model.apply_feature_transform(periodic)

    return model


def main():
    # Define the Search Space
    config_space = {
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "network_width": tune.choice([64, 128]),
        "branch_depth": tune.choice([2, 3]),
        "trunk_depth": tune.choice([2, 3]),
        "batch_size": 32,
        "training_samples": 500,
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "scheduler_step": 10,
        "scheduler_gamma": 0.95,
        "problem_dim": 2,
    }

    # Initial points for HyperOpt
    default_hyper_params = {
        "learning_rate": 0.0005,
        "network_width": 128,
        "branch_depth": 3,
        "trunk_depth": 3,
        "batch_size": 32,
        "training_samples": 500,
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.95,
        "problem_dim": 2,
    }

    loss_fn = LprelLoss(p=2, size_mean=True)

    print("Starting Hyperparameter Optimization for 2D Advection DeepONet...")

    best_result = tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        default_hyper_params=[default_hyper_params],
        num_samples=4,
        max_epochs=20,
        grace_period=5,
        reduction_factor=2,
        runs_per_cpu=1.0,
        runs_per_gpu=0.0,
    )

    print("\nBest hyperparameters found:")
    for key, value in best_result.config.items():
        print(f"  {key}: {value}")

    print(f"\nBest relative loss: {best_result.metrics['relative_loss']:.6f}")


if __name__ == "__main__":
    main()
