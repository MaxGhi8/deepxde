"""
This script runs Ray Tune hyperparameter optimization for a 1D Advection PI-DeepONet using HyperNOs and DeepXDE.
The problem setup is based on DeepXDE/examples/operator/advection_aligned_pideeponet.py.
"""

import os

# Ensure DeepXDE uses PyTorch backend
os.environ["DDE_BACKEND"] = "pytorch"

import numpy as np
import torch
from hypernos.datasets import deeponet_collate_fn
from hypernos.loss_fun import LprelLoss

# Import HyperNOs components
from hypernos.tune import tune_hyperparameters
from ray import tune
from torch.utils.data import DataLoader

import deepxde as dde


# 1. Define the DeepXDE PDE problem
def solve_advection_1d():
    # PDE: u_t + u_x = 0
    def pde_fn(x, y):
        dy_x = dde.grad.jacobian(y, x, j=0)
        dy_t = dde.grad.jacobian(y, x, j=1)
        return dy_t + dy_x

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def func_ic(x, v):
        return v

    ic = dde.icbc.IC(geomtime, func_ic, lambda _, on_initial: on_initial)

    pde = dde.data.TimePDE(
        geomtime, pde_fn, ic, num_domain=250, num_initial=50, num_test=500
    )

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
    return pde_op, eval_pts


# 2. Custom Dataset class for Cartesian Product DeepONet
class PDECartesianDataset(torch.utils.data.Dataset):
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.tensor(trunk_data, dtype=torch.float32)
        # For PI-DeepONet, target_data might be dummy or initial conditions
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return self.branch_data.shape[0]

    def __getitem__(self, idx):
        return self.branch_data[idx], self.trunk_data, self.target_data[idx]


# 3. Dataset Builder for HyperNOs
def dataset_builder(config):
    pde_op, _ = solve_advection_1d()

    # Generate training and test data
    X_train, y_train, _ = pde_op.train_next_batch(config["training_samples"])
    X_test, y_test, _ = pde_op.test()

    # In PI-DeepONet, we often use dummy targets if we are pure physics-informed,
    # or initial condition values. In this HPO demo, we use ones to avoid division by zero.
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
    dim_trunk = 2  # (x, t)
    p = config["network_width"]

    layer_sizes_branch = (
        [num_eval_points] + [config["network_width"]] * config["branch_depth"] + [p]
    )
    layer_sizes_trunk = (
        [dim_trunk] + [config["network_width"]] * config["trunk_depth"] + [p]
    )

    model = dde.nn.DeepONetCartesianProd(
        layer_sizes_branch,
        layer_sizes_trunk,
        "tanh",
        "Glorot normal",
    )

    # Optional: Feature transform for periodicity as in the base example
    def periodic(x):
        xt, tt = x[:, :1], x[:, 1:]
        xt = xt * 2 * np.pi
        return torch.cat(
            [torch.cos(xt), torch.sin(xt), torch.cos(2 * xt), torch.sin(2 * xt), tt], 1
        )

    # Note: Applying feature transform might change trunk input dim to 5
    # If we apply it, we need to adjust layer_sizes_trunk[0]
    layer_sizes_trunk_transformed = (
        [5] + [config["network_width"]] * config["trunk_depth"] + [p]
    )
    model = dde.nn.DeepONetCartesianProd(
        layer_sizes_branch,
        layer_sizes_trunk_transformed,
        "tanh",
        "Glorot normal",
    )
    model.apply_feature_transform(periodic)

    return model


def main():
    config_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "network_width": tune.choice([32, 64, 128]),
        "branch_depth": tune.choice([2, 3]),
        "trunk_depth": tune.choice([2, 3]),
        "batch_size": 32,
        "training_samples": 500,
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "scheduler_step": 10,
        "scheduler_gamma": 0.95,
        "problem_dim": 1,
    }

    default_hyper_params = {
        "learning_rate": 0.001,
        "network_width": 64,
        "branch_depth": 3,
        "trunk_depth": 3,
        "batch_size": 32,
        "training_samples": 500,
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.95,
        "problem_dim": 1,
    }

    loss_fn = LprelLoss(p=2, size_mean=True)

    print("Starting HPO for Advection DeepONet...")

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


if __name__ == "__main__":
    main()
