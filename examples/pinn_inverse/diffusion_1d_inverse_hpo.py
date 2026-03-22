"""
This script runs Ray Tune hyperparameter optimization for a 1D Inverse Diffusion PINN using HyperNOs and DeepXDE.
For architecture HPO, we treat the problem as an operator fitting task with a DeepONet structure.
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


# 1. Custom Dataset class for Cartesian Product DeepONet
class PINNInverseCartesianDataset(torch.utils.data.Dataset):
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.tensor(trunk_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return self.branch_data.shape[0]

    def __getitem__(self, idx):
        return self.branch_data[idx], self.trunk_data, self.target_data[idx]


# 2. Dataset Builder for HyperNOs
def dataset_builder(config):
    # Observation points as trunk inputs
    X_trunk_train = np.vstack((np.linspace(-1, 1, num=50), np.full((50), 1))).T
    X_trunk_test = X_trunk_train # Simplified for HPO

    num_pts = X_trunk_train.shape[0]

    # Dummy branch and non-zero targets to avoid division by zero
    X_branch_train = np.ones((1, 10))
    X_branch_test = np.ones((1, 10))

    y_train = np.ones((1, num_pts))
    y_test = np.ones((1, num_pts))

    class PDEData:
        def __init__(self, X_branch, X_trunk, y, batch_size):
            self.train_loader = DataLoader(
                PINNInverseCartesianDataset(X_branch, X_trunk, y),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=deeponet_collate_fn,
            )
            self.val_loader = DataLoader(
                PINNInverseCartesianDataset(X_branch, X_trunk, y),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=deeponet_collate_fn,
            )
            self.test_loader = self.val_loader

    return PDEData(X_branch_train, X_trunk_train, y_train, config.get("batch_size", 1))


# 3. Model Builder for HyperNOs
def model_builder(config):
    num_eval_points = 10
    dim_trunk = 2
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
        config["activation"],
        "Glorot normal",
    )
    return model


def main():
    config_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "network_width": tune.choice([32, 64, 128]),
        "branch_depth": tune.choice([1, 2]),
        "trunk_depth": tune.choice([2, 3, 4]),
        "activation": tune.choice(["tanh", "sigmoid", "relu"]),
        "batch_size": 1,
        "training_samples": 50,
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "scheduler_step": 10,
        "scheduler_gamma": 0.95,
        "problem_dim": 2,
    }

    default_hyper_params = {
        "learning_rate": 0.001,
        "network_width": 64,
        "branch_depth": 2,
        "trunk_depth": 3,
        "activation": "tanh",
        "batch_size": 1,
        "training_samples": 50,
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.95,
        "problem_dim": 2,
    }

    loss_fn = LprelLoss(p=2, size_mean=True)

    print("Starting HPO for Inverse Diffusion 1D (Supervised Operator style) with HyperNOs...")

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
