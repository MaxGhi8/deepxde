2D advection: Comprehensive HPO with HyperNOs
=============================================

This example demonstrates a rich hyper-parameter optimization (HPO) for a 2D advection problem using `HyperNOs <https://github.com/MaxGhi8/HyperNOs>`_ and `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_.

In scientific machine learning, finding the right combination of network architecture, optimizer settings, and non-linear activation functions is often more of an art than a science. This demo showcases how to automate this process by exploring a high-dimensional search space efficiently.

Problem setup
-------------

We consider the 2D advection equation:

.. math:: \frac{\partial u}{\partial y} + \frac{\partial u}{\partial x} = 0, \quad (x, y) \in [0, 1] \times [0, 1]

The goal is to learn the solution operator mapping initial conditions at :math:`y=0` to the full spatial domain. We use a PI-DeepONet architecture with a periodic feature transform on the trunk net to capture the physics of the problem.

Multi-Dimensional Search Space
------------------------------

We define a rich search space that covers:

1. **Architecture**:
   - ``branch_depth`` & ``trunk_depth``: Exploring the capacity of each sub-network (2 to 4 layers).
   - ``network_width``: Hidden layer size (64 to 256).
   - ``activation``: Comparing standard (``tanh``, ``sigmoid``, ``relu``) and modern (``silu``) functions.

2. **Optimization**:
   - ``learning_rate``: Log-uniform search between :math:`1e-4` and :math:`5e-3`.
   - ``weight_decay``: Regularization strength.

3. **Learning Rate Scheduler**:
   - ``scheduler_step`` & ``scheduler_gamma``: Parameter for the learning rate scheduler.

Implementation Details
----------------------

The HPO process requires three main components: a search space definition, a dataset builder, and a model builder.

1. Configuration Space
~~~~~~~~~~~~~~~~~~~~~~

We use ``ray.tune`` to define the distributions for our hyper-parameters. This dictionary tells the optimizer which values are allowed and how to sample them.

.. code-block:: python

    config_space = {
        # Architecture choices
        "branch_depth": tune.choice([2, 3, 4]),
        "trunk_depth": tune.choice([2, 3, 4]),
        "network_width": tune.choice([64, 128, 256]),
        "activation": tune.choice(["tanh", "relu", "sigmoid", "silu"]),
        
        # Optimizer parameters (continuous)
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        
        # Scheduler parameters
        "scheduler_step": tune.choice([5, 10, 15]),
        "scheduler_gamma": tune.uniform(0.9, 0.99),

        # Fixed parameters (not tuned)
        "batch_size": 32,
        "training_samples": 500,
        "problem_dim": 2, 
    }

2. Dataset Builder
~~~~~~~~~~~~~~~~~~

The ``dataset_builder`` is responsible for generating the training and validation data for each trial. To optimize the network correctly using a relative L2 loss, we need supervised targets. For the advection equation :math:`u_y + u_x = 0`, the analytical solution is :math:`u(x, y) = v(x - y)`, where :math:`v(x)` is the initial condition at :math:`y=0`. We use ``scipy.interpolate.interp1d`` to shift the discrete GRF branch inputs and compute the exact targets.

.. code-block:: python

    from scipy.interpolate import interp1d

    def generate_advection_data(num_functions, num_eval_points=50, num_trunk_points=1000):
        # 1. Generate Branch Inputs (Initial conditions at y=0)
        func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)
        # Use endpoint=False for periodic domain to avoid duplicate 0.0/1.0
        eval_pts = np.linspace(0, 1, num=num_eval_points, endpoint=False)[:, None]
        features = func_space.random(num_functions)
        # Add a constant offset to ensure the norm is never zero (avoids division by zero in relative loss)
        branch_inputs = func_space.eval_batch(features, eval_pts) + 1.0
        
        # 2. Generate Trunk Inputs (Random points in the domain)
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
        trunk_inputs = geom.uniform_points(num_trunk_points)
        num_actual_trunk_points = trunk_inputs.shape[0]
        
        # 3. Generate Targets (Analytical solution u(x, y) = v((x - y) mod 1))
        x_coords, y_coords = trunk_inputs[:, 0], trunk_inputs[:, 1]
        shifted_x = (x_coords - y_coords) % 1.0
        
        # Initialize targets based on actual points to avoid broadcasting errors
        targets = np.zeros((num_functions, num_actual_trunk_points))
        eval_pts_flat = eval_pts.flatten()
        
        # Append the first point to the end at x=1.0 to close the periodic loop
        x_vals = np.append(eval_pts_flat, 1.0)
        
        for i in range(num_functions):
            v_vals = np.append(branch_inputs[i], branch_inputs[i][0])
            interpolator = interp1d(x_vals, v_vals, kind='cubic', bounds_error=False, fill_value="extrapolate")
            targets[i] = interpolator(shifted_x)
            
        return branch_inputs, trunk_inputs, targets

    def dataset_builder(config):
        X_b_train, X_t_train, y_train = generate_advection_data(config["training_samples"])
        X_b_test, X_t_test, y_test = generate_advection_data(100)

        class PDEData:
            def __init__(self, bs):
                self.train_loader = DataLoader(
                    PDECartesianDataset(X_b_train, X_t_train, y_train), 
                    batch_size=bs, shuffle=True, collate_fn=deeponet_collate_fn
                )
                self.val_loader = DataLoader(
                    PDECartesianDataset(X_b_test, X_t_test, y_test), 
                    batch_size=bs, shuffle=False, collate_fn=deeponet_collate_fn
                )
                self.test_loader = self.val_loader

        return PDEData(config["batch_size"])

3. Model Builder
~~~~~~~~~~~~~~~~

The ``model_builder`` constructs the DeepONet model. It dynamically adjusts all the hyper-parameters based on the current configuration trial.

.. code-block:: python

    def model_builder(config):
        p = config["network_width"]
        model = dde.nn.DeepONetCartesianProd(
            [50] + [p] * config["branch_depth"] + [p],
            [5] + [p] * config["trunk_depth"] + [p],
            config["activation"], 
            "Glorot normal"
        )

        # Apply a periodic feature transform to embed domain knowledge
        def periodic(x):
            xt, yt = x[:, :1] * 2 * np.pi, x[:, 1:]
            return torch.cat([
                torch.cos(xt), torch.sin(xt), 
                torch.cos(2 * xt), torch.sin(2 * xt), yt
            ], 1)

        model.apply_feature_transform(periodic)
        return model

4. Loss Function
~~~~~~~~~~~~~~~~

For operator learning, we typically use the **Relative L2 Loss** (``LprelLoss``). This loss function is scale-invariant, making it ideal for comparing performance across different PDE scales and configurations during the HPO process.

.. code-block:: python

    loss_fn = LprelLoss(p=2, size_mean=True)

Advanced HPO Configuration
--------------------------

The ``tune_hyperparameters`` function orchestrates the search. Here is a detailed breakdown of its execution parameters:

.. list-table:: tune_hyperparameters Parameters
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``config_space``
     - The dictionary defining the search distributions.
   * - ``model_builder``
     - Function to instantiate the PyTorch model for each trial.
   * - ``dataset_builder``
     - Function to provide the data loaders.
   * - ``loss_fn``
     - The metric to be minimized (objective function).
   * - ``num_samples``
     - Total number of trials (combinations of hyper-parameters) to test.
   * - ``max_epochs``
     - The budget for training each trial.
   * - ``grace_period``
     - Number of epochs before a trial can be stopped by the ASHA scheduler.
   * - ``reduction_factor``
     - Controls the pruning aggressiveness of the ASHA scheduler.
   * - ``runs_per_cpu``
     - Fraction of CPU resources assigned to each concurrent trial.
   * - ``runs_per_gpu``
     - Fraction of GPU resources assigned to each trial (set to 0.0 for CPU-only).
   * - ``default_hyper_params``
     - A list of initial "best guess" configurations to try first.

Running the Example
-------------------

To run this HPO demo, ensure you have ``HyperNOs`` and ``ray[tune]`` installed. You can execute the script directly from the command line:

.. code-block:: bash

    python examples/operator/advection_2d_hpo.py

The script will launch the Ray Tune dashboard where you can monitor the progress of each trial in real-time. Once finished, it will print the best configuration and the corresponding relative loss.

Full Code
---------

.. literalinclude:: ../../../examples/operator/advection_2d_hpo.py
    :language: python
