"""
Minimal test configuration for binary_action_learning/main.py

This module mimics the structure of group_agf/binary_action_learning/default_config.py
but with minimal values for fast testing.
"""

# Dataset Parameters
group_name = "cn"
group_n = [5]  # Small cyclic group C5
template_type = "one_hot"

powers = {
    "cn": [[0, 10, 5]],  # Single power configuration
    "cnxcn": [[0, 10, 5]],
    "dihedral": [[0, 5, 0]],
    "octahedral": [[0, 10, 0, 0, 0]],
    "A5": [[0, 10, 0, 0, 0]],
}

# Model Parameters
hidden_factor = [2]  # Small hidden size

# Learning Parameters
seed = [42]
init_scale = {
    "cn": [1e-2],
    "cnxcn": [1e-2],
    "dihedral": [1e-2],
    "octahedral": [1e-3],
    "A5": [1e-3],
}
lr = {
    "cn": [0.01],
    "cnxcn": [0.01],
    "dihedral": [0.01],
    "octahedral": [0.001],
    "A5": [0.001],
}
mom = [0.9]
optimizer_name = ["SGD"]  # Simple optimizer
epochs = [2]  # Minimal epochs
verbose_interval = 1
checkpoint_interval = 1000
batch_size = [32]  # Small batch size

# Plotting parameters
power_logscale = False

# Checkpoint settings
resume_from_checkpoint = False
checkpoint_epoch = 0

# cnxcn specific parameters
image_length = [3]  # Small image for cnxcn

dataset_fraction = {
    "cn": 1.0,
    "cnxcn": 1.0,
    "dihedral": 1.0,
    "octahedral": 1.0,
    "A5": 1.0,
}

# Use temp directory - will be overwritten in tests
model_save_dir = "/tmp/test_bal/"
