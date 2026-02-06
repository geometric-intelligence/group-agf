"""
Tests for group_agf/binary_action_learning/main.py

This module tests that the main() entry point runs successfully with minimal
configuration. Tests are only run when MAIN_TEST_MODE=1 environment variable
is set to avoid long-running tests in regular CI.

Expected runtime: < 1 minute with MAIN_TEST_MODE=1

Usage:
    MAIN_TEST_MODE=1 pytest test/test_bal_main.py -v
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check for MAIN_TEST_MODE
MAIN_TEST_MODE = os.environ.get("MAIN_TEST_MODE", "0") == "1"

# Add test directory to path and register test_default_config BEFORE any imports
# that might trigger loading of group_agf.binary_action_learning.main
_test_dir = Path(__file__).parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))

# Import and register test_default_config as 'default_config' in sys.modules
# This must happen before any import of group_agf.binary_action_learning.main
import test_default_config  # noqa: E402

sys.modules["default_config"] = test_default_config


@pytest.fixture
def temp_save_dir():
    """Create a temporary directory for saving model outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_wandb():
    """Mock wandb to avoid actual logging."""
    mock_run = MagicMock()
    mock_run.id = "test_run_123"
    mock_run.name = "test_run"

    mock_config = MagicMock()

    with (
        patch("wandb.init") as mock_init,
        patch("wandb.config", mock_config),
        patch("wandb.run", mock_run),
        patch("wandb.log") as mock_log,
        patch("wandb.finish") as mock_finish,
        patch("wandb.Image") as mock_image,
    ):
        mock_init.return_value = mock_run
        mock_image.return_value = MagicMock()
        yield {
            "init": mock_init,
            "config": mock_config,
            "run": mock_run,
            "log": mock_log,
            "finish": mock_finish,
            "image": mock_image,
        }


@pytest.fixture
def mock_plots():
    """Mock plot functions to skip visualization."""
    with (
        patch("group_agf.binary_action_learning.plot.plot_loss_curve") as mock_loss,
        patch("group_agf.binary_action_learning.plot.plot_training_power_over_time") as mock_power,
        patch("group_agf.binary_action_learning.plot.plot_neuron_weights") as mock_weights,
        patch("group_agf.binary_action_learning.plot.plot_model_outputs") as mock_outputs,
    ):
        # Return mock figure objects
        mock_fig = MagicMock()
        mock_loss.return_value = mock_fig
        mock_power.return_value = mock_fig
        mock_weights.return_value = mock_fig
        mock_outputs.return_value = mock_fig
        yield {
            "loss": mock_loss,
            "power": mock_power,
            "weights": mock_weights,
            "outputs": mock_outputs,
        }


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_run_cn_group(temp_save_dir, mock_wandb, mock_plots):
    """
    Test main_run() with a minimal cyclic group (C5) configuration.

    This tests the core training pipeline without the full main() iteration.
    """
    # Update test_default_config to use temp directory
    test_default_config.model_save_dir = temp_save_dir + "/"

    from group_agf.binary_action_learning.main import main_run

    # Create minimal config for C5 group
    config = {
        "group_name": "cn",
        "group_size": 5,
        "group_n": 5,
        "epochs": 2,
        "batch_size": 32,
        "hidden_factor": 2,
        "init_scale": 1e-2,
        "lr": 0.01,
        "mom": 0.9,
        "optimizer_name": "SGD",
        "seed": 42,
        "verbose_interval": 1,
        "model_save_dir": temp_save_dir + "/",
        "dataset_fraction": 1.0,
        "powers": [0, 10, 5],
        "fourier_coef_diag_values": [0, 10, 5],
        "power_logscale": False,
        "resume_from_checkpoint": False,
        "checkpoint_interval": 1000,
        "checkpoint_path": None,
        "template_type": "one_hot",
        "run_start_time": "test_run",
    }

    # Run the training
    main_run(config)

    # Verify wandb was called (at least once - may be called multiple times in some flows)
    assert mock_wandb["init"].call_count >= 1
    assert mock_wandb["finish"].call_count >= 1


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_entry_point(temp_save_dir, mock_wandb, mock_plots):
    """
    Test the full main() entry point with mocked default_config.

    This tests what happens when you run `python main.py`.
    """
    # Update model_save_dir in test_default_config to use temp directory
    test_default_config.model_save_dir = temp_save_dir + "/"

    # Import the main module (default_config is already mocked at module level)
    from group_agf.binary_action_learning.main import main

    # Run main()
    main()

    # Verify wandb was called (at least once for the single config combination)
    assert mock_wandb["init"].call_count >= 1
    assert mock_wandb["finish"].call_count >= 1


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_run_with_adam_optimizer(temp_save_dir, mock_wandb, mock_plots):
    """Test main_run() with Adam optimizer."""
    # Update test_default_config to use temp directory
    test_default_config.model_save_dir = temp_save_dir + "/"

    from group_agf.binary_action_learning.main import main_run

    config = {
        "group_name": "cn",
        "group_size": 5,
        "group_n": 5,
        "epochs": 2,
        "batch_size": 32,
        "hidden_factor": 2,
        "init_scale": 1e-2,
        "lr": 0.001,
        "mom": 0.9,
        "optimizer_name": "Adam",
        "seed": 42,
        "verbose_interval": 1,
        "model_save_dir": temp_save_dir + "/",
        "dataset_fraction": 1.0,
        "powers": [0, 10, 5],
        "fourier_coef_diag_values": [0, 10, 5],
        "power_logscale": False,
        "resume_from_checkpoint": False,
        "checkpoint_interval": 1000,
        "checkpoint_path": None,
        "template_type": "one_hot",
        "run_start_time": "test_run",
    }

    main_run(config)

    assert mock_wandb["init"].call_count >= 1
    assert mock_wandb["finish"].call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
