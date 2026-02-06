"""
Tests for gagf/rnns/main.py

This module tests that the main() entry point runs successfully with minimal
configuration. Tests are only run when MAIN_TEST_MODE=1 environment variable
is set to avoid long-running tests in regular CI.

Expected runtime: < 1 minute with MAIN_TEST_MODE=1

Usage:
    MAIN_TEST_MODE=1 pytest test/test_rnns_main.py -v
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Check for MAIN_TEST_MODE
MAIN_TEST_MODE = os.environ.get("MAIN_TEST_MODE", "0") == "1"


@pytest.fixture
def temp_run_dir():
    """Create a temporary directory for run outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config_path():
    """Return the path to the test config file."""
    return Path(__file__).parent / "test_rnns_config.yaml"


@pytest.fixture
def mock_plots():
    """Mock all plot functions to skip visualization."""
    with patch("gagf.rnns.main.produce_plots_1d") as mock_1d, patch(
        "gagf.rnns.main.produce_plots_2d"
    ) as mock_2d, patch("gagf.rnns.main.produce_plots_D3") as mock_d3, patch(
        "matplotlib.pyplot.savefig"
    ) as mock_savefig, patch("matplotlib.pyplot.close") as mock_close:
        yield {
            "produce_plots_1d": mock_1d,
            "produce_plots_2d": mock_2d,
            "produce_plots_D3": mock_d3,
            "savefig": mock_savefig,
            "close": mock_close,
        }


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_with_config_file(temp_run_dir, test_config_path, mock_plots):
    """
    Test main() by loading the test config file.

    This tests what happens when you run `python main.py --config test_rnns_config.yaml`.
    """
    from gagf.rnns.main import load_config, main

    # Load the test config
    config = load_config(str(test_config_path))

    # Patch the setup_run_directory to use our temp directory
    with patch("gagf.rnns.main.setup_run_directory") as mock_setup:
        mock_setup.return_value = temp_run_dir

        # Run main
        main(config)

    # Verify that plotting was skipped via mocking
    # (we mock produce_plots_1d since we use dimension=1 in test config)
    mock_plots["produce_plots_1d"].assert_called_once()


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_train_single_run_1d(temp_run_dir, mock_plots):
    """
    Test train_single_run() directly with a minimal 1D config.
    """
    from gagf.rnns.main import train_single_run

    # Create minimal config programmatically
    config = {
        "data": {
            "dimension": 1,
            "p": 5,
            "k": 2,
            "batch_size": 32,
            "seed": 42,
            "template_type": "onehot",
            "mode": "sampled",
            "num_samples": 100,
        },
        "model": {
            "model_type": "SequentialMLP",
            "hidden_dim": 10,
            "init_scale": 1e-2,
            "return_all_outputs": False,
            "transform_type": "quadratic",
        },
        "training": {
            "mode": "offline",
            "epochs": 2,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "degree": None,
            "scaling_factor": -3,
            "grad_clip": 0.1,
            "verbose_interval": 1,
            "save_param_interval": None,
            "reduction_threshold": None,
        },
        "device": "cpu",
        "analysis": {
            "checkpoints": [0.0, 1.0],
        },
    }

    # Run training
    results = train_single_run(config, run_dir=temp_run_dir)

    # Verify results
    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert "training_time" in results
    assert results["final_train_loss"] > 0
    assert results["final_val_loss"] > 0


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_train_single_run_with_quadratic_rnn(temp_run_dir, mock_plots):
    """
    Test train_single_run() with QuadraticRNN model type.
    """
    from gagf.rnns.main import train_single_run

    config = {
        "data": {
            "dimension": 1,
            "p": 5,
            "k": 2,
            "batch_size": 32,
            "seed": 42,
            "template_type": "onehot",
            "mode": "sampled",
            "num_samples": 100,
        },
        "model": {
            "model_type": "QuadraticRNN",
            "hidden_dim": 10,
            "init_scale": 1e-2,
            "return_all_outputs": False,
            "transform_type": "quadratic",
        },
        "training": {
            "mode": "offline",
            "epochs": 2,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "degree": None,
            "scaling_factor": -3,
            "grad_clip": 0.1,
            "verbose_interval": 1,
            "save_param_interval": None,
            "reduction_threshold": None,
        },
        "device": "cpu",
        "analysis": {
            "checkpoints": [0.0, 1.0],
        },
    }

    results = train_single_run(config, run_dir=temp_run_dir)

    assert results["final_train_loss"] > 0
    assert results["final_val_loss"] > 0


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_load_config(test_config_path):
    """Test that load_config correctly loads the YAML file."""
    from gagf.rnns.main import load_config

    config = load_config(str(test_config_path))

    # Verify expected keys exist
    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert "device" in config
    assert "analysis" in config

    # Verify some specific values from our test config
    assert config["data"]["dimension"] == 1
    assert config["data"]["p"] == 5
    assert config["training"]["epochs"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
