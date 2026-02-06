"""
Tests for src/main.py

This module tests that the main() entry point runs successfully with minimal
configuration for all supported groups: cn (C_10), cnxcn (C_4 x C_4),
dihedral (D3), octahedral, and A5.

Tests are only run when MAIN_TEST_MODE=1 environment variable is set
to avoid long-running tests in regular CI.

Expected runtime: < 1 minute with MAIN_TEST_MODE=1

Usage:
    MAIN_TEST_MODE=1 pytest test/test_main.py -v
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Check for MAIN_TEST_MODE
MAIN_TEST_MODE = os.environ.get("MAIN_TEST_MODE", "0") == "1"

# Paths to test config files
TEST_DIR = Path(__file__).parent
CONFIG_FILES = {
    "c10": TEST_DIR / "test_config_c10.yaml",
    "c4x4": TEST_DIR / "test_config_c4x4.yaml",
    "d3": TEST_DIR / "test_config_d3.yaml",
    "octahedral": TEST_DIR / "test_config_octahedral.yaml",
    "a5": TEST_DIR / "test_config_a5.yaml",
}


@pytest.fixture
def temp_run_dir():
    """Create a temporary directory for run outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_all_plots():
    """Mock all produce_plots_* and plt.savefig/close to skip visualization entirely."""
    import src.main  # noqa: F401

    with (
        patch("src.main.produce_plots_1d") as mock_1d,
        patch("src.main.produce_plots_2d") as mock_2d,
        patch("src.main.produce_plots_D3") as mock_d3,
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.close") as mock_close,
    ):
        yield {
            "produce_plots_1d": mock_1d,
            "produce_plots_2d": mock_2d,
            "produce_plots_D3": mock_d3,
            "savefig": mock_savefig,
            "close": mock_close,
        }


@pytest.fixture
def mock_savefig():
    """Mock only plt.savefig and plt.close so plotting code runs but files aren't saved."""
    with (
        patch("matplotlib.pyplot.savefig") as mock_sf,
        patch("matplotlib.pyplot.close") as mock_cl,
    ):
        yield {"savefig": mock_sf, "close": mock_cl}


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_load_config():
    """Test that load_config correctly loads a YAML file."""
    from src.main import load_config

    config = load_config(str(CONFIG_FILES["c10"]))

    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert "device" in config
    assert "analysis" in config
    assert config["data"]["group_name"] == "cn"
    assert config["data"]["p"] == 10
    assert config["training"]["epochs"] == 2


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_c10(temp_run_dir, mock_all_plots):
    """Test main() with C_10 cyclic group config."""
    from src.main import load_config, train_single_run

    config = load_config(str(CONFIG_FILES["c10"]))
    results = train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0
    mock_all_plots["produce_plots_1d"].assert_called_once()


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_c4x4(temp_run_dir, mock_all_plots):
    """Test main() with C_4 x C_4 product group config."""
    from src.main import load_config, train_single_run

    config = load_config(str(CONFIG_FILES["c4x4"]))
    results = train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0
    mock_all_plots["produce_plots_2d"].assert_called_once()


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_d3(temp_run_dir, mock_savefig):
    """Test main() with D3 dihedral group config.

    Full integration test: does NOT mock produce_plots_D3 so the entire
    plotting pipeline (TwoLayerNet eval data via group_dataset, power spectrum)
    is exercised. D3 (order 6) is the smallest group so this stays fast.
    This validates the TwoLayerNet-compatible eval data path in produce_plots_D3,
    which is shared by octahedral and A5 (mocked in their tests for speed).
    """
    from src.main import load_config, train_single_run

    config = load_config(str(CONFIG_FILES["d3"]))
    results = train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_octahedral(temp_run_dir, mock_all_plots):
    """Test main() with octahedral group config.

    Mocks produce_plots_D3 for speed (octahedral order=24, plotting is expensive).
    Training + data pipeline still fully exercised.
    """
    from src.main import load_config, train_single_run

    config = load_config(str(CONFIG_FILES["octahedral"]))
    results = train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0
    mock_all_plots["produce_plots_D3"].assert_called_once()


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_a5(temp_run_dir, mock_all_plots):
    """Test main() with A5 (icosahedral) group config.

    Mocks produce_plots_D3 for speed (A5 order=60, plotting is expensive).
    Training + data pipeline still fully exercised.
    """
    from src.main import load_config, train_single_run

    config = load_config(str(CONFIG_FILES["a5"]))
    results = train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0
    mock_all_plots["produce_plots_D3"].assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
