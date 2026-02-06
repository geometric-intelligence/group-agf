"""
Tests for src/main.py

This module tests that the main() entry point runs successfully with minimal
configuration for all supported groups: cn (C_10), cnxcn (C_4 x C_4),
dihedral (D3), octahedral, and A5.

Tests are only run when MAIN_TEST_MODE=1 environment variable is set
to avoid long-running tests in regular CI.

Expected runtime: < 2 minutes with MAIN_TEST_MODE=1

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
def mock_plots_cn():
    """Mock plot functions for cn (1D) tests - skip visualization only."""
    import src.main  # noqa: F401

    with (
        patch("src.main.produce_plots_1d") as mock_1d,
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.close") as mock_close,
    ):
        yield {
            "produce_plots_1d": mock_1d,
            "savefig": mock_savefig,
            "close": mock_close,
        }


@pytest.fixture
def mock_plots_cnxcn():
    """Mock plot functions for cnxcn (2D) tests - skip visualization only."""
    import src.main  # noqa: F401

    with (
        patch("src.main.produce_plots_2d") as mock_2d,
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.close") as mock_close,
    ):
        yield {
            "produce_plots_2d": mock_2d,
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
def test_main_c10(temp_run_dir, mock_plots_cn):
    """Test main() with C_10 cyclic group config."""
    from src.main import load_config, train_single_run

    config = load_config(str(CONFIG_FILES["c10"]))
    results = train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0
    mock_plots_cn["produce_plots_1d"].assert_called_once()


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_c4x4(temp_run_dir, mock_plots_cnxcn):
    """Test main() with C_4 x C_4 product group config."""
    from src.main import load_config, train_single_run

    config = load_config(str(CONFIG_FILES["c4x4"]))
    results = train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0
    mock_plots_cnxcn["produce_plots_2d"].assert_called_once()


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_d3(temp_run_dir, mock_savefig):
    """Test main() with D3 dihedral group config.

    Does NOT mock produce_plots_D3 so the full plotting pipeline
    (including dataset generation with the correct group) is exercised.
    """
    from src.main import load_config, train_single_run

    config = load_config(str(CONFIG_FILES["d3"]))
    results = train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_octahedral(temp_run_dir, mock_savefig):
    """Test main() with octahedral group config.

    Does NOT mock produce_plots_D3 so the full plotting pipeline
    (including dataset generation with the correct group) is exercised.
    This ensures the octahedral group (order 24) is passed correctly
    and not confused with D3 (order 6).
    """
    from src.main import load_config, train_single_run

    config = load_config(str(CONFIG_FILES["octahedral"]))
    results = train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0


@pytest.mark.skipif(not MAIN_TEST_MODE, reason="Only run with MAIN_TEST_MODE=1")
def test_main_a5(temp_run_dir, mock_savefig):
    """Test main() with A5 (icosahedral) group config.

    Does NOT mock produce_plots_D3 so the full plotting pipeline
    (including dataset generation with the correct group) is exercised.
    This ensures the A5 group (order 60) is passed correctly
    and not confused with D3 (order 6).
    """
    from src.main import load_config, train_single_run

    config = load_config(str(CONFIG_FILES["a5"]))
    results = train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
