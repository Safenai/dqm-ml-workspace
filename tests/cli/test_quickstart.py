"""E2E tests for quickstart.md examples.

This module verifies the examples in quickstart.md work correctly.
It tests both the CLI and configuration examples.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from dqm_ml_job.cli import execute


FIXTURES_DIR = Path("tests/fixtures/getting_started")


@pytest.fixture
def fixtures_path() -> Path:
    """Return the path to getting started fixtures."""
    return FIXTURES_DIR


class TestQuickstartCLI:
    """Tests for CLI examples in quickstart.md."""

    def test_completeness_cli_with_config(self, fixtures_path: Path) -> None:
        """Test CLI with getting_started completeness config works.

        This tests the example:
        dqm-ml process -p tests/fixtures/getting_started/completeness.yaml

        The config expects to be run from the fixtures directory or the path
        needs to be adjusted. For this test, we'll change to the fixtures directory.
        """
        config_path = fixtures_path / "completeness.yaml"

        # Verify config file exists
        assert config_path.exists(), f"Config file not found: {config_path}"

        # Verify data file exists
        data_path = fixtures_path / "data.csv"
        assert data_path.exists(), f"Data file not found: {data_path}"

        # Run CLI from fixtures directory so relative path works
        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(fixtures_path)
            # Run CLI - should complete without error
            execute(["-p", "completeness.yaml"])
        finally:
            os.chdir(original_cwd)


class TestQuickstartDataFiles:
    """Test data files for quickstart examples."""

    def test_data_csv_structure(self, fixtures_path: Path) -> None:
        """Verify the data.csv has expected structure."""
        data_path = fixtures_path / "data.csv"
        df = pd.read_csv(data_path)

        assert list(df.columns) == ["name", "age", "score"]
        assert len(df) == 4

    def test_data_csv_has_nulls(self, fixtures_path: Path) -> None:
        """Verify the data.csv has null values for testing completeness."""
        data_path = fixtures_path / "data.csv"
        df = pd.read_csv(data_path)

        # Check that there are null values (for testing completeness)
        assert df["name"].isna().sum() == 1  # Diana has no name
        assert df["age"].isna().sum() == 1  # Diana has no age

    def test_completeness_config_valid(self, fixtures_path: Path) -> None:
        """Verify the completeness.yaml has valid configuration."""
        config_path = fixtures_path / "completeness.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify config structure
        assert "config" in config
        assert "dataloaders" in config["config"]
        assert "metrics_processor" in config["config"]
        assert "outputs" in config["config"]

        # Verify completeness metric config
        completeness = config["config"]["metrics_processor"]["completeness"]
        assert completeness["type"] == "completeness"
        assert "name" in completeness["input_columns"]
        assert "age" in completeness["input_columns"]
        assert "score" in completeness["input_columns"]
