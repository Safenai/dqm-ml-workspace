"""Tests for README examples.

Tests that the examples referenced in README.md are valid and working.
"""

import subprocess
from pathlib import Path

import pytest

EXAMPLES_DIR = Path("examples")


class TestReadmeExamples:
    """Test examples referenced in README.md."""

    @pytest.fixture
    def examples_path(self) -> Path:
        """Path to examples directory."""
        return EXAMPLES_DIR

    def test_examples_directory_exists(self, examples_path: Path) -> None:
        """Verify examples directory exists."""
        assert examples_path.exists(), "examples/ directory not found"
        assert examples_path.is_dir(), "examples/ is not a directory"

    def test_jupyter_notebook_exists(self, examples_path: Path) -> None:
        """Test that jupyter notebook referenced in README exists."""
        notebook_path = examples_path / "multiple_metrics_tests_v2.ipynb"
        assert notebook_path.exists(), f"Notebook not found: {notebook_path}"

    def test_jupyter_notebook_is_valid(self, examples_path: Path) -> None:
        """Test that jupyter notebook has valid structure."""
        import json

        notebook_path = examples_path / "multiple_metrics_tests_v2.ipynb"
        with Path(notebook_path).open() as f:
            notebook = json.load(f)

        assert "cells" in notebook, "Notebook missing 'cells' key"
        assert len(notebook["cells"]) > 0, "Notebook has no cells"
        assert notebook["cells"][0]["cell_type"] == "markdown", "First cell should be markdown"

    def test_completeness_script_exists(self, examples_path: Path) -> None:
        """Test that completeness.py script exists."""
        script_path = examples_path / "script" / "completeness.py"
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_completeness_config_exists(self, examples_path: Path) -> None:
        """Test that completeness.yaml config exists."""
        config_path = examples_path / "config" / "completeness.yaml"
        assert config_path.exists(), f"Config not found: {config_path}"

    def test_completeness_config_valid(self, examples_path: Path) -> None:
        """Test that completeness.yaml is valid YAML with expected structure."""
        import yaml

        config_path = examples_path / "config" / "completeness.yaml"
        with Path(config_path).open() as f:
            config = yaml.safe_load(f)

        assert "config" in config, "Config missing 'config' key"
        assert "dataloaders" in config["config"], "Config missing dataloaders"
        assert "metrics_processor" in config["config"], "Config missing metrics_processor"

    def test_all_example_configs_valid(self, examples_path: Path) -> None:
        """Test that all example configs are valid YAML."""
        import yaml

        config_dir = examples_path / "config"
        for config_file in config_dir.glob("*.yaml"):
            with Path(config_file).open() as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {config_file}: {e}")

    def test_completeness_script_runs(self, examples_path: Path) -> None:
        """Test that completeness.py script can be executed."""
        script_path = examples_path / "script" / "completeness.py"

        result = subprocess.run(
            ["python3", str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"Script failed with code {result.returncode}")

    def test_example_configs_in_test_job_cli(self) -> None:
        """Verify test_job_cli.py tests the example configs."""
        test_file = Path("tests/cli/test_job_cli.py")
        assert test_file.exists(), "test_job_cli.py not found"

        with Path(test_file).open() as f:
            content = f.read()

        assert "examples/config/completeness.yaml" in content, "Completeness example not tested"
        assert "examples/config/representativness.yaml" in content, "Representativeness example not tested"
