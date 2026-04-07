"""Configuration fixtures for DQM-ML tests.

This module provides fixtures for loading test configuration data.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture(scope="session")
def tests_config(test_path: str) -> Any:
    """Load test configuration from expected.yaml.

    Args:
        test_path: Path to the tests directory.

    Returns:
        Dictionary containing test configuration with expected scores and tolerances.
    """
    config_path = Path(test_path) / "integration" / "fixtures" / "expected" / "expected.yaml"

    with Path.open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    return config
