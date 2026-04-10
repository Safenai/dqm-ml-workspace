"""Path fixtures for DQM-ML tests.

This module provides fixtures for commonly used paths in tests.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_path() -> str:
    """Return the path to the tests directory.

    Returns:
        Absolute path to the tests directory with trailing separator.
    """
    return str(Path(__file__).parent.parent.parent.resolve()) + os.sep


@pytest.fixture(scope="session")
def output_path(test_path: str) -> Path:
    """Return the path to test output data directory.

    Creates the directory if it doesn't exist.

    Args:
        test_path: Path to the tests directory.

    Returns:
        Path to the output data directory.
    """
    path = Path(test_path) / "outputs" / "data"
    Path.mkdir(path, exist_ok=True, parents=True)
    return path
