"""Test utility fixtures for DQM-ML tests.

This module provides fixtures used in unit and integration tests.
"""

from collections.abc import Generator
from pathlib import Path
import shutil
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def temp_output_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Provide a temporary output directory and clean it up.

    Args:
        tmp_path: Pytest's temporary directory fixture.

    Returns:
        Path to a temporary output subdirectory.
    """
    path = tmp_path / "output"
    yield path
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def mock_parquet_dataset():
    """Provide a mocked ParquetDataset for testing.

    Yields:
        Mocked ParquetDataset class.
    """
    with patch("pyarrow.parquet.ParquetDataset") as mock:
        yield mock


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Provide a sample DataFrame for testing.

    Returns:
        A pandas DataFrame with columns 'a' and 'b'.
    """
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
