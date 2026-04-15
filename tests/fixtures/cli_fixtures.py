"""CLI test fixtures for DQM-ML tests.

This module provides fixtures for CLI tests.
"""

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to getting started fixtures directory.

    Returns:
        Path to tests/fixtures/getting_started.
    """
    return Path(__file__).parent / "getting_started"


@pytest.fixture(scope="session")
def coco_data_dir() -> Path:
    """Return path to test output data directory.

    Returns:
        Path to tests/outputs/data.
    """
    return Path(__file__).parent.parent / "outputs" / "data"


@pytest.fixture(scope="session")
def coco_parquet_path(coco_data_dir: Path) -> Path:
    """Return path to source COCO parquet with class column.

    Args:
        coco_data_dir: Path to test output data directory.

    Returns:
        Path to source_1000.parquet.
    """
    return coco_data_dir / "source_1000.parquet"


@pytest.fixture(scope="session")
def all_classes(coco_parquet_path: Path, coco_data: Any) -> list[str]:
    """Get all unique classes from the parquet file.

    Args:
        coco_parquet_path: Path to COCO parquet file.
        coco_data: Fixture providing COCO dataset.

    Returns:
        Sorted list of unique class names.
    """
    table = pq.read_table(coco_parquet_path)
    df = table.to_pandas()
    classes = sorted(df["class"].unique().tolist())
    return classes


@pytest.fixture(scope="session")
def split_by_config_path(fixtures_dir: Path) -> Path:
    """Return path to split_by config for top 10 classes.

    Args:
        fixtures_dir: Path to fixtures directory.

    Returns:
        Path to domain_gap_split_top10.yaml.
    """
    return fixtures_dir / "domain_gap_split_top10.yaml"
