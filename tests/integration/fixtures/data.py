"""Data fixtures for DQM-ML tests.

This module provides fixtures for generating and loading test data.
"""

from pathlib import Path
from typing import Any

import fiftyone.zoo as foz
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.utils.files import write_path_list_to_parquet
from tests.utils.plots import plot_histograms

OUTPUT_PLOTS = "outputs/plots"
OUTPUT_DATA = "outputs/data"

_rng = np.random.default_rng()


def _get_test_path() -> str:
    """Get the tests directory path."""
    return str(Path(__file__).parent.parent.resolve()) + "/"


@pytest.fixture(scope="session")
def coco_data(test_path: str) -> list[Path]:
    """Generate COCO dataset for domain gap tests.

    Downloads COCO-2017 dataset and creates source/target parquet files
    for domain gap testing.

    Args:
        test_path: Path to the tests directory.

    Returns:
        List containing paths to source and target parquet files.
    """
    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    source_path = Path(gen_path) / "source_1000.parquet"
    target_path = Path(gen_path) / "target_1000.parquet"

    if Path.exists(source_path) and Path.exists(target_path):
        print("Parquet found, no need to recreate")
        return [source_path, target_path]

    foz.download_zoo_dataset(
        "coco-2017",
        splits=["train"],
        classes=[
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
        ],
        max_samples=2000,
    )
    dataset_path = Path.home() / "fiftyone" / "coco-2017" / "train" / "data"

    files = sorted(Path(dataset_path).glob("*.jpg"))

    source = files[: len(files) // 2]
    target = files[len(files) // 2 :]

    write_path_list_to_parquet(source, source_path)
    write_path_list_to_parquet(target, target_path)

    return [source_path, target_path]


@pytest.fixture(scope="session")
def uniform_dist(test_path: str) -> Any:
    """Generate uniform distribution test data.

    Creates parquet file with uniformly distributed data and generates
    histogram plots.

    Args:
        test_path: Path to the tests directory.

    Returns:
        None.
    """
    plot_path = Path(test_path) / OUTPUT_PLOTS
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / f"{OUTPUT_DATA}/uniform_distribution.parquet"

    data_1 = _rng.uniform(0, 0.05, 1000000)
    data_2 = _rng.uniform(0.1, 1, 1000000)
    data_3 = _rng.uniform(0.2, 2, 1000000)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / OUTPUT_DATA,
        "uniform_distribution.parquet",
    )


@pytest.fixture(scope="session")
def not_uniform_dist(test_path: str) -> Any:
    """Generate non-uniform distribution test data.

    Creates parquet file with non-uniformly distributed data and generates
    histogram plots.

    Args:
        test_path: Path to the tests directory.

    Returns:
        None.
    """
    plot_path = Path(test_path) / OUTPUT_PLOTS
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / f"{OUTPUT_DATA}/not_uniform_distribution.parquet"

    a = _rng.uniform(0, 0.05, 500000)
    b = _rng.uniform(0.05, 2, 500000)
    data_1 = np.concatenate((a, b), axis=None)
    a = _rng.uniform(0.1, 1, 200000)
    b = _rng.uniform(0.1, 3, 800000)
    data_2 = np.concatenate((a, b), axis=None)
    a = _rng.uniform(0.2, 2, 200000)
    b = _rng.uniform(0.2, 3, 600000)
    c = _rng.uniform(0.2, 2, 200000)
    data_3 = np.concatenate((a, b, c), axis=None)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / OUTPUT_DATA,
        "not_uniform_distribution.parquet",
    )


@pytest.fixture(scope="session")
def normal_dist(test_path: str) -> Any:
    """Generate normal distribution test data.

    Creates parquet file with normally distributed data and generates
    histogram plots.

    Args:
        test_path: Path to the tests directory.

    Returns:
        None.
    """
    plot_path = Path(test_path) / OUTPUT_PLOTS
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / f"{OUTPUT_DATA}/normal_distribution.parquet"

    data_1 = _rng.normal(0, 0.5, 1000000)
    data_2 = _rng.normal(0, 5, 1000000)
    data_3 = _rng.normal(0, 50, 1000000)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / OUTPUT_DATA,
        "normal_distribution.parquet",
    )


@pytest.fixture(scope="session")
def not_normal_dist(test_path: str) -> Any:
    """Generate non-normal distribution test data.

    Creates parquet file with non-normally distributed data (bimodal) and
    generates histogram plots.

    Args:
        test_path: Path to the tests directory.

    Returns:
        None.
    """
    plot_path = Path(test_path) / OUTPUT_PLOTS
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / f"{OUTPUT_DATA}/not_normal_distribution.parquet"

    a = _rng.normal(0, 0.5, 500000)
    b = _rng.normal(5, 0.5, 500000)
    data_1 = np.concatenate((a, b), axis=None)
    a = _rng.normal(0, 5, 500000)
    b = _rng.normal(50, 5, 500000)
    data_2 = np.concatenate((a, b), axis=None)
    a = _rng.normal(0, 50, 500000)
    b = _rng.normal(500, 50, 500000)
    data_3 = np.concatenate((a, b), axis=None)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / OUTPUT_DATA,
        "not_normal_distribution.parquet",
    )
