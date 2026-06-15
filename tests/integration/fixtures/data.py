"""Data fixtures for DQM-ML tests.

This module provides fixtures for generating and loading test data.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.utils.files import write_path_list_to_parquet
from tests.utils.plots import plot_histograms


# COMPATIBILITY : Lazy import fiftyone.zoo to avoid Python 3.10/3.11 compatibility issues
# (glob2 package has SyntaxError on older Python versions)
def _get_fiftyone():
    import fiftyone.zoo as foz

    return foz


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
    with class information for domain gap testing.

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
        source_table = pq.read_table(source_path)
        if source_table.num_rows > 0:
            sample_path = source_table.column("image_path")[0].as_py()
            if Path(sample_path).exists():
                print("Parquet found, images available, no need to recreate")
                return [source_path, target_path]
        print("Parquet found but images missing, re-downloading")

    foz = _get_fiftyone()
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
    annotations_path = Path.home() / "fiftyone" / "coco-2017" / "raw" / "instances_train2017.json"

    import json

    with Path(annotations_path).open() as f:
        coco_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    image_to_class = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id not in image_to_class:
            image_to_class[img_id] = categories[cat_id]

    files = sorted(Path(dataset_path).glob("*.jpg"))

    source_files = files[: len(files) // 2]
    target_files = files[len(files) // 2 :]

    source_classes = []
    source_paths = []
    for f in source_files:
        img_id = int(f.stem)
        class_name = image_to_class.get(img_id, "unknown")
        source_classes.append(class_name)
        source_paths.append(f)

    target_classes = []
    target_paths = []
    for f in target_files:
        img_id = int(f.stem)
        class_name = image_to_class.get(img_id, "unknown")
        target_classes.append(class_name)
        target_paths.append(f)

    write_path_list_to_parquet(source_paths, source_path, source_classes)
    write_path_list_to_parquet(target_paths, target_path, target_classes)

    return [source_path, target_path]


@pytest.fixture(scope="session")
def coco_data_500(coco_data: list[Path], output_path: Path) -> None:
    """Create 500-image parquet fixtures from the existing 1000-image parquets.

    Args:
        coco_data: Fixture providing paths to source_1000.parquet and target_1000.parquet.
        output_path: Path to test output data directory.
    """
    source_500 = output_path / "source_500.parquet"
    target_500 = output_path / "target_500.parquet"

    if source_500.exists() and target_500.exists():
        return

    for src, dst in [(coco_data[0], source_500), (coco_data[1], target_500)]:
        table = pq.read_table(src).slice(0, 500)
        pq.write_table(table, dst)


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
def diversity_data(test_path: str) -> Any:
    """Generate synthetic diversity test data.

    Creates parquet file with integer categorical data drawn from
    uniform discrete distributions at different uniqueness levels
    for streaming pipeline tests.

    Args:
        test_path: Path to the tests directory.

    Returns:
        None.
    """
    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    path = Path(gen_path) / "diversity.parquet"

    if path.exists():
        return

    rng = np.random.default_rng(42)
    n = 200
    pa_table = pa.table(
        {
            "column_2": rng.integers(0, 40, size=n).astype(float),
            "column_4": rng.integers(0, 20, size=n).astype(float),
            "column_6": rng.integers(0, 8, size=n).astype(float),
        }
    )
    pq.write_table(pa_table, path)


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
