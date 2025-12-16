import os
from pathlib import Path
from typing import Any

import fiftyone.zoo as foz
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


@pytest.fixture(scope="session")
def test_path() -> str:
    # To point on test directory
    return str(Path(__file__).parent.resolve()) + os.sep


@pytest.fixture(scope="session")
def tests_config(test_path: str) -> Any:
    config_path = Path(test_path) / "expected" / "expected.yaml"

    # Load global unit tests configuration
    with Path.open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    return config


@pytest.fixture(scope="session")
def output_path(test_path: str) -> Path:
    path = Path(test_path) / "output"

    Path.mkdir(path, exist_ok=True)

    return path

    # yield path

    # files = list(Path(path).glob("*"))
    # for f in files:
    #     Path.unlink(f)
    # Path.rmdir(path)


def write_path_list_to_parquet(path_list: list[Path], save_path: Path) -> None:
    # We ignore type warning from mypy as we really want to convert Path to str
    path_list = [str(x) for x in path_list]  # type: ignore
    path_array = pa.array(path_list)
    path_table = pa.Table.from_arrays([path_array], names=["image_path"])
    pq.write_table(path_table, save_path)


@pytest.fixture(scope="session")
def coco_data(test_path: str) -> list[Path]:
    source_path = Path(test_path) / "data/source_1000.parquet"
    target_path = Path(test_path) / "data/target_1000.parquet"

    # temporary benchmark with dqm which doesn't support more than 512
    # images for klmvn and wasserstein
    source_500_path = Path(test_path) / "data/source_500.parquet"
    target_500_path = Path(test_path) / "data/target_500.parquet"

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

    source_500 = source[: len(source) // 2]
    target_500 = source[len(source) // 2 :]

    write_path_list_to_parquet(source, source_path)
    write_path_list_to_parquet(target, target_path)

    write_path_list_to_parquet(source_500, source_500_path)
    write_path_list_to_parquet(target_500, target_500_path)

    return [source_path, target_path]
