import os
from pathlib import Path
from typing import Any

import fiftyone.zoo as foz
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ruamel.yaml
import yaml


def get_files_list(path: Path, pattern: str = "*.parquet") -> list[str]:
    files = sorted(Path(path).glob(pattern))
    path_list = [str(x) for x in files]
    return path_list


def plot_histograms(
    output_path: Path,
    columns: list[str],
    parquets_path: Path,
    parquet_pattern: str,
) -> None:
    path_list = get_files_list(parquets_path, pattern=parquet_pattern)

    for parquet_path in path_list:
        parquet_name = Path(parquet_path).stem

        table = pq.read_table(parquet_path)
        for column in columns:
            dist = table[column].to_numpy()

            fig, ax = plt.subplots()

            ax.hist(dist, bins=100)

            plot_name = f"hist_{parquet_name}_{column}.png"

            plt.title(f"Histogram of {column} in {parquet_name}")
            plt.savefig(f"{output_path}/{plot_name}")
            plt.close(fig)


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
    gen_path = Path(test_path) / "data_generated"
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
    plot_path = Path(test_path) / "data_generated/plot"
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / "data_generated/uniform_distribution.parquet"

    data_1 = np.random.uniform(0, 0.05, 1000000)
    data_2 = np.random.uniform(1, 0.1, 1000000)
    data_3 = np.random.uniform(2, 0.2, 1000000)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / "data_generated",
        "uniform_distribution.parquet",
    )

    return

    # Path.unlink(path)


@pytest.fixture(scope="session")
def not_uniform_dist(test_path: str) -> Any:
    plot_path = Path(test_path) / "data_generated/plot"
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / "data_generated/not_uniform_distribution.parquet"

    a = np.random.uniform(0, 0.05, 500000)
    b = np.random.uniform(2, 0.05, 500000)
    data_1 = np.concatenate((a, b), axis=None)
    a = np.random.uniform(1, 0.1, 200000)
    b = np.random.uniform(3, 0.1, 800000)
    data_2 = np.concatenate((a, b), axis=None)
    a = np.random.uniform(2, 0.2, 200000)
    b = np.random.uniform(3, 0.2, 600000)
    c = np.random.uniform(2, 0.2, 200000)
    data_3 = np.concatenate((a, b, c), axis=None)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / "data_generated",
        "not_uniform_distribution.parquet",
    )

    return


@pytest.fixture(scope="session")
def normal_dist(test_path: str) -> Any:
    plot_path = Path(test_path) / "data_generated/plot"
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / "data_generated/normal_distribution.parquet"

    mu, sigma = 0, 0.5
    data_1 = np.random.normal(mu, sigma, 1000000)
    mu, sigma = 0, 5
    data_2 = np.random.normal(mu, sigma, 1000000)
    mu, sigma = 0, 50
    data_3 = np.random.normal(mu, sigma, 1000000)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / "data_generated",
        "normal_distribution.parquet",
    )

    return


@pytest.fixture(scope="session")
def not_normal_dist(test_path: str) -> Any:
    plot_path = Path(test_path) / "data_generated/plot"
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / "data_generated/not_normal_distribution.parquet"

    mu, sigma = 0, 0.5
    a = np.random.normal(mu, sigma, 500000)
    mu, sigma = 5, 0.5
    b = np.random.normal(mu, sigma, 500000)
    data_1 = np.concatenate((a, b), axis=None)
    mu, sigma = 0, 5
    a = np.random.normal(mu, sigma, 500000)
    mu, sigma = 50, 5
    b = np.random.normal(mu, sigma, 500000)
    data_2 = np.concatenate((a, b), axis=None)
    mu, sigma = 0, 50
    a = np.random.normal(mu, sigma, 500000)
    mu, sigma = 500, 50
    b = np.random.normal(mu, sigma, 500000)
    data_3 = np.concatenate((a, b), axis=None)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / "data_generated",
        "not_normal_distribution.parquet",
    )

    return


def generate_pipeline(
    test_path: str,
    processor_name: str,
    output_category: str,
    parquets_path: Path,
    test_list: list[dict[str, str]],
    # Domain gap specific
    metric_name: str | None = None,
    parquet_source_path: Path | None = None,
) -> None:
    configs_path = Path(test_path) / "config_generated"
    output_path = Path(test_path) / "output"

    domain_gap_infer_params = {
        "fid": {"batch_size": 32, "width": 299, "height": 299},
        "klmvn_diag": {"batch_size": 10, "width": 20, "height": 20},
        "mmd_linear": {"batch_size": 10, "width": 224, "height": 224},
        "wasserstein_1d": {"batch_size": 18, "height": 299, "width": 299},
    }

    Path(configs_path).mkdir(exist_ok=True, parents=True)

    for test in test_list:
        parquet_path = parquets_path / test["parquet"]
        test_name = test["test_name"]

        if processor_name == "domain_gap":
            config_name = f"{processor_name}_{test_name}" if test_name != "" else f"{processor_name}_{metric_name}"
        elif processor_name in ["completeness", "visual_features"]:
            config_name = test_name
        else:
            config_name = f"{processor_name}_{test_name}"

        config_path = Path(f"{configs_path}/{config_name}.yaml")

        template_path = Path(test_path) / f"config_templates/{processor_name}.yaml"
        with Path(template_path).open() as file:
            config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(file)

        pipeline_config = config["pipeline_config"]
        if processor_name == "domain_gap":
            pipeline_config["dataloaders"]["source_dataset"]["path"] = str(parquet_source_path)
            pipeline_config["dataloaders"]["target_dataset"]["path"] = str(parquet_path)
            pipeline_config["metrics_processor"][processor_name]["DELTA"]["metric"] = metric_name
            for param in ["batch_size", "height", "width"]:
                pipeline_config["metrics_processor"]["image_embedding"]["infer"][param] = domain_gap_infer_params[
                    metric_name  # type: ignore
                ][param]
        else:
            pipeline_config["dataloaders"]["source_dataset"]["path"] = str(parquet_path)

        if "batch" in test_name:
            if processor_name == "representativeness":
                pipeline_config["dataloaders"]["source_dataset"]["batch_size"] = 50000
            if processor_name == "domain_gap":
                pipeline_config["dataloaders"]["source_dataset"]["batch_size"] = 50
                pipeline_config["dataloaders"]["target_dataset"]["batch_size"] = 50
            if processor_name == "completeness":
                pipeline_config["dataloaders"]["source_dataset"]["batch_size"] = 100
            if processor_name == "visual_features":
                pipeline_config["dataloaders"]["source_dataset"]["batch_size"] = 100

        if processor_name == "representativeness":
            if "uniform" in test_name:
                pipeline_config["metrics_processor"]["representativeness"]["distribution"] = "uniform"
            else:
                pipeline_config["metrics_processor"]["representativeness"]["distribution"] = "normal"

        if processor_name == "visual_features" and "path" in test_name:
            pipeline_config["metrics_processor"]["visual_features"]["input_columns"] = ["image_path"]

        if processor_name == "domain_gap" and "bytes" in test_name:
            pipeline_config["metrics_processor"]["image_embedding"]["DATA"]["image_column"] = "image_bytes"
            pipeline_config["metrics_processor"]["image_embedding"]["DATA"]["mode"] = "bytes"

        if processor_name == "domain_gap":
            pipeline_config["outputs"][output_category]["path_pattern"] = (
                f"{output_path!s}/metrics_{config_name}" + "_{}-{}.parquet"
            )
        else:
            pipeline_config["outputs"][output_category]["path_pattern"] = (
                f"{output_path!s}/metrics_{config_name}.parquet"
            )

        yaml_config = ruamel.yaml.YAML()
        yaml_config.indent(mapping=ind, sequence=ind, offset=bsi)
        with Path(config_path).open("w") as fp:
            yaml_config.dump(config, fp)


@pytest.fixture(scope="session")
def pipeline_representativeness(
    test_path: str,
    normal_dist: Any,
    not_normal_dist: Any,
    uniform_dist: Any,
    not_uniform_dist: Any,
) -> None:
    test_list = [
        {"test_name": "normal_distribution", "parquet": "normal_distribution.parquet"},
        {"test_name": "not_normal_distribution", "parquet": "not_normal_distribution.parquet"},
        {"test_name": "uniform_distribution", "parquet": "uniform_distribution.parquet"},
        {"test_name": "not_uniform_distribution", "parquet": "not_uniform_distribution.parquet"},
        {"test_name": "batch", "parquet": "normal_distribution.parquet"},
    ]

    generate_pipeline(
        processor_name="representativeness",
        parquets_path=Path(test_path) / "data_generated",
        test_list=test_list,
        output_category="metrics",
        test_path=test_path,
    )


@pytest.fixture(scope="session")
def pipeline_completeness(
    test_path: str,
) -> None:
    test_list = [
        {"test_name": "completeness", "parquet": "completeness.parquet"},
        {"test_name": "completeness_batch", "parquet": "completeness.parquet"},
    ]

    generate_pipeline(
        processor_name="completeness",
        parquets_path=Path(test_path) / "data",
        test_list=test_list,
        output_category="metrics",
        test_path=test_path,
    )


@pytest.fixture(scope="session")
def pipeline_domain_gap(
    test_path: str,
) -> None:
    gen_path = Path(test_path) / "data_generated"
    metrics = ["fid", "klmvn_diag", "mmd_linear", "wasserstein_1d"]

    for metric in metrics:
        generate_pipeline(
            processor_name="domain_gap",
            parquets_path=gen_path,
            test_list=[{"test_name": "", "parquet": "target_1000.parquet"}],
            output_category="metrics",
            test_path=test_path,
            metric_name=metric,
            parquet_source_path=Path(gen_path) / "source_1000.parquet",
        )

    generate_pipeline(
        processor_name="domain_gap",
        parquets_path=Path(test_path) / "data",
        test_list=[{"test_name": "wasserstein_bytes", "parquet": "target_bytes.parquet"}],
        output_category="metrics",
        test_path=test_path,
        metric_name="wasserstein_1d",
        parquet_source_path=Path(test_path) / "data/source_bytes.parquet",
    )


@pytest.fixture(scope="session")
def pipeline_visual_features(
    test_path: str,
) -> None:
    test_list = [
        {"test_name": "visual_features", "parquet": "visual_features.parquet"},
        {"test_name": "visual_features_batch", "parquet": "visual_features.parquet"},
        {"test_name": "visual_features_path", "parquet": "visual_features_path.parquet"},
    ]

    generate_pipeline(
        processor_name="visual_features",
        parquets_path=Path(test_path) / "data",
        test_list=test_list,
        output_category="features",
        test_path=test_path,
    )
