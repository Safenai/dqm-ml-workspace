"""Integration tests for the domain gap metric processor.

This module contains tests that verify the domain gap metric processor
correctly computes statistical distances between source and target datasets.
"""

import math
from pathlib import Path
import shlex
from timeit import default_timer as timer
from typing import Any

from dqm_ml_job.cli import execute
import pyarrow.parquet as pq
import pytest
from tests.utils.jobs import _parse_domain_gap_test_name
import yaml

_DOMAIN_GAP_FEATURES = {
    "name": "image_embedding",
    "type": "features_embeddings",
    "columns": {"input": ["image_path"]},
    "model": {"arch": "resnet18", "n_layer_feature": -2, "device": "cpu"},
    "infer": {
        "batch_size": 10,
        "width": 64,
        "height": 64,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
    },
}

_DOMAIN_GAP_PROC = {
    "name": "domain_gap",
    "type": "domain_gap",
    "columns": {"input": ["image_path_embedding"]},
    "distance": {"metric": "mmd_linear"},
}


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "test_name",
    [
        "wasserstein_1d",
        "fid",
        "klmvn_diag",
        "mmd_linear",
        "wasserstein_bytes",
        "mmd_rbf",
        "mmd_poly",
        "pad",
        "cmd",
        # Parameter coverage variants
        "fid_no_sum_outer",
        "mmd_rbf_no_store",
        "pad_mae",
        "mmd_rbf_gamma2",
        "wasserstein_custom_hist",
    ],
)
def test_domain_gap(
    tests_config: Any,
    test_path: Path,
    output_path: Path,
    test_name: str,
    coco_data: Any,
    job_domain_gap: Any,
) -> None:
    """Test domain gap metric computation between source and target datasets.

    Args:
        tests_config: Test configuration with expected scores and tolerances.
        test_path: Path to the tests directory.
        output_path: Path to write test outputs.
        test_name: Name of the domain gap metric to test.
        coco_data: Fixture providing COCO dataset for testing.
        job_domain_gap: Fixture that generates the test job configuration.
    """
    command = f"-p tests/integration/fixtures/config/generated/domain_gap_{test_name}.yaml"
    start = timer()
    execute(shlex.split(command))
    end = timer()
    print(f"Execution time: {end - start}")

    epsilon = tests_config["domain_gap"][test_name]["params"]["tolerance"]
    expected_scores = tests_config["domain_gap"][test_name]["expected_scores"]

    output_filename = f"metrics_domain_gap_{test_name}_delta-.parquet"

    table = pq.read_table(Path(output_path) / output_filename)
    df = table.to_pandas()

    base_metric, _ = _parse_domain_gap_test_name(test_name)
    metric_key = "wasserstein_1d" if test_name == "wasserstein_bytes" else base_metric

    expected_score = expected_scores["value"]

    # NaN-safe assertion: negative tests expect the metric column to be absent or NaN
    if expected_score is None:
        assert metric_key not in df.columns or df[metric_key].isna().all(), (
            f"Expected NaN for '{metric_key}', got a value"
        )
        print(f"{metric_key} = NaN (expected)")
    else:
        assert metric_key in df.columns, (
            f"Column '{metric_key}' not found in output. Available columns: {list(df.columns)}"
        )
        computed_score = df[metric_key].tolist()[0]
        assert not math.isnan(computed_score), f"Expected {expected_score} for '{metric_key}', got NaN"
        print(f"computed_score = {computed_score}")
        assert computed_score == pytest.approx(expected_score, abs=epsilon), (
            f"For {metric_key}, the distance between computed value : {computed_score}",
            f" and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}",
        )


@pytest.mark.timeout(600)
def test_domain_gap_multi_metrics(
    test_path: str,
    output_path: Path,
    tests_config: Any,
    coco_data: Any,
) -> None:
    """Test domain gap with 2 dataloaders and 2 metric processors (fid + wasserstein_1d).

    Exercises the accumulate + flush pipeline (single output path with no {}
    placeholders) and the pa.concat_tables path in _compute_delta_metrics
    where different processors produce different metric columns.

    Args:
        test_path: Path to the tests directory.
        output_path: Path to write test outputs.
        tests_config: Test configuration with expected scores and tolerances.
        coco_data: Fixture providing COCO dataset for testing.
    """
    output_file = "metrics_domain_gap_multi_fid_wasserstein.parquet"

    config = {
        "dataloaders": {
            "loaders": [
                {
                    "name": "source_dataset",
                    "type": "parquet",
                    "path": str(output_path / "source_1000.parquet"),
                    "batch_size": 50,
                    "sample_path": [{"column": "image_path"}],
                },
                {
                    "name": "target_dataset",
                    "type": "parquet",
                    "path": str(output_path / "target_1000.parquet"),
                    "batch_size": 50,
                    "sample_path": [{"column": "image_path"}],
                },
            ],
        },
        "features": {
            "processors": [
                {
                    "name": "image_embedding",
                    "type": "features_embeddings",
                    "columns": {"input": ["image_path"]},
                    "model": {
                        "arch": "resnet18",
                        "n_layer_feature": -2,
                        "device": "cpu",
                    },
                    "infer": {
                        "batch_size": 8,
                        "height": 64,
                        "width": 64,
                        "norm_mean": [0.485, 0.456, 0.406],
                        "norm_std": [0.229, 0.224, 0.225],
                    },
                },
            ],
        },
        "gap": {
            "outputs": {
                "path": str(output_path / output_file),
            },
            "processors": [
                {
                    "name": "domain_gap_fid",
                    "type": "domain_gap",
                    "columns": {"input": ["image_path_embedding"]},
                    "distance": {"metric": "fid"},
                },
                {
                    "name": "domain_gap_wasserstein",
                    "type": "domain_gap",
                    "columns": {"input": ["image_path_embedding"]},
                    "distance": {"metric": "wasserstein_1d"},
                },
            ],
        },
    }

    # Write config to generated configs directory
    config_dir = Path(test_path) / "integration" / "fixtures" / "config" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "domain_gap_multi_fid_wasserstein.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Execute the job
    start = timer()
    execute(["-p", str(config_path)])
    end = timer()
    print(f"Execution time: {end - start}")

    # Read output
    table = pq.read_table(output_path / output_file)
    df = table.to_pandas()

    print(df.to_string())

    # Assert both metric columns exist
    assert "fid" in df.columns, "fid column missing from output"
    assert "wasserstein_1d" in df.columns, "wasserstein_1d column missing from output"

    # Assert both columns have at least one non-null value
    assert df["fid"].notna().any(), "fid column is all null"
    assert df["wasserstein_1d"].notna().any(), "wasserstein_1d column is all null"

    # Validate fid score
    fid_row = df[df["fid"].notna()]
    fid_expected = tests_config["domain_gap"]["fid"]["expected_scores"]["value"]
    fid_eps = tests_config["domain_gap"]["fid"]["params"]["tolerance"]
    computed_fid = fid_row["fid"].iloc[0]
    print(f"computed_fid = {computed_fid}")
    assert computed_fid == pytest.approx(fid_expected, abs=fid_eps), (
        f"FID distance {computed_fid} != expected {fid_expected} (tolerance {fid_eps})"
    )

    # Validate wasserstein_1d score
    wasserstein_row = df[df["wasserstein_1d"].notna()]
    wasserstein_expected = tests_config["domain_gap"]["wasserstein_1d"]["expected_scores"]["value"]
    wasserstein_eps = tests_config["domain_gap"]["wasserstein_1d"]["params"]["tolerance"]
    computed_wasserstein = wasserstein_row["wasserstein_1d"].iloc[0]
    print(f"computed_wasserstein = {computed_wasserstein}")
    assert computed_wasserstein == pytest.approx(wasserstein_expected, abs=wasserstein_eps), (
        f"Wasserstein distance {computed_wasserstein} != expected {wasserstein_expected} (tolerance {wasserstein_eps})"
    )


@pytest.mark.parametrize("loader_type", ["parquet", "csv"])
def test_split_by_2classes_computation(
    test_path: str, coco_data: list[Path], coco_csv: list[Path], loader_type: str
) -> None:
    """Test split_by with 2 classes actually computes domain gap."""
    suffix = "parquet" if loader_type == "parquet" else "csv"
    paths = coco_csv if loader_type == "csv" else coco_data

    config = {
        "dataloaders": {
            "loaders": [
                {
                    "name": "coco_classes",
                    "type": loader_type,
                    "path": str(paths[0]),
                    "batch_size": 50,
                    "sample_path": [{"column": "image_path"}],
                    "split": {"by": "class", "values": ["bird", "elephant"]},
                },
            ],
        },
        "features": {"processors": [_DOMAIN_GAP_FEATURES]},
        "gap": {
            "outputs": {"path": f"tests/outputs/data/metrics_domain_gap_split_2classes_{suffix}_delta-.parquet"},
            "processors": [_DOMAIN_GAP_PROC],
        },
    }

    config_dir = Path(test_path) / "integration" / "fixtures" / "config" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"domain_gap_split_2classes_{suffix}.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    execute(shlex.split(f"-p {config_path}"))

    table = pq.read_table(f"tests/outputs/data/metrics_domain_gap_split_2classes_{suffix}_delta-.parquet")
    df = table.to_pandas()
    assert len(df) == 1, "Should have 1 domain gap result"
    assert df["selection_source"].iloc[0] == "coco_classes_bird"
    assert df["selection_target"].iloc[0] == "coco_classes_elephant"
    print(f"Domain gap (bird vs elephant, {loader_type}): {df['mmd_linear'].iloc[0]:.2f}")


@pytest.mark.parametrize("loader_type", ["parquet", "csv"])
def test_filter_1class_2loaders_computation(
    test_path: str, coco_data: list[Path], coco_csv: list[Path], loader_type: str
) -> None:
    """Test filter 1 class on 2 loaders actually computes domain gap."""
    suffix = "parquet" if loader_type == "parquet" else "csv"
    paths = coco_csv if loader_type == "csv" else coco_data

    config = {
        "dataloaders": {
            "loaders": [
                {
                    "name": "source_zebra",
                    "type": loader_type,
                    "path": str(paths[0]),
                    "batch_size": 50,
                    "sample_path": [{"column": "image_path"}],
                    "filters": [{"column": "class", "values": ["zebra"]}],
                },
                {
                    "name": "target_zebra",
                    "type": loader_type,
                    "path": str(paths[1]),
                    "batch_size": 50,
                    "sample_path": [{"column": "image_path"}],
                    "filters": [{"column": "class", "values": ["zebra"]}],
                },
            ],
        },
        "features": {"processors": [_DOMAIN_GAP_FEATURES]},
        "gap": {
            "outputs": {
                "path": f"tests/outputs/data/metrics_domain_gap_filter_1class_2loaders_{suffix}_delta-.parquet"
            },
            "processors": [_DOMAIN_GAP_PROC],
        },
    }

    config_dir = Path(test_path) / "integration" / "fixtures" / "config" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"domain_gap_filter_1class_2loaders_{suffix}.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    execute(shlex.split(f"-p {config_path}"))

    table = pq.read_table(f"tests/outputs/data/metrics_domain_gap_filter_1class_2loaders_{suffix}_delta-.parquet")
    df = table.to_pandas()
    assert len(df) == 1, "Should have 1 domain gap result"
    assert df["selection_source"].iloc[0] == "source_zebra"
    assert df["selection_target"].iloc[0] == "target_zebra"
    print(f"Domain gap (zebra source vs zebra target, {loader_type}): {df['mmd_linear'].iloc[0]:.2f}")


@pytest.mark.parametrize("loader_type", ["parquet", "csv"])
def test_split_by_filter_computation(
    test_path: str, coco_data: list[Path], coco_csv: list[Path], loader_type: str
) -> None:
    """Test split_by class + filter on domain column computes domain gap."""
    suffix = "parquet" if loader_type == "parquet" else "csv"
    paths = coco_csv if loader_type == "csv" else coco_data

    config = {
        "dataloaders": {
            "loaders": [
                {
                    "name": "indoor_bird_elephant",
                    "type": loader_type,
                    "path": str(paths[0]),
                    "batch_size": 50,
                    "sample_path": [{"column": "image_path"}],
                    "split": {"by": "class", "values": ["bird", "elephant"]},
                    "filters": [{"column": "domain", "values": ["indoor"]}],
                },
            ],
        },
        "features": {"processors": [_DOMAIN_GAP_FEATURES]},
        "gap": {
            "outputs": {"path": f"tests/outputs/data/metrics_domain_gap_split_by_filter_{suffix}_delta-.parquet"},
            "processors": [_DOMAIN_GAP_PROC],
        },
    }

    config_dir = Path(test_path) / "integration" / "fixtures" / "config" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"domain_gap_split_by_filter_{suffix}.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    execute(shlex.split(f"-p {config_path}"))

    table = pq.read_table(f"tests/outputs/data/metrics_domain_gap_split_by_filter_{suffix}_delta-.parquet")
    df = table.to_pandas()
    assert len(df) == 1, "Should have 1 domain gap result (bird vs elephant from same loader split by class)"
    assert df["selection_source"].iloc[0] == "indoor_bird_elephant_bird"
    assert df["selection_target"].iloc[0] == "indoor_bird_elephant_elephant"
    assert "mmd_linear" in df.columns, "mmd_linear column missing"
    print(f"Domain gap (indoor bird vs indoor elephant, {loader_type}): {df['mmd_linear'].iloc[0]:.2f}")
