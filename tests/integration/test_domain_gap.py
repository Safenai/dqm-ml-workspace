"""Integration tests for the domain gap metric processor.

This module contains tests that verify the domain gap metric processor
correctly computes statistical distances between source and target datasets.
"""

from pathlib import Path
import shlex
from timeit import default_timer as timer
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from dqm_ml_job.cli import execute


@pytest.mark.slow
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

    metric_key = test_name
    if test_name == "wasserstein_bytes":
        metric_key = "wasserstein_1d"

    computed_score = df[metric_key].tolist()[0]
    expected_score = expected_scores["value"]

    print(f"computed_score = {computed_score}")
    assert computed_score == pytest.approx(expected_score, abs=epsilon), (
        f"For {metric_key}, the distance between computed value : {computed_score}",
        f" and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}",
    )


@pytest.mark.slow
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
        "config": {
            "dataloaders": {
                "source_dataset": {
                    "type": "parquet",
                    "path": str(output_path / "source_1000.parquet"),
                    "batch_size": 50,
                    "memory_limit": "2GB",
                    "threads": 4,
                },
                "target_dataset": {
                    "type": "parquet",
                    "path": str(output_path / "target_1000.parquet"),
                    "batch_size": 50,
                    "memory_limit": "2GB",
                    "threads": 4,
                },
            },
            "metrics_processor": {
                "image_embedding": {
                    "type": "image_embedding",
                    "data": {
                        "image_column": "image_path",
                        "mode": "path",
                    },
                    "model": {
                        "arch": "resnet18",
                        "n_layer_feature": -2,
                        "device": "cpu",
                    },
                    "infer": {
                        "batch_size": 8,
                        "height": 299,
                        "width": 299,
                        "norm_mean": [0.485, 0.456, 0.406],
                        "norm_std": [0.229, 0.224, 0.225],
                    },
                },
                "domain_gap_fid": {
                    "type": "domain_gap",
                    "input": {"embedding_col": "embedding"},
                    "delta": {"metric": "fid"},
                },
                "domain_gap_wasserstein": {
                    "type": "domain_gap",
                    "input": {"embedding_col": "embedding"},
                    "delta": {"metric": "wasserstein_1d"},
                },
            },
            "compute_delta": True,
            "outputs": {
                "delta_metrics": {
                    "type": "parquet",
                    "path_pattern": str(output_path / output_file),
                    "columns": [],
                },
            },
        }
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
