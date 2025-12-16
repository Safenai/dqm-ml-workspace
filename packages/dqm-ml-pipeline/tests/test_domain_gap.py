from pathlib import Path
import shlex
from timeit import default_timer as timer
from typing import Any

import pyarrow.parquet as pq
import pytest

from dqm_ml_pipeline.cli import execute


@pytest.mark.timeout(600)
@pytest.mark.parametrize("test_name", ["wasserstein", "fid", "klmvn", "mmd"])
def test_domain_gap(tests_config: Any, test_path: Path, output_path: Path, test_name: str, coco_data: Any) -> None:
    # Why dqm is using inception_v3 for FID and not dqm-ml ?

    # pad and cmd not implemented

    command = f"-p packages/dqm-ml-pipeline/tests/config/domain_gap_{test_name}.yaml"
    start = timer()
    execute(shlex.split(command))
    end = timer()
    print(f"Execution time: {end - start}")

    epsilon = tests_config["domain_gap"][test_name]["params"]["tolerance"]
    col_names = tests_config["domain_gap"][test_name]["params"]["columns_names"]
    expected_scores = tests_config["domain_gap"][test_name]["expected_scores"]

    output_filename = f"metrics_domain_gap_{test_name}_source_dataset_target_dataset-0.parquet"

    table = pq.read_table(Path(output_path) / output_filename)
    for col in col_names:
        computed_score = table.to_pandas()[col].tolist()[0]
        expected_score = expected_scores[col]
        if col == "value":
            print(f"computed_score = {computed_score}")
            assert computed_score == pytest.approx(expected_score, abs=epsilon), (
                f"For {col}, the distance between computed value : {computed_score}",
                f" and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}",
            )
        else:
            # test metric name
            assert computed_score == expected_score


def test_domain_gap_bytes(tests_config: Any, test_path: str, output_path: str) -> None:
    test_name = "wasserstein_bytes"

    command = f"-p packages/dqm-ml-pipeline/tests/config/domain_gap_{test_name}.yaml"
    execute(shlex.split(command))

    epsilon = tests_config["domain_gap"][test_name]["params"]["tolerance"]
    col_names = tests_config["domain_gap"][test_name]["params"]["columns_names"]
    expected_scores = tests_config["domain_gap"][test_name]["expected_scores"]

    output_filename = f"metrics_domain_gap_{test_name}_source_dataset_target_dataset-0.parquet"

    table = pq.read_table(Path(output_path) / output_filename)
    for col in col_names:
        computed_score = table.to_pandas()[col].tolist()[0]
        expected_score = expected_scores[col]
        if col == "value":
            assert computed_score == pytest.approx(expected_score, abs=epsilon), (
                f"For {col}, the distance between computed value : {computed_score}",
                f" and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}",
            )
        else:
            # test metric value
            assert computed_score == expected_score
