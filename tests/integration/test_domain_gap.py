from pathlib import Path
import shlex
from timeit import default_timer as timer
from typing import Any

import pyarrow.parquet as pq
import pytest

from dqm_ml_job.cli import execute


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "test_name",
    ["wasserstein_1d", "fid", "klmvn_diag", "mmd_linear", "wasserstein_bytes"],
)
def test_domain_gap(
    tests_config: Any,
    test_path: Path,
    output_path: Path,
    test_name: str,
    coco_data: Any,
    job_domain_gap: Any,
) -> None:
    # pad and cmd not implemented

    command = f"-p tests/fixtures/config/generated/domain_gap_{test_name}.yaml"
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
