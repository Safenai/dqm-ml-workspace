from pathlib import Path
import shlex
from typing import Any

import pyarrow.parquet as pq
import pytest

from dqm_ml_job.cli import execute


@pytest.mark.parametrize("test_name", ["visual_features", "visual_features_batch", "visual_features_path"])
def test_visual_features(
    tests_config: Any,
    test_path: str,
    output_path: str,
    test_name: str,
    job_visual_features: Any,
) -> None:
    command = f"-p tests/fixtures/config/generated/{test_name}.yaml"
    execute(shlex.split(command))

    expected_scores = tests_config["visual_features"]["expected_scores"]
    col_names = tests_config["visual_features"]["params"]["columns_names"]
    epsilon = tests_config["visual_features"]["params"]["tolerance"]

    output_filename = f"metrics_{test_name}_source_dataset-0.parquet"

    for col in col_names:
        table = pq.read_table(Path(output_path) / output_filename)

        computed_score = table.to_pandas()[col].tolist()
        expected_score = expected_scores[col]
        assert computed_score == pytest.approx(expected_score, abs=epsilon), (
            f"For {col}, the distance between computed value : {computed_score}",
            f" and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}",
        )
