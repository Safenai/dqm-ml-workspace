from pathlib import Path
import shlex
from typing import Any

import pyarrow.parquet as pq
import pytest

from dqm_ml_pipeline.cli import execute


@pytest.mark.parametrize("test_name", ["visual_features", "visual_features_batch"])
def test_visual_features(tests_config: Any, test_path: str, output_path: str, test_name: str) -> None:
    command = f"-p packages/dqm-ml-pipeline/tests/config/{test_name}.yaml"
    execute(shlex.split(command))

    expected_scores = tests_config["visual_features"]["expected_scores"]
    col_names = tests_config["visual_features"]["params"]["columns_names"]
    epsilon = tests_config["visual_features"]["params"]["tolerance"]

    output_filename = f"metrics_{test_name}.parquet"

    for col in col_names:
        table = pq.read_table(Path(output_path) / output_filename)

        computed_score = table.to_pandas()[col].tolist()
        expected_score = expected_scores[col]
        assert computed_score == pytest.approx(expected_score, abs=epsilon), (
            f"For {col}, the distance between computed value : {computed_score}",
            f" and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}",
        )


def test_visual_features_path(tests_config: Any, test_path: str, output_path: str) -> None:
    test_name = "visual_features_path"
    command = f"-p packages/dqm-ml-pipeline/tests/config/{test_name}.yaml"
    execute(shlex.split(command))

    expected_scores = tests_config["visual_features"]["expected_scores"]
    col_names = tests_config["visual_features"]["params"]["columns_names"]
    epsilon = tests_config["visual_features"]["params"]["tolerance"]

    output_filename = f"metrics_{test_name}.parquet"

    for col in col_names:
        table = pq.read_table(Path(output_path) / output_filename)

        computed_score = table.to_pandas()[col].tolist()
        expected_score = expected_scores[col]
        assert computed_score == pytest.approx(expected_score, abs=epsilon), (
            f"For {col}, the distance between computed value : {computed_score}",
            f" and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}",
        )
