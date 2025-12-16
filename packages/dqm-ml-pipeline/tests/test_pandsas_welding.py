from pathlib import Path
import shlex
from typing import Any

import pyarrow.parquet as pq
import pytest

from dqm_ml_pipeline.cli import execute


@pytest.mark.parametrize("test_name", ["pandas_welding"])
def test_representativeness_pandas(tests_config: Any, test_path: Path, output_path: Path, test_name: str) -> None:
    command = f"-p packages/dqm-ml-pipeline/tests/config/{test_name}.yaml"
    execute(shlex.split(command))

    # load test configuration

    expected_scores = tests_config["pandas_welding"]["expected_scores"]
    epsilon = tests_config["pandas_welding"]["params"]["tolerance"]
    col_names = tests_config["pandas_welding"]["params"]["columns_names"]
    metrics = tests_config["pandas_welding"]["params"]["metrics"]
    value_names = tests_config["pandas_welding"]["params"]["value_names"]
    thresholds = tests_config["pandas_welding"]["params"]["thresholds"]
    interpretations = tests_config["pandas_welding"]["params"]["interpretations"]

    # # Compare representativeness metrics with expected values
    output_filename = f"metrics_{test_name}_source_dataset-0.parquet"

    error_messages = []
    for col in col_names:
        for metric, value, threshold, interpretation in zip(
            metrics, value_names, thresholds, interpretations, strict=True
        ):
            expected_score = expected_scores[metric]

            column_value = metric + "_" + col + "_" + value
            column_interpretation = metric + "_" + col + "_interpretation"

            table = pq.read_table(Path(output_path) / output_filename)

            computed_score = table.to_pandas()[column_value].tolist()[0]
            expected_score = expected_score[col]

            expected_interpretation = interpretation[0] if computed_score >= threshold else interpretation[1]
            computed_interpretation = table.to_pandas()[column_interpretation].tolist()[0]

            # TODO check why this metric has such variance
            tmp_epsilon = 0.5 if metric == "kolmogorov-smirnov" and col == "sharpness" else epsilon

            if computed_score != pytest.approx(expected_score, abs=tmp_epsilon):
                error_msg = (
                    f"For {column_value}, the distance between computed value : {computed_score}",
                    f" and expected one ---> {expected_score} is greater than the accepted tolerance {tmp_epsilon}",
                )
                error_messages.append(error_msg)
            if computed_interpretation != expected_interpretation:
                error_msg = (
                    f"For {column_interpretation}, the interpretation differs"
                    f" between computed: {computed_interpretation}",
                    f" and expected one ---> {expected_interpretation}",
                )
                error_messages.append(error_msg)

    assert error_messages == []
