"""Integration tests for pandas welding use case.

This module contains tests that verify the representativeness metric processor
works correctly with pandas-welding style data.
"""

from pathlib import Path
import shlex
from typing import Any

from dqm_ml_job.cli import execute
import pyarrow.parquet as pq
import pytest


def _assert_metric_value(
    df: Any,
    col: str,
    metric: str,
    value_name: str,
    threshold: float,
    interpretation: tuple[str, str],
    expected_scores: dict[str, dict[str, float]],
    epsilon: float,
    error_messages: list,
) -> None:
    column_value = col + "_" + metric + "_" + value_name
    column_interpretation = col + "_" + metric + "_interpretation"

    source_row = df[df["selection"] == "source_dataset"]
    computed_score = source_row[column_value].tolist()[0]
    expected = expected_scores[metric][col]

    tmp_epsilon = 0.5 if metric == "kolmogorov-smirnov" and col == "sharpness" else epsilon

    if computed_score != pytest.approx(expected, abs=tmp_epsilon):
        error_messages.append(
            f"For {column_value}, the distance between computed value : {computed_score}"
            f" and expected one ---> {expected} is greater than the accepted tolerance {tmp_epsilon}"
        )

    expected_interpretation = interpretation[0] if computed_score >= threshold else interpretation[1]
    computed_interpretation = source_row[column_interpretation].tolist()[0]

    if computed_interpretation != expected_interpretation:
        error_messages.append(
            f"For {column_interpretation}, the interpretation differs"
            f" between computed: {computed_interpretation}"
            f" and expected one ---> {expected_interpretation}"
        )


@pytest.mark.parametrize("test_name", ["pandas_welding"])
def test_representativeness_pandas(tests_config: Any, test_path: Path, output_path: Path, test_name: str) -> None:
    """Test representativeness metric with pandas welding data.

    Args:
        tests_config: Test configuration with expected scores and tolerances.
        test_path: Path to the tests directory.
        output_path: Path to write test outputs.
        test_name: Name of the test configuration to run.
    """
    command = f"-p tests/integration/fixtures/config/{test_name}.yaml"
    execute(shlex.split(command))

    expected_scores = tests_config["pandas_welding"]["expected_scores"]
    epsilon = tests_config["pandas_welding"]["params"]["tolerance"]
    col_names = tests_config["pandas_welding"]["params"]["columns_names"]
    metrics = tests_config["pandas_welding"]["params"]["metrics"]
    value_names = tests_config["pandas_welding"]["params"]["value_names"]
    thresholds = tests_config["pandas_welding"]["params"]["thresholds"]
    interpretations = tests_config["pandas_welding"]["params"]["interpretations"]

    output_filename = f"metrics_{test_name}_-.parquet"
    table = pq.read_table(Path(output_path) / output_filename)
    df = table.to_pandas()

    error_messages = []
    for col in col_names:
        for metric, value, threshold, interpretation in zip(
            metrics, value_names, thresholds, interpretations, strict=True
        ):
            _assert_metric_value(
                df, col, metric, value, threshold, interpretation, expected_scores, epsilon, error_messages
            )

    assert error_messages == []
