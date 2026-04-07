"""Integration tests for the representativeness metric processor.

This module contains tests that verify the representativeness metric processor
correctly evaluates distribution representativeness using various statistical tests.
"""

from pathlib import Path
import shlex
from timeit import default_timer as timer
from typing import Any

import pyarrow.parquet as pq
import pytest

from dqm_ml_job.cli import execute


@pytest.mark.parametrize(
    "test_name",
    [
        "representativeness_not_normal_distribution",
        "representativeness_not_uniform_distribution",
        "representativeness_batch",
        "representativeness_normal_distribution",
        "representativeness_uniform_distribution",
    ],
)
def test_representativeness(
    tests_config: Any,
    output_path: Path,
    test_name: str,
    job_representativeness: Any,
) -> None:
    """Test representativeness metric for various distributions.

    Args:
        tests_config: Test configuration with expected scores and tolerances.
        output_path: Path to write test outputs.
        test_name: Name of the test configuration to run.
        job_representativeness: Fixture that generates the test job configuration.
    """
    command = f"-p tests/fixtures/config/generated/{test_name}.yaml"

    start = timer()
    execute(shlex.split(command))
    end = timer()
    print(f"Execution time: {end - start}")

    test_key = "representativeness_normal_distribution" if test_name == "representativeness_batch" else test_name

    # load test configuration

    expected_scores = tests_config[test_key]["expected_scores"]
    epsilons = tests_config[test_key]["params"]["tolerances"]
    col_names = tests_config[test_key]["params"]["columns_names"]
    metrics = tests_config[test_key]["params"]["metrics"]
    value_names = tests_config[test_key]["params"]["value_names"]
    thresholds = tests_config[test_key]["params"]["thresholds"]
    interpretations = tests_config[test_key]["params"]["interpretations"]

    # # Compare representativeness metrics with expected values
    output_filename = f"metrics_{test_name}_-.parquet"

    error_messages = []
    for col in col_names:
        for metric, epsilon, value, threshold, interpretation in zip(
            metrics, epsilons, value_names, thresholds, interpretations, strict=True
        ):
            expected_score = expected_scores[metric]

            print("**************************")
            print(f"{metric} - {col}")

            column_value = metric + "_" + col + "_" + value
            column_interpretation = metric + "_" + col + "_interpretation"

            table = pq.read_table(Path(output_path) / output_filename)

            # Filter for source_dataset row in selection_source column
            df = table.to_pandas()
            source_row = df[df["selection"] == "source_dataset"]

            computed_score = source_row[column_value].tolist()[0]
            expected_score = expected_score[col]

            print(f"computed_score = {computed_score}")

            expected_interpretation = interpretation[0] if computed_score >= threshold else interpretation[1]
            computed_interpretation = source_row[column_interpretation].tolist()[0]

            if metric not in ["kolmogorov-smirnov", "chi-square"] and computed_score != pytest.approx(
                expected_score, abs=epsilon
            ):
                error_msg = (
                    f"For {column_value}, the distance between computed value : {computed_score}",
                    f" and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}",
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
