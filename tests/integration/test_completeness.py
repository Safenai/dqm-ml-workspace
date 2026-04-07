"""Integration tests for the completeness metric processor.

This module contains tests that verify the completeness metric processor
correctly computes completeness scores for tabular data.
"""

from pathlib import Path
import shlex
from typing import Any

import pyarrow.parquet as pq
import pytest

from dqm_ml_job.cli import execute


@pytest.mark.parametrize("test_name", ["completeness", "completeness_batch"])
def test_completeness(
    tests_config: Any,
    test_path: Path,
    output_path: Path,
    test_name: str,
    job_completeness: Any,
) -> None:
    """Test completeness metric computation for single and batch processing.

    Args:
        tests_config: Test configuration with expected scores and tolerances.
        test_path: Path to the tests directory.
        output_path: Path to write test outputs.
        test_name: Name of the test configuration to run.
        job_completeness: Fixture that generates the test job configuration.
    """
    command = f"-p tests/integration/fixtures/config/generated/{test_name}.yaml"
    execute(shlex.split(command))

    expected_scores = tests_config["completeness"]["expected_scores"]
    epsilon = tests_config["completeness"]["params"]["tolerance"]
    col_names = tests_config["completeness"]["params"]["columns_names"]

    # # Test completeness by columns and overall
    output_filename = f"metrics_{test_name}_-.parquet"

    table = pq.read_table(Path(output_path) / output_filename)
    for col in col_names:
        computed_score = table.to_pandas()[col].tolist()[0]
        expected_score = expected_scores[col]
        assert computed_score == pytest.approx(expected_score, abs=epsilon), (
            f"For {col}, the distance between computed value : {computed_score}",
            f" and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}",
        )
