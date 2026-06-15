"""Integration tests for the diversity metric processor.

This module contains tests that verify the diversity metric processor
correctly computes diversity indices for categorical data.
"""

from pathlib import Path
import shlex
from timeit import default_timer as timer
from typing import Any

import pyarrow.parquet as pq
import pytest

from dqm_ml_job.cli import execute


@pytest.mark.parametrize("test_name", ["diversity", "diversity_batch"])
def test_diversity(
    tests_config: Any,
    test_path: Path,
    output_path: Path,
    test_name: str,
    job_diversity: Any,
) -> None:
    """Test diversity metric computation for single and batch processing.

    Args:
        tests_config: Test configuration with expected scores and tolerances.
        test_path: Path to the tests directory.
        output_path: Path to write test outputs.
        test_name: Name of the test configuration to run.
        job_diversity: Fixture that generates the test job configuration.
    """
    command = f"-p tests/integration/fixtures/config/generated/{test_name}.yaml"

    start = timer()
    execute(shlex.split(command))
    end = timer()
    print(f"Execution time: {end - start}")

    expected = tests_config["diversity"]["expected_scores"]
    epsilon = tests_config["diversity"]["params"]["tolerance"]
    col_names = tests_config["diversity"]["params"]["columns_names"]
    metrics = tests_config["diversity"]["params"]["metrics"]

    output_filename = f"metrics_{test_name}_-.parquet"
    table = pq.read_table(Path(output_path) / output_filename)
    pdf = table.to_pandas()

    for col in col_names:
        for metric in metrics:
            key = f"{col}_{metric}"
            computed = pdf[key].tolist()[0]
            expected_val = expected[metric][col]
            assert computed == pytest.approx(expected_val, abs=epsilon), (
                f"For {col}_{metric}, computed={computed}, expected={expected_val}"
            )
