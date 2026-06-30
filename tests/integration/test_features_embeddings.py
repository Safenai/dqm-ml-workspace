"""Integration tests for the embedding features processor.

This module contains tests that verify the features_embeddings processor
correctly extracts image embeddings using a neural network model.
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
        "features_embeddings",
        "features_embeddings_batch",
        "features_embeddings_multi_layer",
        "features_embeddings_n_layer_0",
        "features_embeddings_custom_norm",
        "features_embeddings_prefix",
        "features_embeddings_suffix",
        "features_embeddings_infer_batch_size",
    ],
)
def test_features_embeddings(
    tests_config: Any,
    output_path: str,
    test_name: str,
    job_features_embeddings: Any,
) -> None:
    """Test embedding features extraction with various parameter configurations.

    Args:
        tests_config: Test configuration with expected scores and tolerances.
        output_path: Path to write test outputs.
        test_name: Name of the test configuration to run.
        job_features_embeddings: Fixture that generates the test job configuration.
    """
    command = f"-p tests/integration/fixtures/config/generated/{test_name}.yaml"

    start = timer()
    execute(shlex.split(command))
    end = timer()
    print(f"Execution time: {end - start}")

    test_config = tests_config["features_embeddings"]
    expected_scores = test_config["expected_scores"][test_name]
    col_names = test_config["columns_names"][test_name]
    epsilon = test_config["params"]["tolerance"]

    output_filename = f"metrics_{test_name}_source_dataset-0.parquet"

    table = pq.read_table(Path(output_path) / output_filename)
    df = table.to_pandas()

    for col in col_names:
        assert col in df.columns, f"Column '{col}' not found in output. Available: {list(df.columns)}"
        assert not df[col].isna().all(), f"Column '{col}' is all null"

    first_element = expected_scores.get("first_element")
    if first_element is not None:
        first_col = col_names[0]
        first_val = df[first_col].iloc[0]
        if isinstance(first_val, list) and len(first_val) > 0:
            computed = first_val[0]
            assert computed == pytest.approx(first_element, abs=epsilon), (
                f"First embedding element mismatch: {computed} != {first_element}"
            )
