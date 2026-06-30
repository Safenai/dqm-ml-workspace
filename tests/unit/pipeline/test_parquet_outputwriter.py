"""Unit tests for the Parquet output writer.

This module contains unit tests that verify the ParquetOutputWriter
class correctly writes metrics and features to Parquet files.
"""

import logging

import pyarrow as pa
import pyarrow.parquet as pq

from dqm_ml_job.outputwriter.parquet import ParquetOutputWriter


def test_parquet_output_writer_creates_directory(temp_output_path):
    """Test that ParquetOutputWriter creates the output directory if it doesn't exist."""
    output_path_pattern = str(temp_output_path / "metrics.parquet")

    config = {"path_pattern": output_path_pattern, "columns": ["metric_name", "value"]}
    writer = ParquetOutputWriter(name="test_writer", config=config)

    metrics_data = {
        "selection_1": {"metric_name": "completeness", "value": 0.9},
        "selection_2": {"metric_name": "completeness", "value": 0.85},
    }

    # Ensure the directory does not exist initially
    assert not temp_output_path.exists()

    writer.write_metrics_dict(metrics_data)
    writer.flush()

    # Assert that the directory was created and the file exists
    assert temp_output_path.exists()
    assert (temp_output_path / "metrics.parquet").is_file()

    # Verify content (optional, but good practice)
    table = pq.read_table(output_path_pattern)
    assert table.num_rows == 2
    assert table.column_names == ["selection", "metric_name", "value"]
    assert table.column("selection").to_pylist() == ["selection_1", "selection_2"]
    assert table.column("value").to_pylist() == [0.9, 0.85]


def test_write_table_missing_column_logs(caplog):
    """Logs an error when a required column is missing from the written table."""
    writer = ParquetOutputWriter(
        name="test",
        config={"path_pattern": "/tmp/test.parquet", "columns": ["required_col"]},
    )
    with caplog.at_level(logging.ERROR):
        writer.write_table("sel1", {"wrong_col": pa.array([1.0])}, 0)
    assert "Missing" in caplog.text


def test_write_metrics_dict_with_pa_array_value(temp_output_path):
    """Writes metrics whose values are PyArrow arrays correctly."""
    output_path_pattern = str(temp_output_path / "metrics_arr.parquet")
    writer = ParquetOutputWriter(
        name="test",
        config={"path_pattern": output_path_pattern, "columns": ["vals"]},
    )
    metrics = {"sel1": {"vals": pa.array([1.0])}}
    writer.write_metrics_dict(metrics)
    writer.flush()
    table = pq.read_table(output_path_pattern)
    assert table.num_rows == 1


def test_write_table_creates_directory(temp_output_path):
    """Creates nested output directory when it does not exist."""
    output_path_pattern = str(temp_output_path / "sub" / "features.parquet")
    writer = ParquetOutputWriter(
        name="test",
        config={"path_pattern": output_path_pattern, "columns": ["feat"]},
    )
    assert not temp_output_path.exists()
    writer.write_table("sel1", {"feat": pa.array([1.0])}, 0)
    writer.flush()
    assert temp_output_path.exists()
