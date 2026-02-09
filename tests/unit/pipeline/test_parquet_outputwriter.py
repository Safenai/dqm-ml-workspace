import shutil
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from dqm_ml_pipeline.outputwriter.parquet import ParquetOutputWriter

@pytest.fixture
def temp_output_dir(tmp_path):
    """Provides a temporary output directory and cleans it up."""
    path = tmp_path / "output"
    yield path
    if path.exists():
        shutil.rmtree(path)

def test_parquet_output_writer_creates_directory(temp_output_dir):
    """
    Test that ParquetOutputWriter creates the output directory if it doesn't exist.
    """
    output_path_pattern = str(temp_output_dir / "metrics.parquet")
    
    config = {
        "path_pattern": output_path_pattern,
        "columns": ["metric_name", "value"]
    }
    writer = ParquetOutputWriter(name="test_writer", config=config)

    metrics_data = {
        "selection_1": {"metric_name": "completeness", "value": 0.9},
        "selection_2": {"metric_name": "completeness", "value": 0.85},
    }

    # Ensure the directory does not exist initially
    assert not temp_output_dir.exists()

    writer.write_metrics_dict(metrics_data)

    # Assert that the directory was created and the file exists
    assert temp_output_dir.exists()
    assert (temp_output_dir / "metrics.parquet").is_file()

    # Verify content (optional, but good practice)
    table = pq.read_table(output_path_pattern)
    assert table.num_rows == 2
    assert table.column_names == ["selection", "metric_name", "value"]
    assert table.column("selection").to_pylist() == ["selection_1", "selection_2"]
    assert table.column("value").to_pylist() == [0.9, 0.85]
