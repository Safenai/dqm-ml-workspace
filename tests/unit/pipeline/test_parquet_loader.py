"""Unit tests for the Parquet data loader.

This module contains unit tests that verify the ParquetDataLoader
and ParquetDataSelection classes work correctly.
"""

from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from dqm_ml_job.dataloaders.parquet import ParquetDataLoader, ParquetDataSelection


@pytest.fixture
def mock_parquet_dataset():
    """Provide a mocked ParquetDataset for testing.

    Yields:
        Mocked ParquetDataset class.
    """
    with patch("pyarrow.parquet.ParquetDataset") as mock:
        yield mock


def test_parquet_dataloader_init_no_path():
    """Test that ParquetDataLoader raises error when path is not provided."""
    with pytest.raises(ValueError, match="must contain 'path'"):
        ParquetDataLoader("test", {})


def test_parquet_dataloader_get_selections_no_split():
    """Test that get_selections returns a single selection when split_by is not configured."""
    config = {"path": "dummy.parquet"}
    loader = ParquetDataLoader("test", config)
    selections = loader.get_selections()

    assert len(selections) == 1
    assert isinstance(selections[0], ParquetDataSelection)
    assert selections[0].name == "test"


@patch("pyarrow.parquet.read_table")
def test_parquet_dataloader_get_selections_auto_split(mock_read_table):
    """Test that get_selections creates multiple selections when split_by is configured."""
    # Mock reading table to get unique values
    mock_table = MagicMock()
    mock_table.column.return_value = pa.array(["val1", "val2", "val1"])
    mock_read_table.return_value = mock_table

    config = {"path": "dummy.parquet", "split_by": "category"}
    loader = ParquetDataLoader("test", config)
    selections = loader.get_selections()

    assert len(selections) == 2
    names = [s.name for s in selections]
    assert "test_val1" in names
    assert "test_val2" in names


def test_parquet_data_selection_bootstrap(mock_parquet_dataset):
    """Test that ParquetDataSelection.bootstrap initializes the dataset correctly."""
    mock_ds_instance = mock_parquet_dataset.return_value
    mock_fragment = MagicMock()
    mock_fragment.count_rows.return_value = 100
    mock_ds_instance.fragments = [mock_fragment]

    selection = ParquetDataSelection("test", "dummy.parquet", filters_dict={"col1": "val1"})
    selection.bootstrap(["col2"])

    assert selection.samples_count == 100
    assert selection.filter_expr is not None
    # Verify filter_expr was used in ParquetDataset
    mock_parquet_dataset.assert_called_with("dummy.parquet", filters=selection.filter_expr)


def test_parquet_data_selection_iter(mock_parquet_dataset):
    """Test that ParquetDataSelection iteration yields correct batches."""
    mock_ds_instance = mock_parquet_dataset.return_value
    mock_ds_instance.files = ["file1.parquet"]

    selection = ParquetDataSelection("test", "dummy.parquet")
    selection.bootstrap(["col1"])

    mock_batch = pa.RecordBatch.from_arrays([pa.array([1, 2])], names=["col1"])

    with patch("pyarrow.parquet.ParquetFile") as mock_file:
        mock_file_instance = mock_file.return_value
        mock_file_instance.iter_batches.return_value = [mock_batch]

        batches = list(selection)
        assert len(batches) == 1
        assert batches[0].num_rows == 2


def test_parquet_data_selection_get_nb_batches(mock_parquet_dataset):
    """Test that get_nb_batches returns correct number of batches."""
    mock_ds_instance = mock_parquet_dataset.return_value
    mock_fragment = MagicMock()
    mock_fragment.count_rows.return_value = 250
    mock_ds_instance.fragments = [mock_fragment]

    selection = ParquetDataSelection("test", "dummy.parquet", batch_size=100)
    selection.bootstrap([])

    assert selection.get_nb_batches() == 3
