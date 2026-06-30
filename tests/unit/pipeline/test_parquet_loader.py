"""Unit tests for the Parquet data loader.

This module contains unit tests that verify the ParquetDataLoader
and ParquetDataSelection classes work correctly.
"""

import re
from unittest.mock import MagicMock, patch

import pyarrow as pa

from dqm_ml_job.dataloaders.parquet import ParquetDataLoader, ParquetDataSelection


def test_parquet_dataloader_get_selections_no_split() -> None:
    """Test that get_selections returns a single selection when split_by is not configured."""
    config = {"path": "dummy.parquet"}
    loader = ParquetDataLoader("test", config)
    selections = loader.get_selections()

    assert len(selections) == 1
    assert isinstance(selections[0], ParquetDataSelection)
    assert selections[0].name == "test"


@patch("pyarrow.parquet.read_table")
def test_parquet_dataloader_get_selections_auto_split(mock_read_table) -> None:
    """Test that get_selections creates multiple selections when split_by is configured.

    Args:
        mock_read_table: Mocked pyarrow.parquet.read_table.
    """
    # Mock reading table to get unique values
    mock_table = MagicMock()
    mock_table.column.return_value = pa.array(["val1", "val2", "val1"])
    mock_read_table.return_value = mock_table

    config = {"path": "dummy.parquet", "split": {"by": "category"}}
    loader = ParquetDataLoader("test", config)
    selections = loader.get_selections()

    assert len(selections) == 2
    names = [s.name for s in selections]
    assert "test_val1" in names
    assert "test_val2" in names


@patch("pyarrow.parquet.read_table")
def test_split_auto_with_exclude_literal(mock_read_table) -> None:
    """Auto-discovered split values filtered by literal exclude.

    Args:
        mock_read_table: Mocked pyarrow.parquet.read_table.
    """
    mock_table = MagicMock()
    mock_table.column.return_value = pa.array(["a", "b", "c"])
    mock_read_table.return_value = mock_table

    config = {"path": "dummy.parquet", "split": {"by": "category", "exclude": ["b"]}}
    loader = ParquetDataLoader("test", config)
    selections = loader.get_selections()

    assert len(selections) == 2
    names = [s.name for s in selections]
    assert "test_a" in names
    assert "test_c" in names


@patch("pyarrow.parquet.read_table")
def test_split_auto_with_exclude_wildcard(mock_read_table) -> None:
    """Auto-discovered split values filtered by wildcard exclude.

    Args:
        mock_read_table: Mocked pyarrow.parquet.read_table.
    """
    mock_table = MagicMock()
    mock_table.column.return_value = pa.array(["meta_x", "meta_y", "normal"])
    mock_read_table.return_value = mock_table

    config = {"path": "dummy.parquet", "split": {"by": "category", "exclude": ["meta_*"]}}
    loader = ParquetDataLoader("test", config)
    selections = loader.get_selections()

    assert len(selections) == 1
    assert selections[0].name == "test_normal"


def test_split_explicit_values() -> None:
    """Explicit split values produce one selection per value."""
    config = {"path": "dummy.parquet", "split": {"by": "category", "values": ["a", "b"]}}
    loader = ParquetDataLoader("test", config)
    selections = loader.get_selections()

    assert len(selections) == 2
    names = [s.name for s in selections]
    assert "test_a" in names
    assert "test_b" in names


def test_split_explicit_values_with_exclude_wildcard() -> None:
    """Explicit split values filtered by wildcard exclude."""
    config = {
        "path": "dummy.parquet",
        "split": {"by": "category", "values": ["a", "b", "meta_x"], "exclude": ["meta_*"]},
    }
    loader = ParquetDataLoader("test", config)
    selections = loader.get_selections()

    assert len(selections) == 2
    names = [s.name for s in selections]
    assert "test_a" in names
    assert "test_b" in names


def test_split_filter_merging() -> None:
    """Split filter merges with existing dataloader filter."""
    config = {
        "path": "dummy.parquet",
        "filters": [{"column": "other", "values": [True]}],
        "split": {"by": "category", "values": ["a"]},
    }
    loader = ParquetDataLoader("test", config)
    selections = loader.get_selections()

    assert len(selections) == 1
    assert selections[0].filters_dict == {"other": [True], "category": "a"}


@patch("pyarrow.parquet.read_table")
def test_split_explicit_values_with_wildcard(mock_read_table) -> None:
    """Wildcard patterns in split.values are expanded against available values.

    Args:
        mock_read_table: Mocked pyarrow.parquet.read_table.
    """
    mock_table = MagicMock()
    mock_table.column.return_value = pa.array(["dog_brown", "dog_black", "cat"])
    mock_read_table.return_value = mock_table

    config = {"path": "dummy.parquet", "split": {"by": "category", "values": ["cat", "dog_*"]}}
    loader = ParquetDataLoader("test", config)
    selections = loader.get_selections()

    assert len(selections) == 3
    names = [s.name for s in selections]
    assert "test_cat" in names
    assert "test_dog_brown" in names
    assert "test_dog_black" in names


def test_parquet_data_selection_bootstrap(mock_parquet_dataset) -> None:
    """Test that ParquetDataSelection.bootstrap initializes the dataset correctly.

    Args:
        mock_parquet_dataset: Fixture providing mocked ParquetDataset.
    """
    mock_ds_instance = mock_parquet_dataset.return_value
    mock_fragment = MagicMock()
    mock_fragment.count_rows.return_value = 100
    mock_ds_instance.fragments = [mock_fragment]

    selection = ParquetDataSelection("test", "dummy.parquet", filters_dict={"col1": ["val1"]})
    selection.bootstrap(["col2"])

    assert selection.samples_count == 100
    assert selection.filter_expr is not None
    # Verify filter_expr was used in ParquetDataset
    mock_parquet_dataset.assert_called_with("dummy.parquet", filters=selection.filter_expr, filesystem=None)


def test_parquet_data_selection_iter(mock_parquet_dataset) -> None:
    """Test that ParquetDataSelection iteration yields correct batches.

    Args:
        mock_parquet_dataset: Fixture providing mocked ParquetDataset.
    """
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


def test_parquet_data_selection_get_nb_batches(mock_parquet_dataset) -> None:
    """Test that get_nb_batches returns correct number of batches.

    Args:
        mock_parquet_dataset: Fixture providing mocked ParquetDataset.
    """
    mock_ds_instance = mock_parquet_dataset.return_value
    mock_fragment = MagicMock()
    mock_fragment.count_rows.return_value = 250
    mock_ds_instance.fragments = [mock_fragment]

    selection = ParquetDataSelection("test", "dummy.parquet", batch_size=100)
    selection.bootstrap([])

    assert selection.get_nb_batches() == 3


def test_parquet_data_selection_bootstrap_with_multiple_values(mock_parquet_dataset) -> None:
    """Test bootstrap with multiple filter values.

    Args:
        mock_parquet_dataset: Fixture providing mocked ParquetDataset.
    """
    mock_ds_instance = mock_parquet_dataset.return_value
    mock_fragment = MagicMock()
    mock_fragment.count_rows.return_value = 100
    mock_ds_instance.fragments = [mock_fragment]

    selection = ParquetDataSelection("test", "dummy.parquet", filters_dict={"col1": ["val1", "val2"]})
    selection.bootstrap(["col2"])

    assert selection.samples_count == 100
    assert selection.filter_expr is not None


def test_parquet_data_selection_bootstrap_with_wildcard(mock_parquet_dataset) -> None:
    """Test bootstrap with wildcard filter pattern.

    Args:
        mock_parquet_dataset: Fixture providing mocked ParquetDataset.
    """
    mock_ds_instance = mock_parquet_dataset.return_value
    mock_fragment = MagicMock()
    mock_fragment.count_rows.return_value = 50
    mock_ds_instance.fragments = [mock_fragment]

    selection = ParquetDataSelection("test", "dummy.parquet", filters_dict={"col1": ["meta_*"]})
    selection.bootstrap(["col2"])

    assert selection.samples_count == 50
    assert selection.filter_expr is not None


class TestParquetUtils:
    """Tests for Parquet utility functions."""

    def test_fnmatch_to_regex(self) -> None:
        """Test _fnmatch_to_regex converts wildcard pattern to valid regex."""
        from dqm_ml_job.dataloaders.parquet import _fnmatch_to_regex

        pattern = _fnmatch_to_regex("meta_*")
        re.compile(pattern)  # should be a valid regex
        assert "meta_" in pattern

    def test_resolve_pyarrow_type(self) -> None:
        """Test _resolve_pyarrow_type returns correct PyArrow types for basic types."""
        from dqm_ml_job.dataloaders.parquet import _resolve_pyarrow_type

        assert _resolve_pyarrow_type("int32") == pa.int32()
        assert _resolve_pyarrow_type("float64") == pa.float64()
        assert _resolve_pyarrow_type("bool") == pa.bool_()
        assert _resolve_pyarrow_type("str") == pa.utf8()

    def test_resolve_pyarrow_type_categorical(self) -> None:
        """Test _resolve_pyarrow_type returns dictionary type for categorical."""
        from dqm_ml_job.dataloaders.parquet import _resolve_pyarrow_type

        result = _resolve_pyarrow_type("categorical")
        assert result == pa.dictionary(pa.int32(), pa.utf8())


class TestParquetApplyTransforms:
    """Tests for ParquetDataSelection._apply_transforms method."""

    def test_in_place_transform(self) -> None:
        """Test in_place=True casts column in place."""
        transforms = [{"column": "col1", "to_type": "float32", "in_place": True}]
        selection = ParquetDataSelection("test", "dummy.parquet", transforms=transforms)
        batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3], type=pa.int64())], names=["col1"])
        result = selection._apply_transforms(batch)
        assert result.schema.field("col1").type == pa.float32()

    def test_append_transform(self) -> None:
        """Test default (in_place=False) appends new column with suffix."""
        transforms = [{"column": "col1", "to_type": "float32"}]
        selection = ParquetDataSelection("test", "dummy.parquet", transforms=transforms)
        batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3], type=pa.int64())], names=["col1"])
        result = selection._apply_transforms(batch)
        assert "col1_float32" in result.schema.names
        assert result.schema.field("col1_float32").type == pa.float32()

    def test_transform_column_not_found(self) -> None:
        """Test transform referencing non-existent column returns batch unchanged."""
        transforms = [{"column": "nonexistent", "to_type": "float32"}]
        selection = ParquetDataSelection("test", "dummy.parquet", transforms=transforms)
        batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=["col1"])
        result = selection._apply_transforms(batch)
        assert result == batch  # unchanged
