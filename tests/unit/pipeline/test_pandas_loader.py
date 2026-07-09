"""Unit tests for the Pandas (CSV) data loader.

Tests split, exclude, and wildcard expansion for CSV data loaders
using actual CSV files written to a temporary directory.
"""

from pathlib import Path

from dqm_ml_job.dataloaders.pandas import PandasDataLoader, PandasDataSelection, _apply_pandas_transforms
import pandas as pd
import pytest


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """Write a small CSV with a category column and return its path.

    Args:
        tmp_path: Pytest fixture providing a temporary directory.

    Returns:
        Path to the created CSV file.
    """
    df = pd.DataFrame(
        {
            "category": ["a", "b", "c", "b", "a"],
            "value": [1, 2, 3, 4, 5],
        }
    )
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


def test_get_selections_no_split(csv_path: Path) -> None:
    """Without split, returns a single selection.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    loader = PandasDataLoader("test", {"path": str(csv_path)})
    selections = loader.get_selections()
    assert len(selections) == 1
    assert selections[0].name == "test"
    assert isinstance(selections[0], PandasDataSelection)


def test_split_auto_discover(csv_path: Path) -> None:
    """Auto-discovers unique split values.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    loader = PandasDataLoader("test", {"path": str(csv_path), "split": {"by": "category"}})
    selections = loader.get_selections()
    assert len(selections) == 3
    names = [s.name for s in selections]
    assert "test_a" in names
    assert "test_b" in names
    assert "test_c" in names


def test_split_explicit_values(csv_path: Path) -> None:
    """Explicit split values produce one selection per value.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    loader = PandasDataLoader("test", {"path": str(csv_path), "split": {"by": "category", "values": ["a", "c"]}})
    selections = loader.get_selections()
    assert len(selections) == 2
    names = [s.name for s in selections]
    assert "test_a" in names
    assert "test_c" in names


def test_split_auto_with_exclude_literal(csv_path: Path) -> None:
    """Auto-discovered values filtered by literal exclude.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    loader = PandasDataLoader("test", {"path": str(csv_path), "split": {"by": "category", "exclude": ["b"]}})
    selections = loader.get_selections()
    assert len(selections) == 2
    names = [s.name for s in selections]
    assert "test_a" in names
    assert "test_c" in names


def test_split_auto_with_exclude_wildcard(csv_path: Path) -> None:
    """Auto-discovered values filtered by wildcard exclude.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    loader = PandasDataLoader("test", {"path": str(csv_path), "split": {"by": "category", "exclude": ["[ab]"]}})
    selections = loader.get_selections()
    assert len(selections) == 1
    assert selections[0].name == "test_c"


def test_split_explicit_values_with_exclude_wildcard(csv_path: Path) -> None:
    """Explicit values filtered by wildcard exclude.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    loader = PandasDataLoader(
        "test",
        {
            "path": str(csv_path),
            "split": {"by": "category", "values": ["a", "b", "c"], "exclude": ["[ab]"]},
        },
    )
    selections = loader.get_selections()
    assert len(selections) == 1
    assert selections[0].name == "test_c"


def test_split_explicit_values_with_wildcard(csv_path: Path) -> None:
    """Wildcard patterns in split.values are expanded.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    loader = PandasDataLoader(
        "test",
        {"path": str(csv_path), "split": {"by": "category", "values": ["a", "[bc]"]}},
    )
    selections = loader.get_selections()
    assert len(selections) == 3
    names = [s.name for s in selections]
    assert "test_a" in names
    assert "test_b" in names
    assert "test_c" in names


def test_split_filter_merging(csv_path: Path) -> None:
    """Split filter merges with existing dataloader filter.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    loader = PandasDataLoader(
        "test",
        {
            "path": str(csv_path),
            "filters": [{"column": "value", "values": [1]}],
            "split": {"by": "category", "values": ["a"]},
        },
    )
    selections = loader.get_selections()
    assert len(selections) == 1
    assert selections[0].filters_dict == {"value": [1], "category": "a"}


def test_bootstrap_applies_filter(csv_path: Path) -> None:
    """PandasDataSelection.bootstrap filters rows according to filters_dict.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    selection = PandasDataSelection(
        name="test_a",
        path=str(csv_path),
        filters_dict={"category": ["a"]},
    )
    selection.bootstrap()
    assert selection.data is not None
    assert list(selection.data["category"]) == ["a", "a"]
    assert list(selection.data["value"]) == [1, 5]


def test_iter_yields_filtered_batch(csv_path: Path) -> None:
    """Iterating a filtered selection yields only matching rows.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    selection = PandasDataSelection(
        name="test_b",
        path=str(csv_path),
        filters_dict={"category": ["b"]},
    )
    selection.bootstrap()
    batches = list(selection)
    assert len(batches) == 1
    assert batches[0].num_rows == 2
    assert batches[0].column("category").to_pylist() == ["b", "b"]


def test_bootstrap_applies_filter_with_multiple_values(csv_path: Path) -> None:
    """PandasDataSelection.bootstrap filters rows with multiple values.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    selection = PandasDataSelection(
        name="test_ab",
        path=str(csv_path),
        filters_dict={"category": ["a", "b"]},
    )
    selection.bootstrap()
    assert selection.data is not None
    assert set(selection.data["category"]) == {"a", "b"}
    assert len(selection.data) == 4


def test_bootstrap_applies_filter_with_wildcard(csv_path: Path) -> None:
    """PandasDataSelection.bootstrap filters rows with wildcard pattern.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    # Add some test data with prefixes
    df = pd.DataFrame(
        {
            "category": ["meta_a", "meta_b", "normal"],
            "value": [1, 2, 3],
        }
    )
    path = csv_path.parent / "test_wildcard.csv"
    df.to_csv(path, index=False)

    selection = PandasDataSelection(
        name="test_wildcard",
        path=str(path),
        filters_dict={"category": ["meta_*"]},
    )
    selection.bootstrap()
    assert selection.data is not None
    assert set(selection.data["category"]) == {"meta_a", "meta_b"}
    assert len(selection.data) == 2


def test_loader_filters_with_wildcard(csv_path: Path) -> None:
    """PandasDataLoader applies wildcard filters correctly.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    df = pd.DataFrame(
        {
            "category": ["dog_brown", "dog_black", "cat"],
            "value": [1, 2, 3],
        }
    )
    path = csv_path.parent / "test_loader_wildcard.csv"
    df.to_csv(path, index=False)

    loader = PandasDataLoader(
        "test",
        {"path": str(path), "filters": [{"column": "category", "values": ["dog_*"]}]},
    )
    selections = loader.get_selections()
    assert len(selections) == 1
    selection = selections[0]
    selection.bootstrap()
    assert set(selection.data["category"]) == {"dog_brown", "dog_black"}


# ---- _apply_pandas_transforms unit tests ----


def test_apply_no_transforms(csv_path: Path) -> None:
    """Empty transform list leaves DataFrame unchanged.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    df = pd.read_csv(str(csv_path))
    original = df.copy()
    _apply_pandas_transforms(df, [])
    pd.testing.assert_frame_equal(df, original)


def test_apply_in_place_cast(csv_path: Path) -> None:
    """in_place=True casts the column in place.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    df = pd.read_csv(str(csv_path))
    assert df["value"].dtype == "int64"
    _apply_pandas_transforms(df, [{"column": "value", "to_type": "float64", "in_place": True}])
    assert df["value"].dtype == "float64"
    assert list(df["value"]) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_apply_copy_cast(csv_path: Path) -> None:
    """in_place=False (default) adds new column, original unchanged.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    df = pd.read_csv(str(csv_path))
    assert df["value"].dtype == "int64"
    _apply_pandas_transforms(df, [{"column": "value", "to_type": "float32"}])
    assert "value_float32" in df.columns
    assert df["value"].dtype == "int64"
    assert list(df["value_float32"]) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_apply_missing_column_skipped(csv_path: Path) -> None:
    """Transform referencing a non-existent column is silently skipped.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    df = pd.read_csv(str(csv_path))
    original = df.copy()
    _apply_pandas_transforms(df, [{"column": "nonexistent", "to_type": "float64"}])
    pd.testing.assert_frame_equal(df, original)


def test_apply_all_type_mappings(csv_path: Path) -> None:
    """All 7 supported types map to valid pandas dtypes.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    df = pd.read_csv(str(csv_path))
    types = ["int32", "int64", "float32", "float64", "bool", "str", "categorical"]
    expected_dtypes = {
        "int32": "int32",
        "int64": "int64",
        "float32": "float32",
        "float64": "float64",
        "bool": "bool",
        "str": "string",
        "categorical": "category",
    }
    for t in types:
        df2 = df[["value"]].copy()
        _apply_pandas_transforms(df2, [{"column": "value", "to_type": t, "in_place": True}])
        assert df2["value"].dtype.name == expected_dtypes[t], f"Mismatch for {t}"


def test_apply_multiple_transforms(csv_path: Path) -> None:
    """Multiple transforms — one in_place, one copy — are both applied.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    df = pd.read_csv(str(csv_path))
    _apply_pandas_transforms(
        df,
        [
            {"column": "value", "to_type": "float64", "in_place": True},
            {"column": "category", "to_type": "str"},
        ],
    )
    assert df["value"].dtype == "float64"
    assert "category_str" in df.columns
    assert df["category_str"].dtype == "string"


# ---- Integration tests through PandasDataLoader / PandasDataSelection ----


def test_transform_through_selection_iter(csv_path: Path) -> None:
    """PandasDataSelection.__iter__ applies transforms to RecordBatch.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    selection = PandasDataSelection(
        name="test",
        path=str(csv_path),
        transforms=[{"column": "value", "to_type": "float64"}],
    )
    selection.bootstrap()
    batches = list(selection)
    assert len(batches) == 1
    batch = batches[0]
    assert batch.column("value_float64") is not None
    assert batch.column("value_float64").to_pylist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    # Original column still present
    assert batch.schema.get_field_index("value") >= 0


def test_transform_does_not_mutate_original(csv_path: Path) -> None:
    """__iter__ does not modify the underlying DataFrame when transforms are set.

    Args:
        csv_path: Fixture providing path to test CSV file.
    """
    selection = PandasDataSelection(
        name="test",
        path=str(csv_path),
        transforms=[{"column": "value", "to_type": "float64"}],
    )
    selection.bootstrap()
    original_dtypes = selection.data.dtypes.to_dict()
    list(selection)  # consume iterator
    after_dtypes = selection.data.dtypes.to_dict()
    assert original_dtypes == after_dtypes
