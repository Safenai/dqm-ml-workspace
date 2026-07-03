"""Unit tests for the completeness metric processor.

This module contains tests that verify the completeness metric processor
correctly computes completeness scores for tabular data, including handling
of missing columns, null values, NaN values, and edge cases.
"""

import logging

from dqm_ml_core.metrics.completeness import CompletenessProcessor
import pyarrow as pa
import pytest


def test_select_columns_column_not_in_batch(caplog):
    """Verify select_columns skips missing columns and logs a warning.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = CompletenessProcessor(name="test", config={"columns": {"input": ["col1", "missing"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2]), pa.array([3, 4])], names=["col1", "col2"])

    with caplog.at_level(logging.WARNING):
        features = proc.select_columns(batch)

    assert "col1" in features
    assert "missing" not in features
    assert "column 'missing' not found in batch" in caplog.text


def test_compute_column_completeness_missing_keys():
    """Verify _compute_column_completeness returns None when batch metrics keys are missing."""
    batch_metrics = {}
    result = CompletenessProcessor._compute_column_completeness("col1", batch_metrics)
    assert result is None


def test_compute_empty_batch_metrics():
    """Verify compute returns error metadata when batch_metrics is None."""
    proc = CompletenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    result = proc.compute(batch_metrics=None)
    assert result == {"_metadata": {"error": "No batch metrics provided"}}


def test_compute_no_columns_found():
    """Verify compute returns error when no configured columns exist in batch metrics."""
    proc = CompletenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    batch_metrics = {"other_metric": pa.array([1])}
    result = proc.compute(batch_metrics=batch_metrics)
    assert "_metadata" in result
    assert result["_metadata"]["error"] == "No columns found in batch metrics"


def test_compute_with_missing_column_metrics(caplog):
    """Verify compute handles missing complete_count gracefully and logs warning.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = CompletenessProcessor(
        name="test",
        config={
            "columns": {"input": ["col1"]},
            "include_per_column": False,
            "include_overall": False,
        },
    )
    batch_metrics = {
        "col1_total_count": pa.array([10]),
    }
    with caplog.at_level(logging.WARNING):
        result = proc.compute(batch_metrics=batch_metrics)

    assert "col1_complete_count" not in batch_metrics
    assert result == {}
    assert "Missing batch metrics" in caplog.text


def test_compute_success():
    """Verify compute returns correct completeness scores for valid batch metrics."""
    proc = CompletenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    batch_metrics = {
        "col1_total_count": pa.array([10]),
        "col1_complete_count": pa.array([8]),
    }
    result = proc.compute(batch_metrics=batch_metrics)
    assert result["completeness_col1"] == pytest.approx(0.8)
    assert result["completeness_overall"] == pytest.approx(0.8)


def test_compute_batch_metric_detects_float_nan():
    """Verify compute_batch_metric treats float NaN as incomplete (not null).

    Float NaN values must NOT be counted as complete since they represent
    missing/invalid data in floating-point columns.
    """
    proc = CompletenessProcessor(name="test", config={"columns": {"input": ["col1"]}})

    # Create Arrow array with float NaN (not Arrow null)
    arr = pa.array([1.0, 2.0, float("nan"), 4.0, 5.0], type=pa.float64())
    features = {"col1": arr}

    metrics = proc.compute_batch_metric(features)
    assert metrics["col1_total_count"][0].as_py() == 5
    assert metrics["col1_complete_count"][0].as_py() == 4


def test_compute_batch_metric_integer_column():
    """Verify compute_batch_metric handles integer columns without is_nan calls.

    Integer columns should not call is_nan (NaN is undefined for ints).
    Only Arrow null values are treated as incomplete.
    """
    proc = CompletenessProcessor(name="test", config={"columns": {"input": ["col1"]}})

    # Integer array with an Arrow null
    arr = pa.array([1, 2, None, 4], type=pa.int64())
    features = {"col1": arr}

    metrics = proc.compute_batch_metric(features)
    assert metrics["col1_total_count"][0].as_py() == 4
    assert metrics["col1_complete_count"][0].as_py() == 3


class TestExtremeDataValues:
    """Tests for edge cases and extreme data values in completeness computation.

    Covers zero totals, all-complete/all-null columns, missing metric keys,
    and boundary conditions preventing division errors.
    """

    def test_zero_total_returns_zero(self) -> None:
        """Verify zero total count returns zero completeness without division error."""
        result = CompletenessProcessor._compute_column_completeness(
            "col1",
            {
                "col1_total_count": pa.array([0]),
                "col1_complete_count": pa.array([0]),
            },
        )
        assert result == pytest.approx(0.0)

    def test_all_complete(self) -> None:
        """Verify 100% completeness when all values are present."""
        result = CompletenessProcessor._compute_column_completeness(
            "col1",
            {
                "col1_total_count": pa.array([10]),
                "col1_complete_count": pa.array([10]),
            },
        )
        assert result == pytest.approx(1.0)

    def test_all_null(self) -> None:
        """Verify 0% completeness when all values are null."""
        result = CompletenessProcessor._compute_column_completeness(
            "col1",
            {
                "col1_total_count": pa.array([10]),
                "col1_complete_count": pa.array([0]),
            },
        )
        assert result == pytest.approx(0.0)

    def test_missing_total_key(self) -> None:
        """Verify returns None when total_count key is missing from batch metrics."""
        result = CompletenessProcessor._compute_column_completeness(
            "col1",
            {"col1_complete_count": pa.array([5])},
        )
        assert result is None

    def test_missing_complete_key(self) -> None:
        """Verify returns None when complete_count key is missing from batch metrics."""
        result = CompletenessProcessor._compute_column_completeness(
            "col1",
            {"col1_total_count": pa.array([5])},
        )
        assert result is None
