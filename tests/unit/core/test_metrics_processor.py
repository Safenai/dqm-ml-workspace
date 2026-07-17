"""Unit tests for the MetricsProcessor base class.

This module contains unit tests that verify the MetricsProcessor
correctly handles initialization, configuration validation, and method behaviors.
"""

import logging

from dqm_ml_core.api.metrics_processor import MetricsProcessor
from dqm_ml_core.models.global_ import ErrorsConfig, TabularErrorsConfig
import pyarrow as pa
import pytest


def test_select_columns_column_not_in_batch(caplog):
    """Test that select_columns skips missing columns and logs a warning.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = MetricsProcessor(name="test", config={"columns": {"input": ["col1", "col2", "missing"]}})

    batch = pa.RecordBatch.from_arrays([pa.array([1, 2]), pa.array([3, 4])], names=["col1", "col2"])

    prev_features = {"col1": pa.array([1, 2])}

    with caplog.at_level(logging.WARNING):
        features = proc.select_columns(batch, prev_features)

    # col1 is in prev_features, so skipped
    # col2 is in batch, so added
    # missing is not in batch, so logged and skipped
    assert "col1" not in features
    assert "col2" in features
    assert features["col2"].to_pylist() == [3, 4]
    assert "column 'missing' not found in batch" in caplog.text


def test_input_columns_wildcard():
    """Test that wildcard '*' matches all columns in select_columns."""
    proc = MetricsProcessor(name="test", config={"columns": {"input": ["*"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2]), pa.array([3])], names=["a", "b", "c"])
    features = proc.select_columns(batch, {})
    assert "a" in features
    assert "b" in features
    assert "c" in features


def test_select_columns_exclude_literal(caplog):
    """Test that exclude filters out a literal column from features.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = MetricsProcessor(
        name="test",
        config={"columns": {"input": ["a", "b", "c"], "exclude": ["b"]}},
    )
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2]), pa.array([3])], names=["a", "b", "c"])
    features = proc.select_columns(batch, {})
    assert "a" in features
    assert "b" not in features
    assert "c" in features


def test_select_columns_exclude_wildcard(caplog):
    """Test that exclude supports wildcard patterns.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = MetricsProcessor(
        name="test",
        config={"columns": {"input": ["a", "meta_x"], "exclude": ["meta_*"]}},
    )
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2])], names=["a", "meta_x"])
    features = proc.select_columns(batch, {})
    assert "a" in features
    assert "meta_x" not in features


def test_select_columns_exclude_no_match(caplog):
    """Test that exclude with no matching patterns keeps all.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = MetricsProcessor(name="test", config={"columns": {"input": ["a", "b"], "exclude": ["z"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2])], names=["a", "b"])
    features = proc.select_columns(batch, {})
    assert "a" in features
    assert "b" in features


def test_select_columns_exclude_none(caplog):
    """Test that missing exclude key keeps all input columns.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = MetricsProcessor(name="test", config={"columns": {"input": ["a"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    features = proc.select_columns(batch, {})
    assert "a" in features


def test_default_implementations():
    proc = MetricsProcessor(name="test", config={})
    batch = pa.RecordBatch.from_arrays([], names=[])
    assert proc.select_columns(batch, {}) == {}
    assert proc.compute_batch_metric({}) == {}
    assert proc.compute({}) == {}


def test_check_failure_rate_exceeds_threshold():
    proc = MetricsProcessor(name="test", config={"errors": {"max_failure_rate": 0.1}})
    proc.errors_config = ErrorsConfig(max_failure_rate=0.1)
    proc._failure_count = 3
    proc._total_count = 10
    with pytest.raises(RuntimeError, match="Failure rate"):
        proc._check_failure_rate()


def test_on_missing_column_fail_fast_raises():
    """Verify fail_fast raises KeyError for missing column in select_columns."""
    proc = MetricsProcessor(name="test", config={"columns": {"input": ["missing"]}})
    proc.errors_config = ErrorsConfig(
        tabular=TabularErrorsConfig(on_missing_column="fail_fast"),
    )
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2])], names=["a"])

    with pytest.raises(KeyError, match="'missing' not found"):
        proc.select_columns(batch, {})


def test_on_missing_column_silent_fail_logs_and_skips(caplog):
    """Verify silent_fail logs warning and skips missing column.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = MetricsProcessor(name="test", config={"columns": {"input": ["a", "missing"]}})
    proc.errors_config = ErrorsConfig(
        tabular=TabularErrorsConfig(on_missing_column="silent_fail"),
    )
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2])], names=["a"])

    with caplog.at_level(logging.WARNING):
        features = proc.select_columns(batch, {})

    assert "a" in features
    assert "missing" not in features
    assert "column 'missing' not found in batch" in caplog.text
