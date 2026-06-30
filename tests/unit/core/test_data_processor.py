"""Unit tests for the DatametricProcessor base class.

This module contains unit tests that verify the DatametricProcessor
correctly handles initialization, configuration validation, and method behaviors.
"""

import logging

from dqm_ml_core import DatametricProcessor
from dqm_ml_core.models.global_ import ErrorsConfig, ImageErrorsConfig
import pyarrow as pa
import pytest


def test_processor_compute_features(caplog):
    """Test that compute_features correctly extracts columns from batch.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = DatametricProcessor(name="test", config={"columns": {"input": ["col1", "col2", "missing"]}})

    batch = pa.RecordBatch.from_arrays([pa.array([1, 2]), pa.array([3, 4])], names=["col1", "col2"])

    prev_features = {"col1": pa.array([1, 2])}

    with caplog.at_level(logging.WARNING):
        features = proc.compute_features(batch, prev_features)

    # col1 is in prev_features, so skipped
    # col2 is in batch, so added
    # missing is not in batch, so logged and skipped
    assert "col1" not in features
    assert "col2" in features
    assert features["col2"].to_pylist() == [3, 4]
    assert "column 'missing' not found in batch" in caplog.text


def test_processor_input_columns_wildcard():
    """Test that wildcard '*' matches all columns in compute_features."""
    proc = DatametricProcessor(name="test", config={"columns": {"input": ["*"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2]), pa.array([3])], names=["a", "b", "c"])
    features = proc.compute_features(batch, {})
    assert "a" in features
    assert "b" in features
    assert "c" in features


@pytest.mark.parametrize(
    ("config", "expected_cols"),
    [
        ({}, ["a", "b"]),
        ({"columns": {"input": []}}, ["a", "b"]),  # [] or None → None → all columns
    ],
)
def test_processor_input_columns_empty_or_default_compute_features(config, expected_cols):
    """Test that empty/missing input_columns defaults to all columns."""
    proc = DatametricProcessor(name="test", config=config)
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2])], names=["a", "b"])
    features = proc.compute_features(batch, {})
    for col in expected_cols:
        assert col in features
    if not expected_cols:
        assert features == {}


def test_processor_compute_features_exclude_literal(caplog):
    """Test that exclude filters out a literal column from features.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = DatametricProcessor(name="test", config={"columns": {"input": ["a", "b", "c"], "exclude": ["b"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2]), pa.array([3])], names=["a", "b", "c"])
    features = proc.compute_features(batch, {})
    assert "a" in features
    assert "b" not in features
    assert "c" in features


def test_processor_compute_features_exclude_wildcard(caplog):
    """Test that exclude supports wildcard patterns.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = DatametricProcessor(name="test", config={"columns": {"input": ["a", "meta_x"], "exclude": ["meta_*"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2])], names=["a", "meta_x"])
    features = proc.compute_features(batch, {})
    assert "a" in features
    assert "meta_x" not in features


def test_processor_compute_features_exclude_no_match(caplog):
    """Test that exclude with no matching patterns keeps all.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = DatametricProcessor(name="test", config={"columns": {"input": ["a", "b"], "exclude": ["z"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2])], names=["a", "b"])
    features = proc.compute_features(batch, {})
    assert "a" in features
    assert "b" in features


def test_processor_compute_features_exclude_none(caplog):
    """Test that missing exclude key keeps all input columns.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = DatametricProcessor(name="test", config={"columns": {"input": ["a"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    features = proc.compute_features(batch, {})
    assert "a" in features


def test_processor_default_implementations():
    proc = DatametricProcessor(name="test", config={})
    assert proc.compute_batch_metric({}) == {}
    assert proc.compute({}) == {}
    assert proc.compute_delta({}, {}) == {}


def test_check_failure_rate_exceeds_threshold():
    proc = DatametricProcessor(name="test", config={"errors": {"max_failure_rate": 0.1}})
    proc.errors_config = ErrorsConfig(max_failure_rate=0.1)
    proc._failure_count = 3
    proc._total_count = 10
    with pytest.raises(RuntimeError, match="Failure rate"):
        proc._check_failure_rate()


def test_check_image_fail_fast_raises():
    proc = DatametricProcessor(
        name="test",
        config={"errors": {"images": {"on_decode_failure": "fail_fast"}}},
    )
    proc.errors_config = ErrorsConfig(images=ImageErrorsConfig(on_decode_failure="fail_fast"))
    with pytest.raises(ValueError, match="test error"):
        proc._check_image_fail_fast(ValueError("test error"), "on_decode_failure")
