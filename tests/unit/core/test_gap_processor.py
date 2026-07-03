"""Unit tests for the GapProcessor base class.

This module contains tests that verify the GapProcessor correctly
handles feature extraction with cross-resolution, and default
implementations of batch metric, compute, and delta methods.
"""

import logging

from dqm_ml_core.api.gap_processor import GapProcessor
import pyarrow as pa


def test_select_features_from_batch():
    """Verify select_features resolves columns from batch."""
    proc = GapProcessor(name="test", config={"columns": {"input": ["a", "b"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2])], names=["a", "b"])
    features = proc.select_features(batch, {})
    assert "a" in features
    assert "b" in features


def test_select_features_from_prev_features():
    """Verify select_features skips columns in prev_features (resolved from there)."""
    proc = GapProcessor(name="test", config={"columns": {"input": ["a", "b"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    prev = {"b": pa.array([10])}
    features = proc.select_features(batch, prev)
    assert "a" in features
    assert "b" not in features  # in prev_features, skip for batch extraction


def test_select_features_wildcard():
    """Verify wildcard resolves across both batch columns and prev_features."""
    proc = GapProcessor(name="test", config={"columns": {"input": ["*"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    prev = {"b": pa.array([10])}
    features = proc.select_features(batch, prev)
    assert "a" in features
    assert "b" not in features  # prev_features columns are not re-extracted from batch


def test_select_features_empty_input():
    """Verify empty input list returns empty dict."""
    proc = GapProcessor(name="test", config={"columns": {"input": []}})
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    features = proc.select_features(batch, {})
    assert features == {}


def test_select_features_no_input_config():
    """Verify missing input_columns defaults to all batch columns."""
    proc = GapProcessor(name="test", config={})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2])], names=["a", "b"])
    features = proc.select_features(batch, {})
    assert "a" in features
    assert "b" in features


def test_select_features_column_missing_logs_warning(caplog):
    """Verify warning logged when input column not in batch nor prev_features."""
    proc = GapProcessor(name="test", config={"columns": {"input": ["missing"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    with caplog.at_level(logging.WARNING):
        features = proc.select_features(batch, {})
    assert features == {}
    assert "column 'missing' not found in batch" in caplog.text


def test_compute_batch_metric_default():
    """Verify default compute_batch_metric returns empty dict."""
    proc = GapProcessor(name="test", config={})
    assert proc.compute_batch_metric({}) == {}


def test_compute_default():
    """Verify default compute returns empty dict."""
    proc = GapProcessor(name="test", config={})
    assert proc.compute({}) == {}


def test_compute_delta_default():
    """Verify default compute_delta returns empty dict."""
    proc = GapProcessor(name="test", config={})
    assert proc.compute_delta({}, {}) == {}
