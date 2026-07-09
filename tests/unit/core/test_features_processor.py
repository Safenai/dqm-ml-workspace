"""Unit tests for the FeaturesProcessor base class.

This module contains tests that verify the FeaturesProcessor correctly
handles feature extraction, column resolution, output naming, and
image fail-fast error handling.
"""

from dqm_ml_core.api.features_processor import FeaturesProcessor
from dqm_ml_core.models.global_ import ErrorsConfig, ImageErrorsConfig
import pyarrow as pa
import pytest


def _make_proc(extra_config: dict | None = None) -> FeaturesProcessor:
    """Create a FeaturesProcessor with optional config overrides."""
    config = {"columns": {"input": ["a", "b", "c"]}}
    if extra_config:
        cols = config.setdefault("columns", {})
        cols.update(extra_config)
    return FeaturesProcessor(name="test", config=config)


def test_generated_features_empty():
    """Verify generated_features returns empty list when no output_features configured."""
    proc = FeaturesProcessor(name="test", config={})
    assert proc.generated_features() == []


def test_compute_features_selects_columns():
    """Verify compute_features extracts matching columns from batch."""
    proc = _make_proc()
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2]), pa.array([3])], names=["a", "b", "c"])
    features = proc.compute_features(batch, {})
    assert "a" in features
    assert "b" in features
    assert "c" in features


def test_compute_features_skips_prev_features():
    """Verify columns in prev_features are skipped."""
    proc = _make_proc()
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2]), pa.array([3])], names=["a", "b", "c"])
    prev = {"a": pa.array([10])}
    features = proc.compute_features(batch, prev)
    assert "a" not in features  # already in prev_features
    assert "b" in features
    assert "c" in features


def test_compute_features_wildcard():
    """Verify wildcard '*' matches all batch columns."""
    proc = FeaturesProcessor(name="test", config={"columns": {"input": ["*"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2])], names=["a", "b"])
    features = proc.compute_features(batch, {})
    assert "a" in features
    assert "b" in features


def test_compute_features_exclude():
    """Verify exclude filters out matched columns."""
    proc = FeaturesProcessor(name="test", config={"columns": {"input": ["a", "b", "c"], "exclude": ["b"]}})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2]), pa.array([3])], names=["a", "b", "c"])
    features = proc.compute_features(batch, {})
    assert "a" in features
    assert "b" not in features
    assert "c" in features


def test_compute_features_empty_input():
    """Verify empty input list matches all batch columns."""
    proc = FeaturesProcessor(name="test", config={"columns": {"input": []}})
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    features = proc.compute_features(batch, {})
    assert "a" in features


def test_compute_features_no_input_config():
    """Verify missing input_columns defaults to all batch columns."""
    proc = FeaturesProcessor(name="test", config={})
    batch = pa.RecordBatch.from_arrays([pa.array([1]), pa.array([2])], names=["a", "b"])
    features = proc.compute_features(batch, {})
    assert "a" in features
    assert "b" in features


def test_resolve_output_name_prefix():
    """Verify _resolve_output_name applies prefix from columns_config."""
    proc = FeaturesProcessor(
        name="test",
        config={"columns": {"input": ["col"], "prefix": "pfx_"}},
    )
    result = proc._resolve_output_name("col", "feat")
    assert result == "pfx_col_feat"


def test_resolve_output_name_suffix():
    """Verify _resolve_output_name applies suffix from columns_config."""
    proc = FeaturesProcessor(
        name="test",
        config={"columns": {"input": ["col"], "suffix": "_sfx"}},
    )
    result = proc._resolve_output_name("col", "feat")
    assert result == "col_feat_sfx"


def test_resolve_output_name_prefix_and_suffix():
    """Verify _resolve_output_name applies both prefix and suffix."""
    proc = FeaturesProcessor(
        name="test",
        config={"columns": {"input": ["col"], "prefix": "pfx_", "suffix": "_sfx"}},
    )
    result = proc._resolve_output_name("col", "feat")
    assert result == "pfx_col_feat_sfx"


def test_resolve_output_name_no_config():
    """Verify _resolve_output_name returns base name with no prefix/suffix."""
    proc = FeaturesProcessor(name="test", config={"columns": {"input": ["col"]}})
    result = proc._resolve_output_name("col", "feat")
    assert result == "col_feat"


def test_check_image_fail_fast_raises():
    """Verify image fail fast raises ValueError."""
    proc = FeaturesProcessor(
        name="test",
        config={"errors": {"images": {"on_decode_failure": "fail_fast"}}},
    )
    proc.errors_config = ErrorsConfig(images=ImageErrorsConfig(on_decode_failure="fail_fast"))
    error = ValueError("test error")
    with pytest.raises(ValueError, match="test error"):
        proc._check_image_fail_fast(error, "on_decode_failure")


def test_check_image_fail_fast_silent_does_not_raise():
    """Verify image silent_fail does not raise."""
    proc = FeaturesProcessor(
        name="test",
        config={"errors": {"images": {"on_decode_failure": "silent_fail"}}},
    )
    proc.errors_config = ErrorsConfig(images=ImageErrorsConfig(on_decode_failure="silent_fail"))
    proc._check_image_fail_fast(ValueError("test error"), "on_decode_failure")
