"""Unit tests for runtime error handling policies.

Tests that verify each error handling parameter (fail_fast, silent_fail,
max_failure_rate) is correctly consumed at runtime by the processors.
"""

import logging

from dqm_ml_images import VisualFeaturesProcessor
from dqm_ml_pytorch import ImageEmbeddingProcessor
import pyarrow as pa
import pytest

from dqm_ml_core import DatametricProcessor
from dqm_ml_core.models.global_ import ErrorsConfig, ImageErrorsConfig, TabularErrorsConfig

# --- on_missing_column (TabularErrorsConfig) ---


class TestOnMissingColumn:
    """Tests for on_missing_column: fail_fast (raises KeyError) and silent_fail (logs warning, skips column)."""

    def test_fail_fast_raises_key_error(self) -> None:
        """Verify fail_fast raises KeyError for missing column."""
        proc = DatametricProcessor(name="test", config={"columns": {"input": ["missing"]}})
        proc.errors_config = ErrorsConfig(
            tabular=TabularErrorsConfig(on_missing_column="fail_fast"),
        )
        batch = pa.RecordBatch.from_arrays([pa.array([1, 2])], names=["a"])

        with pytest.raises(KeyError, match="'missing' not found"):
            proc.compute_features(batch, {})

    def test_silent_fail_logs_and_skips(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify silent_fail logs warning and skips missing column.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        proc = DatametricProcessor(name="test", config={"columns": {"input": ["a", "missing"]}})
        proc.errors_config = ErrorsConfig(
            tabular=TabularErrorsConfig(on_missing_column="silent_fail"),
        )
        batch = pa.RecordBatch.from_arrays([pa.array([1, 2])], names=["a"])

        with caplog.at_level(logging.WARNING):
            features = proc.compute_features(batch, {})

        assert "a" in features
        assert "missing" not in features
        assert "column 'missing' not found in batch" in caplog.text


# --- on_file_not_found (TabularErrorsConfig) ---


class TestOnFileNotFound:
    """Tests for on_file_not_found: fail_fast (raises ValueError) and silent_fail (logs warning, returns None)."""

    def test_fail_fast_raises_value_error(self) -> None:
        """Verify fail_fast raises ValueError for nonexistent file path."""
        proc = VisualFeaturesProcessor(name="test", config={"name": "test"})
        proc.errors_config = ErrorsConfig(
            tabular=TabularErrorsConfig(on_file_not_found="fail_fast"),
        )

        with pytest.raises(ValueError, match="Path does not exist"):
            proc._to_gray_np("/nonexistent/file.jpg")

    def test_silent_fail_logs_and_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify silent_fail logs warning and returns None for nonexistent file.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        proc = VisualFeaturesProcessor(name="test", config={"name": "test"})
        proc.errors_config = ErrorsConfig(
            max_failure_rate=1.0,
            tabular=TabularErrorsConfig(on_file_not_found="silent_fail"),
            images=ImageErrorsConfig(on_unsupported_format="silent_fail"),
        )

        with caplog.at_level(logging.WARNING):
            result = proc._process_single_image(0, "/nonexistent/file.jpg")

        assert result is None
        assert "Path does not exist" in caplog.text


# --- on_transform_error (ImageErrorsConfig) ---


class TestOnTransformError:
    """Tests for on_transform_error: fail_fast (raises ValueError) and silent_fail (logs warning, returns None)."""

    def test_fail_fast_raises(self) -> None:
        """Verify fail_fast raises ValueError for unsupported transform input type."""
        proc = VisualFeaturesProcessor(name="test", config={"name": "test"})
        proc.errors_config = ErrorsConfig(
            tabular=TabularErrorsConfig(on_file_not_found="silent_fail"),
            images=ImageErrorsConfig(on_transform_error="fail_fast", on_unsupported_format="silent_fail"),
        )

        with pytest.raises(ValueError, match="Unsupported type"):
            proc._process_single_image(0, 42)

    def test_silent_fail_logs_and_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify silent_fail logs warning, returns None, and tracks failure count.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        proc = VisualFeaturesProcessor(name="test", config={"name": "test"})
        proc.errors_config = ErrorsConfig(
            max_failure_rate=1.0,
            tabular=TabularErrorsConfig(on_file_not_found="silent_fail"),
            images=ImageErrorsConfig(on_transform_error="silent_fail", on_unsupported_format="silent_fail"),
        )

        with caplog.at_level(logging.WARNING):
            result = proc._process_single_image(0, 42)

        assert result is None
        assert proc._failure_count == 1
        assert proc._total_count == 1


# --- on_unsupported_format (ImageErrorsConfig) ---


class TestOnUnsupportedFormat:
    """Tests for on_unsupported_format: fail_fast (raises ValueError) and silent_fail (logs warning, returns None)."""

    def test_fail_fast_raises(self) -> None:
        """Verify fail_fast raises ValueError for unsupported format input type."""
        proc = VisualFeaturesProcessor(name="test", config={"name": "test"})
        proc.errors_config = ErrorsConfig(
            tabular=TabularErrorsConfig(on_file_not_found="silent_fail"),
            images=ImageErrorsConfig(on_transform_error="silent_fail", on_unsupported_format="fail_fast"),
        )

        with pytest.raises(ValueError, match="Unsupported type"):
            proc._process_single_image(0, 42)

    def test_silent_fail_logs_and_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify silent_fail logs warning, returns None, and tracks failure count.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        proc = VisualFeaturesProcessor(name="test", config={"name": "test"})
        proc.errors_config = ErrorsConfig(
            max_failure_rate=1.0,
            tabular=TabularErrorsConfig(on_file_not_found="silent_fail"),
            images=ImageErrorsConfig(on_transform_error="silent_fail", on_unsupported_format="silent_fail"),
        )

        with caplog.at_level(logging.WARNING):
            result = proc._process_single_image(0, 42)

        assert result is None
        assert proc._failure_count == 1
        assert proc._total_count == 1


# --- on_decode_failure (ImageErrorsConfig) ---


class TestOnDecodeFailure:
    """Tests for on_decode_failure: fail_fast (raises OSError) and silent_fail (logs warning, appends None)."""

    def test_fail_fast_raises(self) -> None:
        """Verify fail_fast raises OSError for invalid image data."""
        proc = ImageEmbeddingProcessor(name="test", config={"name": "test"})
        proc.errors_config = ErrorsConfig(
            images=ImageErrorsConfig(on_decode_failure="fail_fast", on_transform_error="silent_fail"),
        )

        with pytest.raises(OSError, match="cannot identify image file"):
            proc._load_image_tensors([b""])

    def test_silent_fail_logs_and_appends_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify silent_fail logs warning, appends None, and tracks failure count.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        proc = ImageEmbeddingProcessor(name="test", config={"name": "test"})
        proc.errors_config = ErrorsConfig(
            max_failure_rate=1.0,
            images=ImageErrorsConfig(on_decode_failure="silent_fail", on_transform_error="silent_fail"),
        )

        with caplog.at_level(logging.WARNING):
            result = proc._load_image_tensors([b""])

        assert result == [None]
        assert proc._failure_count == 1
        assert proc._total_count == 1
        assert "failed to load image" in caplog.text


# --- max_failure_rate (ErrorsConfig) ---


class TestMaxFailureRate:
    """Tests for max_failure_rate: exceeded threshold (raises RuntimeError) and zero tolerance (any failure raises)."""

    def test_exceeded_raises_runtime_error(self) -> None:
        """Verify exceeded failure rate raises RuntimeError."""
        proc = DatametricProcessor(name="test", config={})
        proc.errors_config = ErrorsConfig(max_failure_rate=0.05)
        proc._failure_count = 6
        proc._total_count = 100

        with pytest.raises(RuntimeError, match="exceeds max"):
            proc._check_failure_rate()

    def test_zero_tolerance_any_failure_raises(self) -> None:
        """Verify zero tolerance raises RuntimeError on any failure."""
        proc = DatametricProcessor(name="test", config={})
        proc.errors_config = ErrorsConfig(max_failure_rate=0.0)
        proc._failure_count = 1
        proc._total_count = 1

        with pytest.raises(RuntimeError, match="exceeds max"):
            proc._check_failure_rate()
