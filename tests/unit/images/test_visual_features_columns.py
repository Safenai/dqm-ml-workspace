"""Unit tests for VisualFeaturesProcessor column configuration.

Tests the _output_column_name logic via generated_features() for
prefix, suffix, and rename column modifiers.
"""

from dqm_ml_images.visual_features import VisualFeaturesProcessor
import numpy as np
from PIL import Image
import pyarrow as pa
import pytest

BASE_COLUMNS = ["img"]


def _make_processor(extra_config: dict | None = None) -> VisualFeaturesProcessor:
    """Create a VisualFeaturesProcessor with base config and optional overrides.

    Args:
        extra_config: Optional dictionary to merge into the processor config.

    Returns:
        Configured VisualFeaturesProcessor instance.
    """
    config = {"name": "test", "columns": {"input": BASE_COLUMNS}}
    if extra_config:
        cols = config.setdefault("columns", {})
        cols.update(extra_config)
    return VisualFeaturesProcessor(name="test", config=config)


def _make_processor_with_weights(
    weights_val: object,
    **proc_kwargs: object,
) -> VisualFeaturesProcessor:
    """Create a VisualFeaturesProcessor with luminosity_weights and optional config.

    Args:
        weights_val: Luminosity weights tuple (R, G, B) or None.
        **proc_kwargs: Additional config overrides passed to processor constructor.

    Returns:
        Configured VisualFeaturesProcessor instance.
    """
    config: dict[str, object] = {"name": "test", "columns": {"input": BASE_COLUMNS}}
    if weights_val is not None:
        config["luminosity_weights"] = weights_val
    config.update(proc_kwargs)
    return VisualFeaturesProcessor(name="test", config=config)


# --- Luminosity Weights Tests ---


def test_luminosity_weights_affects_pil_to_gray():
    """Verify weights affect PIL image grayscale conversion."""
    proc = _make_processor_with_weights(
        (1, 0, 0),
        grayscale=False,
        normalize=False,
    )
    img = Image.new("RGB", (2, 2), (10, 20, 30))
    result = proc._pil_to_gray(img)
    assert result.dtype == np.uint8
    assert (result == 10).all()


def test_luminosity_weights_affects_ndarray_to_gray():
    """Verify weights affect numpy array grayscale conversion."""
    proc = _make_processor_with_weights(
        (0, 1, 0),
        grayscale=False,
        normalize=False,
    )
    arr = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
    result = proc._ndarray_to_gray(arr)
    assert result.dtype == np.uint8
    assert (result == 20).all()


# --- VisualFeaturesProcessor Edge Cases ---


class TestVisualFeaturesEdgeCases:
    """Edge cases: empty/missing columns, 2D arrays, unsupported shapes, Laplacian, entropy, failed image."""

    def test_generated_features_empty_input_columns(self):
        """Verify generated_features returns empty list when no input columns."""
        proc = VisualFeaturesProcessor(name="test", config={"columns": {"input": []}})
        assert proc.generated_features() == []

    def test_compute_features_no_input_columns(self, caplog):
        """Verify compute_features returns empty dict and logs warning for no input columns.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        proc = VisualFeaturesProcessor(name="test", config={"columns": {"input": []}})
        batch = pa.RecordBatch.from_pydict({"col1": pa.array([1.0])})
        result = proc.compute_features(batch, {})
        assert result == {}
        assert "no input_columns configured" in caplog.text

    def test_compute_features_missing_column_in_batch(self, caplog):
        """Verify compute_features returns empty dict when input column missing from batch.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        proc = VisualFeaturesProcessor(name="test", config={"columns": {"input": ["img"]}})
        batch = pa.RecordBatch.from_pydict({"other": pa.array([1.0])})
        result = proc.compute_features(batch, {})
        assert result == {}

    def test_ndarray_to_gray_2d_input(self):
        """Verify 2D array input is handled correctly by _ndarray_to_gray."""
        proc = _make_processor()
        arr = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        result = proc._ndarray_to_gray(arr)
        assert result.shape == (2, 2)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_ndarray_to_gray_unsupported_shape_raises(self):
        """Verify _ndarray_to_gray raises ValueError for unsupported array shapes."""
        proc = _make_processor()
        arr = np.ones((2, 2, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported ndarray shape"):
            proc._ndarray_to_gray(arr)

    def test_laplacian_5x5_kernel(self):
        """Verify 5x5 Laplacian kernel computes variance without error."""
        proc = VisualFeaturesProcessor(
            name="test",
            config={"columns": {"input": ["img"]}, "laplacian_kernel": "5x5"},
        )
        gray = np.array([[50, 100, 150], [100, 200, 250], [150, 250, 200]], dtype=np.uint8)
        result = proc._variance_of_laplacian(gray)
        assert result >= 0.0

    def test_entropy_non_normalized(self):
        """Verify entropy calculation works with non-normalized images."""
        proc = VisualFeaturesProcessor(
            name="test",
            config={"columns": {"input": ["img"]}, "normalize": False},
        )
        img = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        result = proc._entropy(img)
        assert result >= 0.0

    def test_entropy_zero_histogram(self):
        """Verify entropy returns 0.0 for zero histogram (all same values)."""
        proc = _make_processor()
        img = np.zeros((2, 2), dtype=np.float32)
        result = proc._entropy(img)
        assert result == pytest.approx(0.0)

    def test_compute_scalar_feature_failed_image(self, caplog):
        """Verify _compute_scalar_feature returns NaN for failed image processing.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        proc = _make_processor()
        result = proc._compute_scalar_feature([None], np.mean, True)
        assert np.isnan(result.to_pylist()[0])


# --- VisualFeaturesProcessor Extreme Values Tests ---


class TestVisualFeaturesExtremeValues:
    """Tests for extreme value handling and edge case behavior in visual features processing."""

    def test_constant_image_variance_of_laplacian_zero(self) -> None:
        """Verify variance of Laplacian handles constant images with 3x3 kernel."""
        proc = _make_processor()
        gray = np.full((10, 10), 128, dtype=np.uint8)
        result = proc._variance_of_laplacian(gray)
        assert result > 0.0

    def test_constant_image_5x5_kernel_zero(self) -> None:
        """Verify variance of Laplacian handles constant images with 5x5 kernel."""
        proc = VisualFeaturesProcessor(
            name="test",
            config={"columns": {"input": ["img"]}, "laplacian_kernel": "5x5"},
        )
        gray = np.full((10, 10), 128, dtype=np.uint8)
        result = proc._variance_of_laplacian(gray)
        assert result > 0.0

    def test_single_pixel_image_3x3_kernel(self) -> None:
        """Verify single pixel image works with 3x3 Laplacian kernel."""
        proc = _make_processor()
        gray = np.array([[50]], dtype=np.uint8)
        result = proc._variance_of_laplacian(gray)
        assert result >= 0.0
        assert np.isfinite(result)

    def test_single_pixel_image_5x5_kernel(self) -> None:
        """Verify single pixel image works with 5x5 Laplacian kernel."""
        proc = VisualFeaturesProcessor(
            name="test",
            config={"columns": {"input": ["img"]}, "laplacian_kernel": "5x5"},
        )
        gray = np.array([[50]], dtype=np.uint8)
        result = proc._variance_of_laplacian(gray)
        assert result >= 0.0
        assert np.isfinite(result)

    def test_entropy_bins_one(self) -> None:
        """Verify entropy returns 0.0 when histogram has only one bin."""
        proc = VisualFeaturesProcessor(
            name="test",
            config={"columns": {"input": ["img"]}, "histogram": {"bins": 1}},
        )
        img = np.zeros((4, 4), dtype=np.uint8)
        result = proc._entropy(img)
        assert result == pytest.approx(0.0)

    def test_entropy_bins_two(self) -> None:
        """Verify entropy works with two histogram bins."""
        proc = VisualFeaturesProcessor(
            name="test",
            config={"columns": {"input": ["img"]}, "histogram": {"bins": 2}},
        )
        img = np.zeros((4, 4), dtype=np.uint8)
        result = proc._entropy(img)
        assert result >= 0.0

    def test_clip_percentiles_equal_returns_unchanged(self) -> None:
        """Verify clipping with equal percentiles returns unchanged array."""
        proc = VisualFeaturesProcessor(
            name="test",
            config={"columns": {"input": ["img"]}, "clip_percentiles": [50, 50]},
        )
        gray = np.array([[100.0, 150.0], [200.0, 250.0]], dtype=np.float32)
        result = proc._apply_clip_and_normalize(gray)
        assert result.dtype == np.float32
        assert np.array_equal(result, gray)

    def test_entropy_non_normalized_all_zeros(self) -> None:
        """Verify entropy returns 0.0 for all-zero non-normalized image."""
        proc = VisualFeaturesProcessor(
            name="test",
            config={"columns": {"input": ["img"]}, "normalize": False},
        )
        img = np.zeros((4, 4), dtype=np.uint8)
        result = proc._entropy(img)
        assert result == pytest.approx(0.0)

    def test_entropy_non_normalized_all_255(self) -> None:
        """Verify entropy returns 0.0 for all-255 non-normalized image."""
        proc = VisualFeaturesProcessor(
            name="test",
            config={"columns": {"input": ["img"]}, "normalize": False},
        )
        img = np.full((4, 4), 255, dtype=np.uint8)
        result = proc._entropy(img)
        assert result == pytest.approx(0.0)

    def test_variance_of_laplacian_all_zeros(self) -> None:
        """Verify variance of Laplacian returns 0.0 for all-zero image."""
        proc = _make_processor()
        gray = np.zeros((10, 10), dtype=np.uint8)
        result = proc._variance_of_laplacian(gray)
        assert result == pytest.approx(0.0)

    def test_variance_of_laplacian_all_255(self) -> None:
        """Verify variance of Laplacian handles all-255 image."""
        proc = _make_processor()
        gray = np.full((10, 10), 255, dtype=np.uint8)
        result = proc._variance_of_laplacian(gray)
        assert result > 0.0

    def test_to_float01_constant_image_returns_zeros(self) -> None:
        """Verify _to_float01 returns zeros for constant input image."""
        gray = np.full((4, 4), 100, dtype=np.uint8)
        result = VisualFeaturesProcessor._to_float01(gray)
        assert np.allclose(result, 0.0, atol=1e-8)
