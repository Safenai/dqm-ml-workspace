"""Visual feature extraction processor for image quality assessment.

This module contains the VisualFeaturesProcessor class that extracts
visual quality features from images including luminosity, contrast,
blur, and entropy.
"""

import io
import logging
from pathlib import Path
from typing import Any

from dqm_ml_core import FeaturesProcessor
from dqm_ml_core.models.columns import ColumnsConfig
from dqm_ml_core.models.processors import _LUMINOSITY_STANDARDS, ImageFeaturesProcessorConfig
from dqm_ml_core.utils.image_loading import ImageLoadingMixin
import numpy as np
from PIL import Image
import pyarrow as pa
from scipy import signal

# COMPATIBILITY : from typing import Any, override # When support of 3.10 and 3.11 will be removed
from typing_extensions import override

logger = logging.getLogger(__name__)


class VisualFeaturesProcessor(ImageLoadingMixin, FeaturesProcessor):
    """
    Computes basic image quality features per sample.

    Features:
      - Luminosity: Mean intensity of the image. By default, it is the
        average gray level mapped to the [0, 1] range.
      - Contrast: RMS contrast, calculated as the standard deviation of
        the gray level intensities, mapped to the [0, 1] range.
      - Blur: Measured as the variance of the Laplacian of the image. A
        higher value indicates more edges and higher sharpness.
      - Entropy: Shannon entropy of the image's grayscale histogram.
        Measures the information content or complexity.

    This processor operates purely at the feature extraction level
    (per-sample).
    """

    DEFAULT_OUTPUTS: dict[str, str] = {
        "luminosity": "luminosity",
        "contrast": "contrast",
        "blur": "blur",
        "entropy": "entropy",
    }

    output_features: dict[str, str]

    def __init__(self, name: str = "image_features", config: dict[str, Any] | None = None) -> None:
        """
        Initialize the visual features processor.

        Args:
            name: Unique name of the processor instance.
            config: Configuration dictionary containing:
                - input_columns: List containing the image column name.
                - output_features: Mapping of feature names to column names.
                - grayscale: Whether to convert images to grayscale.
                - normalize: Whether to normalize pixel values to [0, 1].
                - entropy_bins: Number of bins for entropy calculation.
                - clip_percentiles: Tuple of (low, high) percentiles.
                - laplacian_kernel: Laplacian kernel size ('3x3' or '5x5').
                - path_prefix: Base directory for resolving relative paths
                  (set via dataloader ``mode`` config instead).
        """
        super().__init__(name, config)

        self.columns_config: ColumnsConfig | None = None
        raw_columns = self.config.get("columns")
        if isinstance(raw_columns, dict):
            self.columns_config = ColumnsConfig.model_validate(raw_columns)

        cfg = ImageFeaturesProcessorConfig.model_validate({**self.config, "name": self.name})

        self._configure_storage(self.config)
        self._configure_columns(cfg)
        self._configure_params(cfg)
        self._validate_output_features()

    def _configure_storage(self, cfg: dict[str, Any]) -> None:
        """Configure S3 filesystem support from config.

        Args:
            cfg: Configuration dictionary.
        """
        self.s3_fs = None
        storage_cfg = self.storage_raw
        if not storage_cfg:
            return

        from dqm_ml_core.models.global_ import StorageConfig
        from dqm_ml_job.utils import get_s3_filesystem

        storage_config = StorageConfig.model_validate(storage_cfg)
        if storage_config.type == "s3":
            self.s3_fs = get_s3_filesystem(storage_config)

    def _configure_columns(self, cfg: ImageFeaturesProcessorConfig) -> None:
        """Configure input and output columns.

        Args:
            cfg: Configuration dictionary.
        """
        if not hasattr(self, "input_columns") or "input" not in (self.config.get("columns") or {}):
            self.input_columns = ["image_bytes"]

        if not hasattr(self, "output_features") or not self.output_features:
            self.output_features = self.DEFAULT_OUTPUTS.copy()

    def _configure_params(self, cfg: ImageFeaturesProcessorConfig) -> None:
        """Configure processing parameters.

        Args:
            cfg: Configuration dictionary.
        """
        self.grayscale: bool = cfg.grayscale
        self.normalize: bool = cfg.normalize
        self.entropy_bins: int = cfg.histogram.bins if cfg.histogram else 256
        self.clip_percentiles = cfg.clip_percentiles
        self.laplacian_kernel: str = cfg.laplacian_kernel
        raw = cfg.luminosity_weights
        if isinstance(raw, str):
            self._luminosity_weights = _LUMINOSITY_STANDARDS[raw]
        else:
            self._luminosity_weights = raw if raw is not None else _LUMINOSITY_STANDARDS["bt709"]

    def _validate_output_features(self) -> None:
        """Validate and populate default output features configuration.

        Ensures output_features is a dict and fills in missing feature keys
        with defaults.

        Raises:
            ValueError: If output_features is not a dictionary.
        """
        if not isinstance(self.output_features, dict):
            raise ValueError(f"[{self.name}] 'output_features' must be a dict of metric->column_name")
        for k in ("luminosity", "contrast", "blur", "entropy"):
            if k not in self.output_features:
                self.output_features[k] = self.DEFAULT_OUTPUTS[k]

    def _apply_clip_and_normalize(self, gray: np.ndarray) -> np.ndarray:
        """Apply percentile clipping and optional normalization to a grayscale array.

        Args:
            gray: Input grayscale array.

        Returns:
            Clipped and optionally normalized array.
        """
        if self.clip_percentiles is None:
            return gray
        p_lo, p_hi = self.clip_percentiles
        lo = np.percentile(gray, p_lo)
        hi = np.percentile(gray, p_hi)
        if hi <= lo:
            return gray
        gray = np.clip(gray, lo, hi)
        if self.normalize:
            gray = (gray - lo) / max(1e-12, (hi - lo))
        return gray

    def _handle_image_error(self, exc: Exception, idx: int) -> None:
        """Check error config and either raise or record the failure.

        Increments failure counters and checks failure rate threshold.

        rate against
        configured maximum.

        Args:
            exc: The exception that occurred.
            idx: Sample index for logging.
        """
        self._check_image_fail_fast(exc, "on_transform_error", "on_unsupported_format")
        self._failure_count += 1
        self._total_count += 1
        self._check_failure_rate()
        logger.exception(f"[{self.name}] failed to process sample {idx}: {exc}")

    def _process_single_image(self, idx: int, v: Any, column: str | None = None) -> np.ndarray | None:
        """Convert a raw image value to a grayscale numpy array, applying clipping.

        Args:
            idx: Position index for error logging.
            v: Raw image value (bytes, path, PIL Image, or ndarray).
            column: The input column name (used to resolve path prefix).

        Returns:
            Grayscale numpy array or None if processing fails.
        """
        try:
            gray = self._to_gray_np(v, column=column)
            return self._apply_clip_and_normalize(gray)
        except Exception as e:
            self._handle_image_error(e, idx)
            return None

    @staticmethod
    def _compute_scalar_feature(
        gray_images: list[np.ndarray | None],
        func: Any,
        normalize: bool,
    ) -> pa.Array:
        """Compute a per-image scalar feature using a given function.

        Args:
            gray_images: List of grayscale image arrays (or None for failed images).
            func: Callable that takes a grayscale array and returns a scalar.
            normalize: Whether pixel values are normalized to [0, 1].

        Returns:
            PyArrow array of feature values.
        """
        values = []
        for gray in gray_images:
            if gray is not None:
                values.append(float(func(gray if normalize else gray / 255.0)))
            else:
                values.append(float("nan"))
        return pa.array(values, type=pa.float32())

    def _output_column_name(self, col: str, feature_key: str) -> str:
        """Generate output column name for a feature with rename/prefix/suffix.

        Args:
            col: Input column name.
            feature_key: Feature key (luminosity, contrast, blur, entropy).

        Returns:
            Fully qualified output column name with rename, prefix, and suffix applied.
        """
        name = self.output_features.get(feature_key, feature_key)
        if self.columns_config and self.columns_config.rename:
            for r in self.columns_config.rename:
                if r.from_ == feature_key:
                    name = r.to
                    break
        return super()._resolve_output_name(col, name)

    @override
    def generated_features(self) -> list[str]:
        """Return list of feature column names this processor will generate.

        Combines each input column with each feature key (luminosity, contrast,
        blur, entropy) applying column modifiers (prefix, suffix, rename).

        Returns:
            List of output feature column names.
        """
        if not self.input_columns:
            return []
        return [
            self._output_column_name(col, fk)
            for col in self.input_columns
            for fk in ("luminosity", "contrast", "blur", "entropy")
        ]

    @override
    def compute_features(
        self,
        batch: pa.RecordBatch,
        prev_features: dict[str, pa.Array] | None = None,
    ) -> dict[str, pa.Array]:
        """Compute per-sample image features for all configured input columns.

        Args:
            batch: Input batch of data containing image columns.
            prev_features: Previously computed features (not used in this processor).

        Returns:
            Dictionary mapping feature names to their computed values.
        """
        if not self.input_columns:
            logger.warning(f"[{self.name}] no input_columns configured")
            return {}

        result: dict[str, pa.Array] = {}
        for image_column in self.input_columns:
            if image_column not in batch.schema.names:
                logger.warning(f"[{self.name}] column '{image_column}' not found in batch")
                continue

            values = batch.column(image_column).to_pylist()
            gray_images = [self._process_single_image(idx, v, column=image_column) for idx, v in enumerate(values)]

            for fk in ("luminosity", "contrast", "blur", "entropy"):
                func = {"luminosity": np.mean, "contrast": np.std}.get(fk)
                if fk == "blur":
                    arr = self._compute_scalar_feature(gray_images, self._variance_of_laplacian, True)
                elif fk == "entropy":
                    arr = self._compute_scalar_feature(gray_images, self._entropy, True)
                else:
                    arr = self._compute_scalar_feature(gray_images, func, self.normalize)
                result[self._output_column_name(image_column, fk)] = arr

        return result

    @override
    def reset(self) -> None:
        """Reset processor state for new processing run.

        Resets failure counters and total count for fresh processing.
        """
        self._failure_count = 0
        self._total_count = 0

    # --- helpers --------------------------------------------------------------

    def _to_gray_np(self, image_data: Any, column: str | None = None) -> np.ndarray:
        """Convert various input types to a 2D grayscale numpy array.

        If `self.normalize` is True, returns float32 in [0,1].
        Otherwise returns uint8 [0,255].

        Args:
            image_data: Input data (PIL Image, bytes, string path, or numpy array).
            column: The input column name (used to resolve path prefix).

        Returns:
            2D grayscale array in grayscale.
        """
        if isinstance(image_data, Image.Image):
            return self._pil_to_gray(image_data)
        if isinstance(image_data, (bytes, bytearray)):
            return self._pil_to_gray(Image.open(io.BytesIO(image_data)))
        if isinstance(image_data, str):
            image = self._load_image_from_path(image_data, column=column)
            if image is None:
                raise ValueError(f"Failed to load image from path: {image_data}")
            return self._pil_to_gray(image)
        if isinstance(image_data, np.ndarray):
            return self._ndarray_to_gray(image_data)
        raise ValueError(f"Unsupported type for image input: {type(image_data)}")

    def _load_image_from_path(self, path: str, column: str | None = None) -> Image.Image | None:
        """Load a PIL Image from a string path (S3 or local).

        Resolves relative paths by looking up the prefix for the configured
        input column from ``self.current_path_prefix`` (set by the job for
        each selection).

        Args:
            path: File path or relative path.
            column: The input column name (used to resolve the path prefix).

        Returns:
            PIL Image object.

        Raises:
            ValueError: If the file does not exist and ``fail_fast`` is configured.
        """
        img = super()._open_image_from_path(path, column)
        if img is not None:
            return img

        # File not found — check error configuration
        if (
            self.errors_config
            and self.errors_config.tabular
            and self.errors_config.tabular.on_file_not_found == "fail_fast"
        ):
            prefix = self._current_image_prefix(column)
            img_path = Path(prefix) / path if prefix else Path(path)
            raise ValueError(f"Path does not exist: {img_path}")
        return None

    def _pil_to_gray(self, img: Image.Image) -> np.ndarray:
        """Convert a PIL Image to a 2D grayscale numpy array.

        Args:
            img: PIL Image to convert.

        Returns:
            2D grayscale array.
        """
        if self.grayscale and img.mode != "L":
            img = img.convert("L")
        elif not self.grayscale and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        gray_np = np.array(img)
        if gray_np.ndim == 3:
            r, g, b = self._luminosity_weights
            gray_np = r * gray_np[..., 0] + g * gray_np[..., 1] + b * gray_np[..., 2]

        return self._to_float01(gray_np) if self.normalize else gray_np.astype(np.uint8)

    def _ndarray_to_gray(self, arr: np.ndarray) -> np.ndarray:
        """Convert a numpy image array to 2D grayscale.

        Args:
            arr: Input array (2D gray, 3D RGB/RGBA).

        Returns:
            2D grayscale array.

        Raises:
            ValueError: If the shape is unsupported.
        """
        if arr.ndim == 2:
            gray = arr
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            r, g, b = self._luminosity_weights
            gray = r * rgb[..., 0] + g * rgb[..., 1] + b * rgb[..., 2]
        else:
            raise ValueError(f"Unsupported ndarray shape {arr.shape}")

        return self._to_float01(gray) if self.normalize else gray.astype(np.uint8)

    @staticmethod
    def _to_float01(arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range using min-max scaling.

        Args:
            arr: Input numpy array.

        Returns:
            Normalized array with float32 values in [0, 1].
        """
        arr = arr.astype(np.float32)
        vmin, vmax = float(arr.min()), float(arr.max())
        arr = (arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(arr, dtype=np.float32)
        return arr

    def _variance_of_laplacian(self, gray: np.ndarray) -> float:
        """Variance of Laplacian as a blur metric.

        Args:
            gray: Grayscale image array.

        Returns:
            Variance of Laplacian (higher values indicate
            more edges/sharpness).
        """
        gray = gray.astype(np.float32)
        if self.laplacian_kernel == "5x5":
            kernel = np.array(
                [
                    [0, 0, -1, 0, 0],
                    [0, -1, -2, -1, 0],
                    [-1, -2, 16, -2, -1],
                    [0, -1, -2, -1, 0],
                    [0, 0, -1, 0, 0],
                ],
                dtype=np.float32,
            )
        else:
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        # Use scipy for optimized convolution
        lap = signal.convolve2d(gray, kernel, mode="same")
        return float(np.var(lap))

    def _entropy(self, gray: np.ndarray) -> float:
        """Shannon entropy of the gray histogram (natural log).

        Args:
            gray: Grayscale image array.

        Returns:
            Shannon entropy value. Returns NaN if histogram sum is zero.
        """
        if self.normalize:
            # histogram on [0,1]
            hist, _ = np.histogram(gray, bins=self.entropy_bins, range=(0.0, 1.0))
        else:
            # uint8 range
            hist, _ = np.histogram(gray, bins=min(256, self.entropy_bins), range=(0, 255))
        prob = hist.astype(np.float64)
        total = prob.sum()
        if total <= 0:
            return float("nan")
        prob /= total
        # avoid log(0)
        prob = prob[prob > 0]
        return float(-(prob * np.log(prob)).sum())
