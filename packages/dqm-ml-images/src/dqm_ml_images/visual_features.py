"""Visual feature extraction processor for image quality assessment.

This module contains the VisualFeaturesProcessor class that extracts
visual quality features from images including luminosity, contrast,
blur, and entropy.
"""

import io
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pyarrow as pa
from scipy import signal

# COMPATIBILITY : from typing import Any, override # When support of 3.10 and 3.11 will be removed
from typing_extensions import override

from dqm_ml_core import DatametricProcessor

logger = logging.getLogger(__name__)


class VisualFeaturesProcessor(DatametricProcessor):
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

    DEFAULT_OUTPUTS = {
        "luminosity": "m_luminosity",
        "contrast": "m_contrast",
        "blur": "m_blur_level",
        "entropy": "m_entropy",
    }

    def __init__(self, name: str = "visual_metric", config: dict[str, Any] | None = None) -> None:
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
                - dataset_root_path: Root directory for relative paths.
        """
        super().__init__(name, config)

        # Local view of config for convenience
        cfg = self.config or {}

        # handle relative paths in parquet to a dataset located at dataset_root_path
        self.dataset_root_path = str(cfg.get("dataset_root_path", "undefined"))

        # S3 filesystem support
        self.s3_fs = None
        s3_config = cfg.get("s3_filesystem")
        if s3_config:
            from dqm_ml_job.utils import get_s3_filesystem

            if s3_config is True:
                self.s3_fs = get_s3_filesystem()
            elif isinstance(s3_config, dict):
                self.s3_fs = get_s3_filesystem(
                    access_key=s3_config.get("access_key"),
                    secret_key=s3_config.get("secret_key"),
                    endpoint=s3_config.get("endpoint_override"),
                    region=s3_config.get("region"),
                )

        if not hasattr(self, "input_columns") or not self.input_columns:
            self.input_columns = ["image_bytes"]

        if not hasattr(self, "output_features") or not self.output_features:
            # Use config-provided mapping if present, otherwise defaults
            cfg_outputs = cfg.get("output_features") if isinstance(cfg.get("output_features"), dict) else None
            self.output_features: Any = (
                cfg_outputs.copy() if isinstance(cfg_outputs, dict) else self.DEFAULT_OUTPUTS.copy()
            )

        # param
        self.grayscale: bool = bool(cfg.get("grayscale", True))
        self.normalize: bool = bool(cfg.get("normalize", True))
        self.entropy_bins: int = int(cfg.get("entropy_bins", 256))

        # TODO written to remove noqa 501 and type check error in same line, to be fixed properly later
        if cfg.get("clip_percentiles") is not None:
            self.clip_percentiles = tuple(cfg.get("clip_percentiles"))  # type: ignore
        else:
            self.clip_percentiles = None  # type: ignore

        self.laplacian_kernel: str = str(cfg.get("laplacian_kernel", "3x3"))

        # check if the transformation is defined in the processor
        if not isinstance(self.output_features, dict):
            raise ValueError(f"[{self.name}] 'output_features' must be a dict of metric->column_name")
        for k in ("luminosity", "contrast", "blur", "entropy"):
            if k not in self.output_features:
                self.output_features[k] = self.DEFAULT_OUTPUTS[k]

    @override
    def compute_features(
        self,
        batch: pa.RecordBatch,
        prev_features: dict[str, pa.Array] | None = None,
    ) -> dict[str, pa.Array]:
        """Compute per-sample image features.

        Args:
            batch: Input batch of data containing image column.
            prev_features: Previously computed features (not used in this processor).

        Returns:
            Dictionary mapping feature names to their computed values.
        """
        if not self.input_columns:
            logger.warning(f"[{self.name}] no input_columns configured")
            return {}

        image_column = self.input_columns[0]
        if image_column not in batch.schema.names:
            logger.warning(f"[{self.name}] column '{image_column}' not found in batch")
            return {}

        col = batch.column(image_column)
        values = col.to_pylist()
        gray_images = [self._process_single_image(v, idx) for idx, v in enumerate(values)]

        # Compute each feature type with dedicated functions
        features = {}
        features[self.output_features["luminosity"]] = self._compute_luminosity_feature(gray_images)
        features[self.output_features["contrast"]] = self._compute_contrast_feature(gray_images)
        features[self.output_features["blur"]] = self._compute_blur_feature(gray_images)
        features[self.output_features["entropy"]] = self._compute_entropy_feature(gray_images)
        return features

    @override
    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """No-op aggregation: metrics are image-level only.

        Returns:
            Empty dictionary as this processor computes features only.
        """
        return {}

    @override
    def compute(self, batch_metrics: dict[str, pa.Array] | None = None) -> dict[str, pa.Array]:
        """No dataset-level aggregation required for this processor.

        Returns:
            Empty dictionary as features are computed at batch level.
        """
        return {}

    def reset(self) -> None:
        """Reset processor state for new processing run."""

    # TODO : Check if it can be vectorized, parallelized

    def _compute_luminosity_feature(self, gray_images: list[np.ndarray | None]) -> pa.Array:
        """Compute luminosity (mean gray level) for each image.

        Args:
            gray_images: List of grayscale image arrays (or None for failed images).

        Returns:
            PyArrow array of luminosity values.
        """
        values = []
        for gray in gray_images:
            if gray is not None:
                # Original logic: if not self.normalize, it's uint8 [0,255], divide by 255
                # If self.normalize, it's already [0,1] (min-max)
                luminosity = float(np.mean(gray if self.normalize else gray / 255.0))
                values.append(luminosity)
            else:
                values.append(float("nan"))
        return pa.array(values, type=pa.float32())

    def _compute_contrast_feature(self, gray_images: list[np.ndarray | None]) -> pa.Array:
        """Compute contrast (RMS contrast = std of gray) for each image.

        Args:
            gray_images: List of grayscale image arrays (or None for failed images).

        Returns:
            PyArrow array of contrast values.
        """
        values = []
        for gray in gray_images:
            if gray is not None:
                contrast = float(np.std(gray if self.normalize else gray / 255.0))
                values.append(contrast)
            else:
                values.append(float("nan"))
        return pa.array(values, type=pa.float32())

    def _compute_blur_feature(self, gray_images: list[np.ndarray | None]) -> pa.Array:
        """Compute blur (variance of Laplacian) for each image.

        Args:
            gray_images: List of grayscale image arrays (or None for failed images).

        Returns:
            PyArrow array of blur values.
        """
        values = []
        for gray in gray_images:
            if gray is not None:
                blur_val = float(self._variance_of_laplacian(gray))
                values.append(blur_val)
            else:
                values.append(float("nan"))
        return pa.array(values, type=pa.float32())

    def _compute_entropy_feature(self, gray_images: list[np.ndarray | None]) -> pa.Array:
        """Compute entropy (Shannon entropy) for each image.

        Args:
            gray_images: List of grayscale image arrays (or None for failed images).

        Returns:
            PyArrow array of entropy values.
        """
        values = []
        for gray in gray_images:
            if gray is not None:
                entropy_val = float(self._entropy(gray))
                values.append(entropy_val)
            else:
                values.append(float("nan"))
        return pa.array(values, type=pa.float32())

    # --- helpers --------------------------------------------------------------

    def _is_s3_path(self, path: str) -> bool:
        """Check if a path is an S3 path.

        Args:
            path: The path to check.

        Returns:
            True if the path is an S3 path, False otherwise.
        """
        return path.startswith("s3://") or ("/" in path and not Path(path).is_absolute())

    def _get_s3_path(self, file_path: str) -> str:
        """Construct an S3 path by combining the bucket name with a file path.

        Args:
            file_path: The file path within the bucket.

        Returns:
            str: The full S3 path in format "bucket_name/file_path".
        """
        bucket_name = os.getenv("S3_BUCKET_NAME", "")
        return bucket_name + "/" + file_path

    def _load_image_from_bytes(self, data: bytes | bytearray) -> Image.Image:
        """Load a PIL Image from bytes or bytearray.

        Args:
            data: Raw image bytes.

        Returns:
            PIL Image object.
        """
        return Image.open(io.BytesIO(data))

    def _load_image_from_path(self, image_data: str) -> Image.Image:
        """Load a PIL Image from a string path (S3 or local).

        Args:
            image_data: File path or relative path.

        Returns:
            PIL Image object.

        Raises:
            ValueError: If the path does not exist.
        """
        if self.s3_fs:
            s3_key = f"{self.dataset_root_path}/{image_data}" if self.dataset_root_path != "undefined" else image_data
            bucket_name = os.getenv("S3_BUCKET_NAME", "")
            s3_path = f"{bucket_name}/{s3_key}"
            with self.s3_fs.open_input_stream(s3_path) as f:
                loaded = Image.open(io.BytesIO(f.read()))
                img = loaded.copy()
            return img

        img_path = (
            Path(self.dataset_root_path) / image_data if self.dataset_root_path != "undefined" else Path(image_data)
        )
        if not img_path.is_file():
            raise ValueError(f"Path does not exist: {img_path}")
        return Image.open(img_path)

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
            gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        else:
            raise ValueError(f"Unsupported ndarray shape {arr.shape}")

        return self._to_float01(gray) if self.normalize else gray.astype(np.uint8)

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
            gray_np = 0.2126 * gray_np[..., 0] + 0.7152 * gray_np[..., 1] + 0.0722 * gray_np[..., 2]

        return self._to_float01(gray_np) if self.normalize else gray_np.astype(np.uint8)

    def _process_single_image(self, v: Any, idx: int) -> np.ndarray | None:
        """Convert a single raw image value to a processed grayscale array.

        Applies optional clip-percentile and normalization.

        Args:
            v: Raw image value from the batch column.
            idx: Index for error logging.

        Returns:
            Processed grayscale array, or None on failure.
        """
        try:
            gray = self._to_gray_np(v)
            if self.clip_percentiles is not None:
                p_lo, p_hi = self.clip_percentiles
                lo = np.percentile(gray, p_lo)
                hi = np.percentile(gray, p_hi)
                if hi > lo:
                    gray = np.clip(gray, lo, hi)
                    if self.normalize:
                        gray = (gray - lo) / max(1e-12, (hi - lo))
            return gray
        except Exception as e:
            logger.exception(f"[{self.name}] failed to process sample {idx}: {e}")
            return None

    def _to_gray_np(self, image_data: Any) -> np.ndarray:
        """Convert various input types to a 2D grayscale numpy array.

        If `self.normalize` is True, returns float32 in [0,1].
        Otherwise returns uint8 [0,255].

        Args:
            image_data: Input data (PIL Image, bytes, string path, or numpy array).

        Returns:
            2D numpy array in grayscale.
        """
        if isinstance(image_data, Image.Image):
            return self._pil_to_gray(image_data)
        if isinstance(image_data, (bytes, bytearray)):
            return self._pil_to_gray(self._load_image_from_bytes(image_data))
        if isinstance(image_data, str):
            return self._pil_to_gray(self._load_image_from_path(image_data))
        if isinstance(image_data, np.ndarray):
            return self._ndarray_to_gray(image_data)
        raise ValueError(f"Unsupported type for image input: {type(image_data)}")

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
