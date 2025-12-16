import io
import logging
from pathlib import Path
from typing import Any, override

import numpy as np
from PIL import Image
import pyarrow as pa

from dqm_ml_core import DatametricProcessor

logger = logging.getLogger(__name__)


class VisualFeaturesProcessor(DatametricProcessor):
    """
    Compute basic image quality features per sample :
      - Luminosity (mean gray level in [0, 1])
      - Contrast (RMS contrast = std of gray in [0, 1])
      - Blur (variance of Laplacian on gray)
      - Entropy (Shannon entropy of gray histogram, base e)

    """

    DEFAULT_OUTPUTS = {
        "luminosity": "m_luminosity",
        "contrast": "m_contrast",
        "blur": "m_blur_level",
        "entropy": "m_entropy",
    }

    def __init__(self, name: str = "visual_metrique", config: dict[str, Any] | None = None) -> None:
        super().__init__(name, config)

        # Local view of config for convenience
        cfg = self.config or {}

        # handle relative paths in parquet to a dataset located at dataset_root_path
        self.dataset_root_path = str(cfg.get("dataset_root_path", "undefined"))

        if not hasattr(self, "input_columns") or not self.input_columns:
            self.input_columns = ["image_bytes"]

        if not hasattr(self, "output_features") or not self.output_features:
            # Use config-provided mapping if present, otherwise defaults
            cfg_outputs = cfg.get("output_features") if isinstance(cfg.get("output_features"), dict) else None
            self.output_features: Any = (
                cfg_outputs.copy() if isinstance(cfg_outputs, dict) else self.DEFAULT_OUTPUTS.copy()
            )

        # param
        # TODO : see with Loic how to define parameters in the pipeline.yaml
        self.grayscale: bool = bool(cfg.get("grayscale", True))
        self.normalize: bool = bool(cfg.get("normalize", True))
        self.entropy_bins: int = int(cfg.get("entropy_bins", 256))

        # TODO written to remove noqa 501 and type check error, to be fixed properly later
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
        """Compute per-sample image features."""
        if not self.input_columns:
            logger.warning(f"[{self.name}] no input_columns configured")
            return {}

        image_column = self.input_columns[0]
        if image_column not in batch.schema.names:
            logger.warning(f"[{self.name}] column '{image_column}' not found in batch")
            return {}

        col = batch.column(image_column)
        values = col.to_pylist()  #
        # Utilise rqque les l'image grise
        gray_images: list[Any] = []
        for idx, v in enumerate(values):
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
                gray_images.append(gray)
            except Exception as e:
                logger.exception(f"[{self.name}] failed to process sample {idx}: {e}")
                gray_images.append(None)

        # Compute each feature type with dedicated functions
        features = {}
        features[self.output_features["luminosity"]] = self._compute_luminosity_feature(gray_images)
        features[self.output_features["contrast"]] = self._compute_contrast_feature(gray_images)
        features[self.output_features["blur"]] = self._compute_blur_feature(gray_images)
        features[self.output_features["entropy"]] = self._compute_entropy_feature(gray_images)
        return features

    @override
    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """No-op aggregation: metrics are image-level only.."""
        return {}

    @override
    def compute(self, batch_metrics: dict[str, pa.Array] | None = None) -> dict[str, pa.Array]:
        """No dataset-level aggregation required for this processor."""
        return {}

    def reset(self) -> None:
        pass

    # les différentes fonctions de calcul des features
    # TODO :     voir si on peut les vectoriser

    def _compute_luminosity_feature(self, gray_images: list[np.ndarray | None]) -> pa.Array:
        """Compute luminosity (mean gray level) for each image."""
        values = []
        for gray in gray_images:
            if gray is not None:
                luminosity = float(np.mean(gray if self.normalize else gray / 255.0))
                values.append(luminosity)
            else:
                values.append(float("nan"))
        return pa.array(values, type=pa.float32())

    def _compute_contrast_feature(self, gray_images: list[np.ndarray | None]) -> pa.Array:
        """Compute contrast (RMS contrast = std of gray) for each image."""
        values = []
        for gray in gray_images:
            if gray is not None:
                contrast = float(np.std(gray if self.normalize else gray / 255.0))
                values.append(contrast)
            else:
                values.append(float("nan"))
        return pa.array(values, type=pa.float32())

    def _compute_blur_feature(self, gray_images: list[np.ndarray | None]) -> pa.Array:
        """Compute blur (variance of Laplacian) for each image."""
        values = []
        for gray in gray_images:
            if gray is not None:
                blur_val = float(self._variance_of_laplacian(gray))
                values.append(blur_val)
            else:
                values.append(float("nan"))
        return pa.array(values, type=pa.float32())

    def _compute_entropy_feature(self, gray_images: list[np.ndarray | None]) -> pa.Array:
        """Compute entropy (Shannon entropy) for each image."""
        values = []
        for gray in gray_images:
            if gray is not None:
                entropy_val = float(self._entropy(gray))
                values.append(entropy_val)
            else:
                values.append(float("nan"))
        return pa.array(values, type=pa.float32())

    # --- helpers --------------------------------------------------------------

    def _to_gray_np(self, x: Any) -> np.ndarray:
        """Convert various input types to a 2D grayscale numpy array.

        If `self.normalize` is True, returns float32 in [0,1]. Otherwise returns uint8 [0,255].
        """
        img: Image.Image | None = None

        if isinstance(x, Image.Image):
            img = x
        elif isinstance(x, (bytes, bytearray)):
            img = Image.open(io.BytesIO(x))
        elif isinstance(x, str):
            img_path = Path(self.dataset_root_path) / x if self.dataset_root_path != "undefined" else Path(x)
            if img_path.is_file():
                img = Image.open(img_path)
            else:
                raise ValueError(f"Path does not exist: {img_path}")
        elif isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 2:  # already gray
                gray = arr
            elif arr.ndim == 3 and arr.shape[2] in (3, 4):
                # manual luminance conversion to be independent of PIL for ndarray
                rgb = arr[..., :3].astype(np.float32)
                gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            else:
                raise ValueError(f"Unsupported ndarray shape {arr.shape}")

            gray = self._to_float01(gray) if self.normalize else gray.astype(np.uint8)
            return gray
        else:
            raise ValueError(f"Unsupported type for image input: {type(x)}")

        # Use PIL pipeline
        if self.grayscale and img.mode != "L":
            img = img.convert("L")
        elif not self.grayscale and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        gray_np = np.array(img)
        if gray_np.ndim == 3:  # RGB -> gray
            gray_np = 0.2126 * gray_np[..., 0] + 0.7152 * gray_np[..., 1] + 0.0722 * gray_np[..., 2]

        if self.normalize:
            return self._to_float01(gray_np)
        else:
            return gray_np.astype(np.uint8)

    @staticmethod
    def _to_float01(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        vmin, vmax = float(arr.min()), float(arr.max())
        arr = (arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(arr, dtype=np.float32)
        return arr

    def _variance_of_laplacian(self, gray: np.ndarray) -> float:
        """Variance of Laplacian as a blur metric.

        Works with gray in [0,1] or [0,255]; scaling does not change the *relative* ranking,
        but absolute values differ. We always use the working array as float32.
        """
        g = gray.astype(np.float32)
        if self.laplacian_kernel == "5x5":
            # 5x5 Laplacian (approx.)
            k = np.array(
                [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]],
                dtype=np.float32,
            )
        else:
            # classic 3x3
            k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = self._conv2d_same(g, k)
        return float(np.var(lap))

    def _entropy(self, gray: np.ndarray) -> float:
        """Shannon entropy of the gray histogram (natural log)."""
        g = gray
        if self.normalize:
            # histogram on [0,1]
            hist, _ = np.histogram(g, bins=self.entropy_bins, range=(0.0, 1.0))
        else:
            # uint8 range
            hist, _ = np.histogram(g, bins=min(256, self.entropy_bins), range=(0, 255))
        p = hist.astype(np.float64)
        s = p.sum()
        if s <= 0:
            return float("nan")
        p /= s
        # avoid log(0)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    @staticmethod
    def _conv2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution with 'same' output shape and zero padding."""
        ih, iw = img.shape[:2]
        kh, kw = kernel.shape
        pad_h = kh // 2
        pad_w = kw // 2
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        out = np.zeros_like(img, dtype=np.float32)
        kf = np.flipud(np.fliplr(kernel)).astype(np.float32)
        for i in range(ih):
            for j in range(iw):
                region = padded[i : i + kh, j : j + kw]
                out[i, j] = float(np.sum(region * kf))
        return out
