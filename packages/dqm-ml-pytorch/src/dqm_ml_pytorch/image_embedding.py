"""Image embedding processor using pre-trained deep learning models.

This module contains the ImageEmbeddingProcessor class that extracts
high-dimensional embeddings from images using PyTorch and torchvision
pre-trained models.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from PIL import Image
import pyarrow as pa
import torch
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

# COMPATIBILITY : from typing import Any, override # When support of 3.10 and 3.11 will be removed
from typing_extensions import override

from dqm_ml_core import DatametricProcessor
from dqm_ml_core.models.columns import ColumnsConfig
from dqm_ml_core.models.processors import FeaturesEmbeddingsProcessorConfig
from dqm_ml_core.utils.matching import resolve_include_exclude

logger = logging.getLogger(__name__)


class ImageEmbeddingProcessor(DatametricProcessor):
    """
    Computes high-dimensional latent vectors (embeddings) for images
    using deep learning models.

    This processor uses PyTorch and Torchvision to:
    1. Load images from bytes or file paths.
    2. Preprocess images (resize, normalize) for the selected model.
    3. Run batch inference using a pre-trained model (e.g., ResNet, ViT).
    4. Extract features from a specific layer (e.g., 'avgpool').

    The resulting embeddings are stored as a `FixedSizeListArray`
    in the features.
    """

    def __init__(
        self,
        name: str = "image_embedding",
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the image embedding processor.

        Args:
            name: Unique name of the processor instance.
            config: Configuration dictionary containing:
                - infer:
                    - width, height: Input resolution for the model (default: 224x224).
                    - batch_size: Number of images per inference pass (default: 32).
                    - norm_mean, norm_std: Preprocessing normalization stats.
                - model:
                    - arch: Torchvision model name (default: "resnet18").
                    - n_layer_feature: Target layer for feature extraction (default: "avgpool").
                    - device: Execution device, "cpu" or "cuda" (default: "cpu").
        """
        super().__init__(name, config)

        self.columns_config: ColumnsConfig | None = None
        raw_columns = self.config.get("columns")
        if isinstance(raw_columns, dict):
            self.columns_config = ColumnsConfig.model_validate(raw_columns)

        cfg = FeaturesEmbeddingsProcessorConfig.model_validate({**self.config, "name": self.name})

        # Storage filesystem support
        self.s3_fs = None
        storage_cfg = self.config.get("storage")
        if storage_cfg:
            from dqm_ml_core.models.global_ import StorageConfig
            from dqm_ml_job.utils import get_s3_filesystem

            storage_config = StorageConfig.model_validate(storage_cfg)
            if storage_config.type == "s3":
                self.s3_fs = get_s3_filesystem(storage_config)

        self.size: tuple[int, int] = (cfg.infer.width, cfg.infer.height)
        self.batch_size: int = cfg.infer.batch_size
        self.arch: str = cfg.model.arch
        n_layer_feature = cfg.model.n_layer_feature

        # Multi-layer support for CMD: n_layer_feature can be a list
        if isinstance(n_layer_feature, list):
            self.multi_layer = True
            self.target_layers: list[str] = n_layer_feature
            self.target_layer: Any = n_layer_feature
            self._embed_dims: dict[str, int] = {}
        else:
            self.multi_layer = False
            self.target_layer = n_layer_feature
            self._embed_dim: int | None = None

        # Build transform (fast, no model needed)
        safe_std = [s if s != 0 else 1e-12 for s in cfg.infer.norm_std]
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.infer.norm_mean, std=safe_std),
            ]
        )

        # Model and extractor — loaded lazily by _ensure_model_loaded()
        self.model: Any = None
        self.feature_extractor: Any = None
        self.device = "cpu"
        self._model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """Load the PyTorch model and create the feature extractor.

        This is deferred from ``__init__`` because:
        - Model loading is expensive (download + GPU allocation).
        - ``compute_device`` is injected by DatasetJob after __init__.
        """
        if self._model_loaded:
            return
        cfg = FeaturesEmbeddingsProcessorConfig.model_validate({**self.config, "name": self.name})
        compute_device = getattr(self, "compute_device", None)
        self.device = self._resolve_device(compute_device) if compute_device else self._resolve_device(cfg.model.device)
        self.model = self._load_model(self.arch, self.device)
        self.feature_extractor = self._make_extractor(self.model, self.target_layer)
        self._model_loaded = True

    def check_config(self) -> None:
        """Validate configuration and load model.

        Kept for backward compatibility. Delegates to ``_ensure_model_loaded``.
        """
        self._ensure_model_loaded()

    @override
    def needed_columns(self) -> list[str]:
        """Return the list of columns required for image embedding extraction.

        Returns:
            List of input column names.
        """
        return self.input_columns

    def _output_column_name(self, col: str, base: str) -> str:
        """Generate output column name with prefix and suffix.

        Args:
            col: Input column name.
            base: Base feature name (e.g., "embedding", "emb_layer1").

        Returns:
            Fully qualified output column name with prefix and suffix applied.
        """
        p = self.columns_config.prefix or "" if self.columns_config else ""
        s = self.columns_config.suffix or "" if self.columns_config else ""
        return f"{p}{col}_{base}{s}"

    def generated_columns(self) -> list[str]:
        """Return the list of columns generated by this processor.

        For multi-layer mode, returns one column per layer per input column.
        For single-layer mode, returns one embedding column per input column.

        Returns:
            A list of column names.
        """
        if not self.input_columns:
            return []
        cols: list[str] = []
        for col in self.input_columns:
            if getattr(self, "multi_layer", False):
                for layer in self.target_layers:
                    layer_base = f"emb_{layer.replace('.', '_')}"
                    cols.append(self._output_column_name(col, layer_base))
                    cols.append(self._output_column_name(col, f"{layer_base}_channels"))
            else:
                cols.append(self._output_column_name(col, "embedding"))
        return cols

    def _open_image(self, image_data: Any, column: str) -> Image.Image:
        """Open a PIL Image from bytes, S3 path, or local filesystem path."""
        if isinstance(image_data, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        return self._open_image_from_path(image_data, column)

    def _open_image_from_path(self, path: str, column: str) -> Image.Image:
        """Open a PIL Image from an S3 or local filesystem path."""
        prefix = self._current_image_prefix(column)
        if prefix is not None and self.s3_fs:
            return self._open_s3_image(prefix, path)
        full_path = Path(prefix) / path if prefix else Path(path)
        return Image.open(full_path).convert("RGB")

    def _open_s3_image(self, prefix: str, path: str) -> Image.Image:
        """Open a PIL Image from S3, reading into memory and copying."""
        bucket = os.getenv("S3_BUCKET_NAME", "")
        s3_path = f"{bucket}/{prefix}/{path}"
        with self.s3_fs.open_input_stream(s3_path) as f:  # type: ignore[union-attr]
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            return img.copy()

    def _handle_load_error(self, exc: Exception, idx: int) -> None:
        """Check error config and either raise or record the failure."""
        self._check_image_fail_fast(exc, "on_decode_failure", "on_transform_error")
        self._failure_count += 1
        self._total_count += 1
        self._check_failure_rate()
        logger.warning(f"[ImageEmbeddingProcessor] failed to load image: {exc}")

    def _load_single_tensor(self, image_data: Any, column: str, idx: int) -> torch.Tensor | None:
        """Load, transform, and return a single image tensor (or None on failure)."""
        if image_data is None:
            return None
        try:
            pil_image = self._open_image(image_data, column)
            return self.transform(pil_image)  # type: ignore[no-any-return]
        except Exception as e:
            self._handle_load_error(e, idx)
            return None

    def _load_image_tensors(
        self,
        image_values: list[Any],
        column: str = "",
    ) -> list[torch.Tensor | None]:
        """Load and transform images from a list of raw image values.

        Auto-detects between bytes and path based on Python type.

        Args:
            image_values: List of raw image column values.
            column: The input column name (used to resolve path prefix).

        Returns:
            List of preprocessed image tensors (or None for failed loads).
        """
        return [self._load_single_tensor(v, column, idx) for idx, v in enumerate(image_values)]

    def _current_image_prefix(self, column: str) -> str | None:
        """Return the path prefix for the given column.

        Reads from ``self.current_path_prefix``, a dict set by the job
        mapping column names to path prefixes.
        """
        prefix_map: dict[str, str] = getattr(self, "current_path_prefix", {})
        return prefix_map.get(column)

    @override
    def compute_features(self, batch: pa.RecordBatch, prev_features: pa.Array = None) -> dict[str, pa.Array]:
        """
        Extract image embeddings for all samples in the batch.

        1. Images are loaded and transformed.
        2. Model inference is performed in sub-batches defined by `infer.batch_size`.
        3. Results are aggregated into a pyarrow `FixedSizeListArray`.

        Args:
            batch: Raw pyarrow batch.
            prev_features: Pre-computed features (not used).

        Returns:
            Dictionary mapping column-prefixed embedding names to arrays.
        """
        self._ensure_model_loaded()

        available = batch.schema.names
        cols = resolve_include_exclude(
            self.input_columns or None,
            self.exclude_columns or None,
            available,
        )
        if not cols:
            logger.warning(f"[{self.name}] no input columns matched in batch")
            return {}

        result: dict[str, pa.Array] = {}
        for col in cols:
            if col not in available:
                logger.warning(f"[ImageEmbeddingProcessor] missing column '{col}'")
                continue

            image_values = batch.column(col).to_pylist()
            image_tensors = self._load_image_tensors(image_values, column=col)

            self.feature_extractor.eval()
            with torch.no_grad():
                if self.multi_layer:
                    raw = self._compute_features_multi_layer(image_tensors)
                else:
                    raw = self._compute_features_single_layer(image_tensors)

            for k, v in raw.items():
                result[self._output_column_name(col, k)] = v

        return result

    def _build_fixed_array(self, embs: list[np.ndarray | None], embed_dim: int) -> pa.FixedSizeListArray:
        """Build a FixedSizeListArray from a list of embedding vectors.

        Args:
            embs: List of embedding arrays or None.
            embed_dim: Expected dimension of each embedding.

        Returns:
            A FixedSizeListArray of float32.
        """
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        flat: list[float] = []
        for emb in embs:
            if emb is None:
                flat.extend([0.0] * embed_dim)
            else:
                flat_emb = emb.ravel()
                if flat_emb.size != embed_dim:
                    if flat_emb.size > embed_dim:
                        flat_emb = flat_emb[:embed_dim]
                    else:
                        flat_emb = np.pad(flat_emb, (0, embed_dim - flat_emb.size))
                flat.extend(flat_emb.tolist())
        flat_array = pa.array(np.asarray(flat, dtype=np.float32))
        return pa.FixedSizeListArray.from_arrays(flat_array, embed_dim)

    def _compute_features_single_layer(self, image_tensors: list[torch.Tensor | None]) -> dict[str, pa.Array]:
        """Compute embeddings for a single target layer.

        Args:
            image_tensors: List of preprocessed image tensors or None.

        Returns:
            Dictionary with 'embedding' key.
        """
        embs: list[np.ndarray | None] = []
        with torch.no_grad():
            for batch_start in range(0, len(image_tensors), self.batch_size):
                batch_slice = image_tensors[batch_start : batch_start + self.batch_size]
                self._process_batch_single(batch_slice, embs)

        embed_dim = self._infer_embed_dim(embs)
        if embed_dim is None or embed_dim <= 0:
            return {}
        return {"embedding": self._build_fixed_array(embs, embed_dim)}

    def _process_batch_single(self, batch_slice: list[torch.Tensor | None], embs: list[np.ndarray | None]) -> None:
        """Process a single batch for single-layer embedding extraction.

        Args:
            batch_slice: Subset of image tensors.
            embs: Output list to append embeddings to.
        """
        valid = [t for t in batch_slice if t is not None]
        if not valid:
            embs.extend([None] * len(batch_slice))
            return

        batch_tensor = torch.stack(valid).to(self.device)
        out = self.feature_extractor(batch_tensor)
        if isinstance(out, dict):
            flat_feats = [layer_output.flatten(1) for layer_output in out.values()]
            feats = torch.cat(flat_feats, dim=1)
        else:
            feats = out.flatten(1) if out.dim() > 2 else out
        batch_embeddings_np = feats.detach().cpu().numpy().astype("float32")

        pos = 0
        for item_or_none in batch_slice:
            if item_or_none is None:
                embs.append(None)
            else:
                embs.append(batch_embeddings_np[pos])
                pos += 1

    def _infer_embed_dim(self, embs: list[np.ndarray | None]) -> int | None:
        """Infer embedding dimension from the first valid embedding.

        Args:
            embs: List of embeddings or None.

        Returns:
            Embedding dimension, or None if no valid embeddings exist.
        """
        if self._embed_dim is not None:
            return self._embed_dim
        for emb in embs:
            if emb is not None:
                self._embed_dim = int(emb.size)
                return self._embed_dim
        return None

    def _compute_features_multi_layer(self, image_tensors: list[torch.Tensor | None]) -> dict[str, pa.Array]:
        """Compute embeddings for multiple target layers.

        Each layer's output is flattened and stored in a separate column
        named ``emb_<layer_name>`` (with dots replaced by underscores).

        Args:
            image_tensors: List of preprocessed image tensors or None.

        Returns:
            Dictionary mapping layer column names to FixedSizeListArrays.
        """
        layer_cols = [f"emb_{layer.replace('.', '_')}" for layer in self.target_layers]
        channel_cols = [f"{col}_channels" for col in layer_cols]
        per_layer_embs: dict[str, list[np.ndarray | None]] = {col: [] for col in layer_cols}
        per_layer_channels: dict[str, list[int | None]] = {col: [] for col in channel_cols}

        with torch.no_grad():
            for batch_start in range(0, len(image_tensors), self.batch_size):
                batch_slice = image_tensors[batch_start : batch_start + self.batch_size]
                self._process_batch_multi(batch_slice, layer_cols, channel_cols, per_layer_embs, per_layer_channels)

        return self._build_multi_layer_results(layer_cols, channel_cols, per_layer_embs, per_layer_channels)

    def _build_batch_np_dict(
        self,
        out_dict: dict[str, torch.Tensor],
        valid_len: int,
    ) -> dict[str, np.ndarray]:
        """Build per-layer numpy arrays from a batch of forward pass outputs.

        Args:
            out_dict: Output dict from the feature extractor.
            valid_len: Number of valid (non-None) samples in the batch.

        Returns:
            Dict mapping layer/column names to numpy arrays.
        """
        batch_np_dict: dict[str, np.ndarray] = {}
        for layer_name in self.target_layers:
            col = f"emb_{layer_name.replace('.', '_')}"
            feats = out_dict[layer_name]
            flat_feats = feats.flatten(1) if feats.dim() > 2 else feats
            batch_np_dict[col] = flat_feats.detach().cpu().numpy().astype("float32")
            batch_np_dict[f"{col}_channels"] = np.full(valid_len, feats.shape[1], dtype=np.int32)
        return batch_np_dict

    @staticmethod
    def _append_none_row(
        layer_cols: list[str],
        channel_cols: list[str],
        per_layer_embs: dict[str, list[np.ndarray | None]],
        per_layer_channels: dict[str, list[int | None]],
    ) -> None:
        """Append None entries for all layer/channel columns."""
        for col in layer_cols:
            per_layer_embs[col].append(None)
        for col in channel_cols:
            per_layer_channels[col].append(None)

    @staticmethod
    def _append_valid_row(
        pos: int,
        layer_cols: list[str],
        channel_cols: list[str],
        batch_np_dict: dict[str, np.ndarray],
        per_layer_embs: dict[str, list[np.ndarray | None]],
        per_layer_channels: dict[str, list[int | None]],
    ) -> None:
        """Append embeddings for a valid (non-None) item at the given position."""
        for col in layer_cols:
            per_layer_embs[col].append(batch_np_dict[col][pos])
        for col in channel_cols:
            per_layer_channels[col].append(int(batch_np_dict[col][pos]))

    @staticmethod
    def _append_batch_results(
        batch_slice: list[torch.Tensor | None],
        layer_cols: list[str],
        channel_cols: list[str],
        batch_np_dict: dict[str, np.ndarray],
        per_layer_embs: dict[str, list[np.ndarray | None]],
        per_layer_channels: dict[str, list[int | None]],
    ) -> None:
        """Append per-layer results for a batch to the per-layer collections.

        Args:
            batch_slice: Subset of image tensors.
            layer_cols: Layer column names.
            channel_cols: Channel column names.
            batch_np_dict: Numpy arrays per column.
            per_layer_embs: Per-layer embedding lists to append to.
            per_layer_channels: Per-layer channel lists to append to.
        """
        pos = 0
        for item_or_none in batch_slice:
            if item_or_none is None:
                ImageEmbeddingProcessor._append_none_row(layer_cols, channel_cols, per_layer_embs, per_layer_channels)
            else:
                ImageEmbeddingProcessor._append_valid_row(
                    pos, layer_cols, channel_cols, batch_np_dict, per_layer_embs, per_layer_channels
                )
                pos += 1

    def _process_batch_multi(
        self,
        batch_slice: list[torch.Tensor | None],
        layer_cols: list[str],
        channel_cols: list[str],
        per_layer_embs: dict[str, list[np.ndarray | None]],
        per_layer_channels: dict[str, list[int | None]],
    ) -> None:
        """Process a single batch for multi-layer embedding extraction.

        Args:
            batch_slice: Subset of image tensors.
            layer_cols: Layer column names.
            channel_cols: Channel column names.
            per_layer_embs: Per-layer embedding lists to append to.
            per_layer_channels: Per-layer channel lists to append to.
        """
        valid = [t for t in batch_slice if t is not None]
        if not valid:
            for col in layer_cols:
                per_layer_embs[col].extend([None] * len(batch_slice))
            for col in channel_cols:
                per_layer_channels[col].extend([None] * len(batch_slice))
            return

        batch_tensor = torch.stack(valid).to(self.device)
        out_dict = self.feature_extractor(batch_tensor)
        batch_np_dict = self._build_batch_np_dict(out_dict, len(valid))
        ImageEmbeddingProcessor._append_batch_results(
            batch_slice, layer_cols, channel_cols, batch_np_dict, per_layer_embs, per_layer_channels
        )

    def _build_multi_layer_results(
        self,
        layer_cols: list[str],
        channel_cols: list[str],
        per_layer_embs: dict[str, list[np.ndarray | None]],
        per_layer_channels: dict[str, list[int | None]],
    ) -> dict[str, pa.Array]:
        """Build the result dictionary from per-layer collections.

        Args:
            layer_cols: Layer column names.
            channel_cols: Channel column names.
            per_layer_embs: Per-layer embedding lists.
            per_layer_channels: Per-layer channel lists.

        Returns:
            Dictionary mapping column names to Arrow arrays.
        """
        result: dict[str, pa.Array] = {}
        for col in layer_cols:
            embs = per_layer_embs[col]
            embed_dim = None
            for emb in embs:
                if emb is not None:
                    embed_dim = int(emb.size)
                    break
            if embed_dim is None or embed_dim == 0:
                continue
            result[col] = self._build_fixed_array(embs, embed_dim)
        for col in channel_cols:
            vals = [v if v is not None else 0 for v in per_layer_channels[col]]
            if any(v is not None for v in per_layer_channels[col]):
                result[col] = pa.array(vals, type=pa.int32())
        return result

    @override
    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """No-op metric computation for image embedding processor.

        Embeddings are stored as features; no batch-level aggregation
        is needed.

        Args:
            features: Dictionary of feature arrays from the batch.

        Returns:
            Empty dictionary.
        """
        return {}

    @override
    def compute(self, batch_metrics: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Compute final dataset-level metrics (not used for embeddings).

        Returns:
            Empty dictionary as embeddings are computed at feature level.
        """
        return {}

    @override
    def compute_delta(self, source: dict[str, pa.Array], target: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Compute delta between source and target embeddings (not used).

        Args:
            source: Source embeddings (not used).
            target: Target embeddings (not used).

        Returns:
            Empty dictionary as delta computation is handled by DomainGapProcessor.
        """
        return {}

    # utils functions
    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``"auto"`` to CUDA if available, else CPU."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, arch: str, device: str) -> Any:
        """Load a pre-trained torchvision model.

        Args:
            arch: Model architecture name (e.g., 'resnet18', 'resnet50').
            device: Device to load the model on ('cpu' or 'cuda').

        Returns:
            The loaded PyTorch model.
        """
        try:
            model = torchvision.models.get_model(arch, weights="DEFAULT")
        except Exception:
            # Fallback for older torchvision that lacks get_model()
            model = getattr(torchvision.models, arch)(pretrained=True)
        return model.to(device)

    def _make_extractor(self, model: torch.nn.Module, target_layer: Any) -> Any:
        """Create a feature extractor from a model.

        Args:
            model: The PyTorch model to extract features from.
            target_layer: Layer name (str), index (int), or list of names to extract.

        Returns:
            A feature extractor that returns the requested layer outputs.
        """
        names = list(dict(model.named_modules()).keys())
        if isinstance(target_layer, list):
            nodes = {n: n for n in target_layer}
        elif isinstance(target_layer, int):
            idx = target_layer if target_layer >= 0 else len(names) + target_layer
            layer_name = names[idx]
            nodes = {layer_name: "features"}
        else:
            nodes = {target_layer: "features"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return create_feature_extractor(model, return_nodes=nodes)
