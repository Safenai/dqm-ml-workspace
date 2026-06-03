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
                - data:
                    - image_column: Column name containing image data (default: "image_bytes").
                    - mode: Source type, "bytes" or "path" (default: "bytes").
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
        self._checked = False

    # ---------------- API ----------------
    def check_config(self) -> None:
        """Validate and initialize model/transforms from configuration.

        This method parses the configuration dictionary and initializes:
        - Image loading parameters (column name, mode, dataset root path)
        - Inference parameters (image size, batch size, normalization)
        - Model parameters (architecture, feature extraction layer, device)
        - Loads the pre-trained model and creates the feature extractor.
        """
        cfg = self.config or {}

        data_cfg = cfg.get("data", {})
        self.image_column: str = data_cfg.get("image_column", "image_bytes")
        self.mode: str = data_cfg.get("mode", "bytes")  # "bytes" or "path"
        if self.mode not in {"bytes", "path"}:
            raise ValueError(f"[{self.name}] data.mode must be 'bytes' or 'path'")

        # handle relative paths in parquet to a dataset located at dataset_root_path
        self.dataset_root_path = cfg.get("dataset_root_path", None)
        logger.info(f"[ImageEmbeddingProcessor] dataset_root_path = '{self.dataset_root_path}'")

        # Storage filesystem support
        self.s3_fs = None
        storage_cfg = cfg.get("storage")
        if storage_cfg:
            from dqm_ml_job.utils import get_s3_filesystem

            if storage_cfg is True:
                self.s3_fs = get_s3_filesystem()
            elif isinstance(storage_cfg, dict) and storage_cfg.get("type") == "s3":
                self.s3_fs = get_s3_filesystem(
                    access_key=storage_cfg.get("access_key"),
                    secret_key=storage_cfg.get("secret_key"),
                    endpoint=storage_cfg.get("endpoint_override"),
                    region=storage_cfg.get("region"),
                )

        infer_cfg = cfg.get("infer", {})
        self.size: tuple[int, int] = (
            int(infer_cfg.get("width", 224)),
            int(infer_cfg.get("height", 224)),
        )
        mean = infer_cfg.get("norm_mean", [0.485, 0.456, 0.406])
        std = infer_cfg.get("norm_std", [0.229, 0.224, 0.225])
        self.batch_size: int = int(infer_cfg.get("batch_size", 32))

        model_cfg = cfg.get("model", {})
        self.arch: str = model_cfg.get("arch", "resnet18")
        n_layer_feature = model_cfg.get("n_layer_feature", "avgpool")
        self.device: str = model_cfg.get("device", "cpu")

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

        # Build once
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.model = self._load_model(self.arch, self.device)
        self.feature_extractor = self._make_extractor(self.model, self.target_layer)
        self._checked = True

    @override
    def needed_columns(self) -> list[str]:
        """Return the list of columns required for image embedding extraction.

        Returns:
            List containing the image column name.
        """
        if not getattr(self, "_checked", False):
            self.check_config()
        return [self.image_column]

    def generated_columns(self) -> list[str]:
        """Return the list of columns generated by this processor.

        For multi-layer mode, returns one column per layer prefixed with ``emb_``.
        For single-layer mode, returns ``['embedding']``.

        Returns:
            A list of column names.
        """
        if getattr(self, "multi_layer", False):
            return [f"emb_{layer.replace('.', '_')}" for layer in self.target_layers]
        return ["embedding"]

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
            Dictionary mapping 'embedding' to the calculated feature vectors.
        """
        if not getattr(self, "_checked", False):
            self.check_config()
        if self.image_column not in batch.schema.names:
            logger.warning(f"[ImageEmbeddingProcessor] missing column '{self.image_column}'")
            return {}

        # 1 load images
        image_values = batch.column(self.image_column).to_pylist()
        image_tensors: list[torch.Tensor | None] = []
        for image_data in image_values:
            if image_data is None:
                image_tensors.append(None)
                continue
            try:
                if self.mode == "bytes":
                    img = Image.open(io.BytesIO(image_data)).convert("RGB")
                else:
                    # Handle path mode with local or S3
                    if self.dataset_root_path is not None and self.s3_fs:
                        # S3 key = bucket/prefix/relative_path
                        s3_path = f"{self.dataset_root_path}/{image_data}"
                        bucket_name = os.getenv("S3_BUCKET_NAME", "")
                        s3_key = f"{bucket_name}/{s3_path}"
                        with self.s3_fs.open_input_stream(s3_key) as f:
                            img = Image.open(io.BytesIO(f.read())).convert("RGB")
                            img = img.copy()  # Load into memory before closing file
                    else:
                        img_path = (
                            Path(self.dataset_root_path) / image_data
                            if self.dataset_root_path is not None
                            else Path(image_data)
                        )
                        img = Image.open(img_path).convert("RGB")
                image_tensors.append(self.transform(img))
            except Exception as e:
                logger.warning(f"[ImageEmbeddingProcessor] failed to load image: {e}")
                image_tensors.append(None)

        # inference in windows, preserve order
        self.feature_extractor.eval()
        with torch.no_grad():
            if self.multi_layer:
                result = self._compute_features_multi_layer(image_tensors)
            else:
                result = self._compute_features_single_layer(image_tensors)
        return result

    def _build_fixed_array(self, embs: list[np.ndarray | None], embed_dim: int) -> pa.FixedSizeListArray:
        """Build a FixedSizeListArray from a list of embedding vectors.

        Args:
            embs: List of embedding arrays or None.
            embed_dim: Expected dimension of each embedding.

        Returns:
            A FixedSizeListArray of float32.
        """
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
        if embed_dim is None:
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

        batch_np_dict: dict[str, np.ndarray] = {}
        for layer_name in self.target_layers:
            col = f"emb_{layer_name.replace('.', '_')}"
            feats = out_dict[layer_name]
            flat_feats = feats.flatten(1) if feats.dim() > 2 else feats
            batch_np_dict[col] = flat_feats.detach().cpu().numpy().astype("float32")
            batch_np_dict[f"{col}_channels"] = np.full(len(valid), feats.shape[1], dtype=np.int32)

        pos = 0
        for item_or_none in batch_slice:
            if item_or_none is None:
                for col in layer_cols:
                    per_layer_embs[col].append(None)
                for col in channel_cols:
                    per_layer_channels[col].append(None)
            else:
                for col in layer_cols:
                    per_layer_embs[col].append(batch_np_dict[col][pos])
                for col in channel_cols:
                    per_layer_channels[col].append(int(batch_np_dict[col][pos]))
                pos += 1

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
