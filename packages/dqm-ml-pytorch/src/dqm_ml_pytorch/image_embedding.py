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
                - DATA:
                    - image_column: Column name containing image data (default: "image_bytes").
                    - mode: Source type, "bytes" or "path" (default: "bytes").
                - INFER:
                    - width, height: Input resolution for the model (default: 224x224).
                    - batch_size: Number of images per inference pass (default: 32).
                    - norm_mean, norm_std: Preprocessing normalization stats.
                - MODEL:
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

        data_cfg = cfg.get("DATA", {})
        self.image_column: str = data_cfg.get("image_column", "image_bytes")
        self.mode: str = data_cfg.get("mode", "bytes")  # "bytes" or "path"
        if self.mode not in {"bytes", "path"}:
            raise ValueError(f"[{self.name}] DATA.mode must be 'bytes' or 'path'")

        # handle relative paths in parquet to a dataset located at dataset_root_path
        self.dataset_root_path = str(cfg.get("dataset_root_path", "undefined"))
        logger.info(f"[ImageEmbeddingProcessor] dataset_root_path = '{self.dataset_root_path}'")

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

        infer_cfg = cfg.get("INFER", {})
        self.size: tuple[int, int] = (
            int(infer_cfg.get("width", 224)),
            int(infer_cfg.get("height", 224)),
        )
        mean = infer_cfg.get("norm_mean", [0.485, 0.456, 0.406])
        std = infer_cfg.get("norm_std", [0.229, 0.224, 0.225])
        self.batch_size: int = int(infer_cfg.get("batch_size", 32))

        model_cfg = cfg.get("MODEL", {})
        self.arch: str = model_cfg.get("arch", "resnet18")
        self.target_layer = model_cfg.get("n_layer_feature", "avgpool")
        self.device: str = model_cfg.get("device", "cpu")

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
        self._embed_dim: int | None = None

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

        Returns:
            A list containing 'embedding'.
        """
        return ["embedding"]

    @override
    def compute_features(self, batch: pa.RecordBatch, prev_features: pa.Array = None) -> dict[str, pa.Array]:
        """
        Extract image embeddings for all samples in the batch.

        1. Images are loaded and transformed.
        2. Model inference is performed in sub-batches defined by `INFER.batch_size`.
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
                    if self.dataset_root_path != "undefined" and self.s3_fs:
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
                            if self.dataset_root_path != "undefined"
                            else Path(image_data)
                        )
                        img = Image.open(img_path).convert("RGB")
                image_tensors.append(self.transform(img))
            except Exception as e:
                logger.warning(f"[ImageEmbeddingProcessor] failed to load image: {e}")
                image_tensors.append(None)

        # inference in windows, preserve order
        embs: list[np.ndarray | None] = []
        self.feature_extractor.eval()
        with torch.no_grad():
            for batch_start in range(0, len(image_tensors), self.batch_size):
                batch_slice = image_tensors[batch_start : batch_start + self.batch_size]
                valid = [t for t in batch_slice if t is not None]
                if valid:
                    batch_tensor = torch.stack(valid).to(self.device)
                    # create_feature_extractor returns dict when multiple nodes requested, else a single tensor
                    out = self.feature_extractor(batch_tensor)
                    if isinstance(out, dict):
                        flat_feats = [layer_output.flatten(1) for layer_output in out.values()]
                        feats = torch.cat(flat_feats, dim=1)  # type : ignore TODO : check type error
                    else:
                        feats = out.flatten(1) if out.dim() > 2 else out
                    batch_embeddings_np = feats.detach().cpu().numpy().astype("float32")
                    # Re-align None images with their original positions in the batch window
                    pos = 0
                    for item_or_none in batch_slice:
                        if item_or_none is None:
                            embs.append(None)
                        else:
                            embs.append(batch_embeddings_np[pos])
                            pos += 1
                else:
                    embs.extend([None] * len(batch_slice))

        # 3. Infer embedding dim
        if self._embed_dim is None:
            for emb in embs:
                if emb is not None:
                    self._embed_dim = int(emb.size)
                    break
            if self._embed_dim is None:
                return {}
        embed_dim = self._embed_dim

        # 4. Build FixedSizeListArray
        flat: list[float] = []
        for emb in embs:
            if emb is None:
                flat.extend([0.0] * embed_dim)
            else:
                flat_emb = emb.ravel()
                # Handle rare dimension mismatches by truncating or zero-padding
                if flat_emb.size != embed_dim:
                    if flat_emb.size > embed_dim:
                        flat_emb = flat_emb[:embed_dim]
                    else:
                        flat_emb = np.pad(flat_emb, (0, embed_dim - flat_emb.size))
                flat.extend(flat_emb.tolist())

        flat_array = pa.array(np.asarray(flat, dtype=np.float32))
        return {"embedding": pa.FixedSizeListArray.from_arrays(flat_array, embed_dim)}

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
            return create_feature_extractor(model, return_nodes={n: n for n in target_layer})
        if isinstance(target_layer, int):
            idx = target_layer if target_layer >= 0 else len(names) + target_layer
            layer_name = names[idx]
            return create_feature_extractor(model, return_nodes={layer_name: "features"})
        return create_feature_extractor(model, return_nodes={target_layer: "features"})
