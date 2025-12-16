from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, override

import numpy as np
from PIL import Image
import pyarrow as pa
import torch
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

from dqm_ml_core import DatametricProcessor

logger = logging.getLogger(__name__)


class ImageEmbeddingProcessor(DatametricProcessor):
    """
    Compute one latent vector per row from images stored in Parquet.
    TODO: tester aussi les paths des images
    Config:
      input_columns:
        image_column: "image_bytes" | "image_path" (default: "image_bytes")
        mode: "bytes" | "path" (default: "bytes")
      infer:
        width, height, batch_size, norm_mean, norm_std
      model_config:
        arch: torchvision model name (default: "resnet18")
        n_layer_feature: layer name | index | list[str] (default: "avgpool")
        device: "cpu" | "cuda" (default: "cpu")
    """

    def __init__(self, name: str = "image_embedding", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._checked = False

    # ---------------- API ----------------
    def check_config(self) -> None:
        cfg = self.config or {}

        dcfg = cfg.get("DATA", {})
        self.image_column: str = dcfg.get("image_column", "image_bytes")
        self.mode: str = dcfg.get("mode", "bytes")  # "bytes" or "path"
        if self.mode not in {"bytes", "path"}:
            raise ValueError(f"[{self.name}] DATA.mode must be 'bytes' or 'path'")

        # handle relative paths in parquet to a dataset located at dataset_root_path
        self.dataset_root_path = str(cfg.get("dataset_root_path", "undefined"))
        logger.info(f"[ImageEmbeddingProcessor] dataset_root_path = '{self.dataset_root_path}'")

        icfg = cfg.get("INFER", {})
        self.size: tuple[int, int] = (int(icfg.get("width", 224)), int(icfg.get("height", 224)))
        mean = icfg.get("norm_mean", [0.485, 0.456, 0.406])
        std = icfg.get("norm_std", [0.229, 0.224, 0.225])
        self.batch_size: int = int(icfg.get("batch_size", 32))

        mcfg = cfg.get("MODEL", {})
        self.arch: str = mcfg.get("arch", "resnet18")
        self.nodes = mcfg.get("n_layer_feature", "avgpool")
        self.device: str = mcfg.get("device", "cpu")

        # Build once
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.model = self._load_model(self.arch, self.device)
        self.fx = self._make_extractor(self.model, self.nodes)
        self._embed_dim: int | None = None

        self._checked = True

    @override
    def needed_columns(self) -> list[str]:
        if not getattr(self, "_checked", False):
            self.check_config()
        return [self.image_column]

    def generated_columns(self) -> list[str]:
        return ["embedding"]

    @override
    def compute_features(self, batch: pa.RecordBatch, prev_features: pa.Array = None) -> dict[str, pa.Array]:
        if not getattr(self, "_checked", False):
            self.check_config()
        if self.image_column not in batch.schema.names:
            logger.warning(f"[ImageEmbeddingProcessor] missing column '{self.image_column}'")
            return {}

        # 1 load images
        vals = batch.column(self.image_column).to_pylist()
        imgs: list[torch.Tensor | None] = []
        for v in vals:
            if v is None:
                imgs.append(None)
                continue
            try:
                if self.mode == "bytes":
                    img = Image.open(io.BytesIO(v)).convert("RGB")
                else:
                    img_path = Path(self.dataset_root_path) / v if self.dataset_root_path != "undefined" else Path(v)
                    img = Image.open(img_path).convert("RGB")
                imgs.append(self.transform(img))
            except Exception as e:
                logger.warning(f"[ImageEmbeddingProcessor] failed to load image: {e}")
                imgs.append(None)

        # inference in windows, preserve order
        embs: list[np.ndarray | None] = []
        self.fx.eval()
        with torch.no_grad():
            i = 0
            while i < len(imgs):
                window = imgs[i : i + self.batch_size]
                valid = [t for t in window if t is not None]
                if valid:
                    bt = torch.stack(valid).to(self.device)
                    out = self.fx(bt)
                    if isinstance(out, dict):
                        flat_feats = [v.flatten(1) for v in out.values()]
                        feats = torch.cat(flat_feats, dim=1)  # type : ignore TODO : check type error
                    else:
                        feats = out.flatten(1) if out.dim() > 2 else out
                    arr = feats.detach().cpu().numpy().astype("float32")
                    p = 0
                    for t in window:
                        if t is None:
                            embs.append(None)
                        else:
                            embs.append(arr[p])
                            p += 1
                else:
                    embs.extend([None] * len(window))
                i += self.batch_size

        # 3. Infer embedding dim
        if self._embed_dim is None:
            for emb in embs:
                if emb is not None:
                    self._embed_dim = int(emb.size)
                    break
            if self._embed_dim is None:
                return {}
        d = self._embed_dim

        # 4. Build FixedSizeListArray
        flat: list[float] = []
        for emb in embs:
            if emb is None:
                flat.extend([0.0] * d)
            else:
                v = emb.ravel()
                if v.size != d:
                    v = v[:d] if v.size > d else np.pad(v, (0, d - v.size))
                flat.extend(v.tolist())

        child = pa.array(np.asarray(flat, dtype=np.float32))
        return {"embedding": pa.FixedSizeListArray.from_arrays(child, d)}

    @override
    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        return features

    @override
    def compute(self, batch_metrics: dict[str, pa.Array]) -> dict[str, pa.Array]:
        return {}

    @override
    def compute_delta(self, source: dict[str, pa.Array], target: dict[str, pa.Array]) -> dict[str, pa.Array]:
        return {}

    # utils functions
    def _load_model(self, arch: str, device: str) -> Any:
        try:
            m = torchvision.models.get_model(arch, weights="DEFAULT")
        except Exception:
            m = getattr(torchvision.models, arch)(pretrained=True)
        return m.to(device)

    def _make_extractor(self, model: torch.nn.Module, nodes: Any) -> Any:
        names = list(dict(model.named_modules()).keys())
        if isinstance(nodes, list):
            return create_feature_extractor(model, return_nodes={n: n for n in nodes})
        if isinstance(nodes, int):
            idx = nodes if nodes >= 0 else len(names) + nodes
            nodes = names[idx]
        return create_feature_extractor(model, return_nodes={nodes: "features"})
