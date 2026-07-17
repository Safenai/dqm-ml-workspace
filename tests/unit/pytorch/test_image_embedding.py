"""Unit tests for the ImageEmbeddingProcessor.

This module contains tests for the PyTorch-based image embedding processor,
including generated columns, fixed array building, device resolution,
extreme value handling, and feature computation with mocked models.
"""

from unittest.mock import MagicMock

from dqm_ml_pytorch.image_embedding import ImageEmbeddingProcessor
import numpy as np
import pyarrow as pa
import torch


class TestImageEmbeddingBuildFixedArray:
    """Tests for _build_fixed_array method."""

    def test_none_embedding_zero_fills(self):
        """Verify None embeddings are zero-filled to target dimension."""
        proc = ImageEmbeddingProcessor(
            name="test",
            config={"columns": {"input": ["img"]}},
        )
        embs = [None, np.array([1.0, 2.0])]
        result = proc._build_fixed_array(embs, 3)
        assert result.type == pa.list_(pa.float32(), 3)
        assert result.to_pylist() == [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]

    def test_truncate_oversized(self):
        """Verify oversized embeddings are truncated to target dimension."""
        proc = ImageEmbeddingProcessor(
            name="test",
            config={"columns": {"input": ["img"]}},
        )
        embs = [np.array([1.0, 2.0, 3.0, 4.0])]
        result = proc._build_fixed_array(embs, 2)
        assert result.to_pylist() == [[1.0, 2.0]]

    def test_pad_undersized(self):
        """Verify undersized embeddings are zero-padded to target dimension."""
        proc = ImageEmbeddingProcessor(
            name="test",
            config={"columns": {"input": ["img"]}},
        )
        embs = [np.array([1.0])]
        result = proc._build_fixed_array(embs, 3)
        assert result.to_pylist() == [[1.0, 0.0, 0.0]]


class TestImageEmbeddingResolveDevice:
    """Tests for _resolve_device method."""

    def test_non_auto_returns_as_is(self):
        """Verify explicit device passed through unchanged."""
        assert ImageEmbeddingProcessor._resolve_device("cpu") == "cpu"
        assert ImageEmbeddingProcessor._resolve_device("cuda") == "cuda"

    def test_auto_returns_cpu_or_cuda(self):
        """Verify auto mode returns valid device."""
        result = ImageEmbeddingProcessor._resolve_device("auto")
        assert result in ("cpu", "cuda")


class TestImageEmbeddingExtremeValues:
    """Tests for extreme parameter values in image embedding."""

    def test_build_fixed_array_embed_dim_zero(self) -> None:
        """Verify _compute_features_single_layer returns empty for zero embed dim."""
        proc = ImageEmbeddingProcessor(
            name="test",
            config={"columns": {"input": ["img"]}},
        )
        result = proc._compute_features_single_layer([])
        assert result == {}

    def test_build_fixed_array_all_none(self) -> None:
        """Verify _build_fixed_array handles all None embeddings."""
        proc = ImageEmbeddingProcessor(
            name="test",
            config={"columns": {"input": ["img"]}},
        )
        result = proc._build_fixed_array([None, None], 4)
        assert result.to_pylist() == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

    def test_compute_features_batch_size_one(self) -> None:
        """Verify compute_features works with batch_size=1."""
        proc = ImageEmbeddingProcessor(
            name="test",
            config={
                "columns": {"input": ["img"]},
                "infer": {"batch_size": 1},
                "model": {"arch": "resnet18"},
            },
        )
        proc._model_loaded = True
        proc.model = MagicMock()
        feat_ext = MagicMock()
        feat_ext.eval = MagicMock()
        feat_ext.return_value = {"features": torch.rand(1, 512, 1, 1)}
        proc.feature_extractor = feat_ext
        proc._load_image_tensors = MagicMock(return_value=[torch.rand(3, 224, 224)])
        batch = pa.RecordBatch.from_pydict({"img": pa.array([b"dummy"])})
        result = proc.compute_features(batch)
        assert result

    def test_compute_features_width_height_one(self) -> None:
        """Verify compute_features works with 1x1 image dimensions."""
        proc = ImageEmbeddingProcessor(
            name="test",
            config={
                "columns": {"input": ["img"]},
                "infer": {"width": 1, "height": 1},
                "model": {"arch": "resnet18"},
            },
        )
        proc._model_loaded = True
        proc.model = MagicMock()
        feat_ext = MagicMock()
        feat_ext.eval = MagicMock()
        feat_ext.return_value = {"features": torch.rand(1, 512, 1, 1)}
        proc.feature_extractor = feat_ext
        proc._load_image_tensors = MagicMock(return_value=[torch.rand(3, 1, 1)])
        batch = pa.RecordBatch.from_pydict({"img": pa.array([b"dummy"])})
        result = proc.compute_features(batch)
        assert result

    def test_compute_features_norm_std_zero(self) -> None:
        """Verify compute_features handles zero norm_std (no normalization)."""
        proc = ImageEmbeddingProcessor(
            name="test",
            config={
                "columns": {"input": ["img"]},
                "infer": {"norm_mean": [0, 0, 0], "norm_std": [0, 0, 0]},
                "model": {"arch": "resnet18"},
            },
        )
        proc._model_loaded = True
        proc.model = MagicMock()
        feat_ext = MagicMock()
        feat_ext.eval = MagicMock()
        feat_ext.return_value = {"features": torch.rand(1, 512, 1, 1)}
        proc.feature_extractor = feat_ext
        proc._load_image_tensors = MagicMock(return_value=[torch.rand(3, 224, 224)])
        batch = pa.RecordBatch.from_pydict({"img": pa.array([b"dummy"])})
        result = proc.compute_features(batch)
        assert result


class TestImageEmbeddingComputeFeatures:
    """Tests for compute_features method."""

    def test_no_matching_columns_logs_warning(self, caplog):
        """Verify warning logged when no input columns match batch."""
        proc = ImageEmbeddingProcessor(
            name="test",
            config={"columns": {"input": ["img"]}, "model": {"arch": "resnet18"}},
        )
        proc._model_loaded = True
        proc.model = MagicMock()
        proc.feature_extractor = MagicMock()
        proc.feature_extractor.eval = MagicMock()
        batch = pa.RecordBatch.from_pydict({"other_col": pa.array([1.0])})
        result = proc.compute_features(batch)
        assert result == {}
        assert "missing column 'img'" in caplog.text.lower()

    def test_missing_column_logs_warning(self, caplog):
        """Verify warning logged when one of multiple input columns is missing."""
        proc = ImageEmbeddingProcessor(
            name="test",
            config={"columns": {"input": ["img", "missing_col"]}, "model": {"arch": "resnet18"}},
        )
        proc._model_loaded = True
        proc.model = MagicMock()
        proc.feature_extractor = MagicMock()
        proc.feature_extractor.eval = MagicMock()
        batch = pa.RecordBatch.from_pydict({"img": pa.array([b"dummy"])})
        result = proc.compute_features(batch)
        assert result == {}
        assert "missing column" in caplog.text.lower()
