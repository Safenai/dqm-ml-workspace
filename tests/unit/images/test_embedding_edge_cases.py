"""Unit tests for image embedding edge cases (PIL + transforms + embedding pipeline).

Exercises internal methods of ImageEmbeddingProcessor for extreme/corner-case
image inputs using mocked model state.
"""

import io
from unittest.mock import MagicMock

from dqm_ml_pytorch.image_embedding import ImageEmbeddingProcessor
import numpy as np
from PIL import Image
import pyarrow as pa
import pytest
import torch


@pytest.fixture(scope="module")
def tiny_image_bytes() -> bytes:
    """Provide a small valid JPEG image as bytes.

    Returns:
        Bytes of a 10x10 RGB JPEG image.
    """
    img = Image.new("RGB", (10, 10), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


@pytest.fixture(scope="module")
def huge_image_bytes() -> bytes:
    """Provide a large valid JPEG image as bytes.

    Returns:
        Bytes of a 4000x4000 RGB JPEG image (low quality to keep size small).
    """
    img = Image.new("RGB", (4000, 4000), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=10)
    return buf.getvalue()


@pytest.fixture(scope="module")
def corrupted_bytes() -> bytes:
    """Provide invalid image bytes for testing error handling.

    Returns:
        Bytes that are not a valid image format.
    """
    return b"not a valid image file"


def test_tiny_image_transform(tiny_image_bytes: bytes) -> None:
    """Verify tiny image transforms correctly through the pipeline.

    Args:
        tiny_image_bytes: Fixture providing small JPEG image bytes.
    """
    proc = ImageEmbeddingProcessor(
        name="test",
        config={"columns": {"input": ["img"]}, "infer": {"width": 224, "height": 224}},
    )
    tensors = proc._load_image_tensors([tiny_image_bytes])
    assert len(tensors) == 1
    assert tensors[0] is not None
    assert tensors[0].shape == (3, 224, 224)


def test_large_image_transform(huge_image_bytes: bytes) -> None:
    """Verify large image transforms correctly through the pipeline.

    Args:
        huge_image_bytes: Fixture providing large JPEG image bytes.
    """
    proc = ImageEmbeddingProcessor(
        name="test",
        config={
            "columns": {"input": ["img"]},
            "infer": {"width": 224, "height": 224},
            "model": {"arch": "resnet18"},
        },
    )
    proc._model_loaded = True
    proc.model = MagicMock()
    fake_emb = np.array([[0.1] * 512])
    feat_ext = MagicMock()
    feat_ext.eval = MagicMock()
    feat_ext.return_value = {"features": torch.tensor(fake_emb.reshape(1, 512, 1, 1))}
    proc.feature_extractor = feat_ext
    batch = pa.RecordBatch.from_pydict({"img": pa.array([huge_image_bytes])})
    result = proc.compute_features(batch)
    assert isinstance(result, dict)


def test_corrupted_image_bytes(corrupted_bytes: bytes) -> None:
    """Verify corrupted image bytes returns empty result.

    Args:
        corrupted_bytes: Fixture providing invalid image bytes.
    """
    proc = ImageEmbeddingProcessor(
        name="test",
        config={"columns": {"input": ["img"]}, "model": {"arch": "resnet18"}},
    )
    proc._model_loaded = True
    proc.model = MagicMock()
    feat_ext = MagicMock()
    feat_ext.eval = MagicMock()
    proc.feature_extractor = feat_ext
    batch = pa.RecordBatch.from_pydict({"img": pa.array([corrupted_bytes])})
    result = proc.compute_features(batch)
    assert result == {}


def test_norm_std_zero_produces_inf() -> None:
    """Verify zero norm_std produces infinite values in tensor."""
    proc = ImageEmbeddingProcessor(
        name="test",
        config={
            "columns": {"input": ["img"]},
            "infer": {"norm_mean": [0, 0, 0], "norm_std": [0, 0, 0]},
        },
    )
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    tensors = proc._load_image_tensors([buf.getvalue()])
    assert len(tensors) == 1
    t = tensors[0]
    assert t is not None
    assert torch.any(t > 1e11).item()


def test_small_image_through_full_embedding_pipeline(tiny_image_bytes: bytes) -> None:
    """Verify small image passes through full embedding pipeline.

    Args:
        tiny_image_bytes: Fixture providing small JPEG image bytes.
    """
    proc = ImageEmbeddingProcessor(
        name="test",
        config={
            "columns": {"input": ["img"]},
            "infer": {"width": 224, "height": 224},
            "model": {"arch": "resnet18"},
        },
    )
    proc._model_loaded = True
    fake_emb = np.array([[0.1] * 512])
    feat_ext = MagicMock()
    feat_ext.eval = MagicMock()
    feat_ext.return_value = {"features": torch.tensor(fake_emb.reshape(1, 512, 1, 1))}
    proc.feature_extractor = feat_ext
    proc.model = MagicMock()
    batch = pa.RecordBatch.from_pydict({"img": pa.array([tiny_image_bytes])})
    result = proc.compute_features(batch)
    assert result, "Expected non-empty result for a valid tiny image"
    col = next(iter(result))
    arr = result[col]
    assert len(arr) == 1
