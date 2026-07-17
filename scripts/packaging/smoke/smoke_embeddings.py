#!/usr/bin/env python3
"""Smoke test: dqm-ml-core + dqm-ml-pytorch (no job).

Verifies that embedding features can be computed via the Python API.
"""

import io
import sys

import numpy as np
import pandas as pd
from PIL import Image
from utils import get_test_seed


def test_image_embedding():
    from dqm_ml_core import ProcessorRunner
    from dqm_ml_pytorch import ImageEmbeddingProcessor

    rng = np.random.default_rng(get_test_seed())
    images = []
    for _ in range(4):
        img = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())

    df = pd.DataFrame({"image_bytes": images})
    processor = ImageEmbeddingProcessor(
        name="test",
        config={
            "columns": {"input": ["image_bytes"]},
            "model": {"arch": "resnet18", "n_layer_feature": -2, "device": "cpu"},
            "infer": {"batch_size": 2},
        },
    )
    runner = ProcessorRunner()
    result = runner.run(df, [processor])
    assert "image_bytes_embedding" in result
    embeddings = result["image_bytes_embedding"]
    assert len(embeddings) == 4
    assert len(embeddings[0]) == 512


if __name__ == "__main__":
    try:
        test_image_embedding()
        print("dqm-ml-pytorch embeddings smoke test PASSED")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
