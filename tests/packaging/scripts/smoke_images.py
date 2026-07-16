#!/usr/bin/env python3
"""Smoke test: dqm-ml-core + dqm-ml-images (no job).

Verifies that visual features can be computed via the Python API.
"""

import io
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
from tests.utils.seeds import get_test_seed


def test_visual_features_runner():
    from dqm_ml_core import ProcessorRunner
    from dqm_ml_images import VisualFeaturesProcessor

    rng = np.random.default_rng(get_test_seed())
    images = []
    for _ in range(5):
        img = Image.fromarray(rng.integers(0, 255, (100, 100, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())

    df = pd.DataFrame({"image_bytes": images})
    processor = VisualFeaturesProcessor(
        name="test",
        config={
            "columns": {"input": ["image_bytes"]},
            "features": ["luminosity", "contrast", "blur", "entropy"],
            "grayscale": True,
            "normalize": True,
            "laplacian_kernel": "3x3",
        },
    )
    runner = ProcessorRunner()
    result = runner.run(df, [processor])
    assert "image_bytes_luminosity" in result
    assert len(result["image_bytes_luminosity"]) == 5


def test_visual_features_direct():
    from dqm_ml_images import VisualFeaturesProcessor

    rng = np.random.default_rng(get_test_seed())
    images = []
    for _ in range(3):
        img = Image.fromarray(rng.integers(0, 255, (50, 50, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())

    batch = pa.record_batch([pa.array(images)], names=["image_bytes"])
    processor = VisualFeaturesProcessor(
        name="test",
        config={
            "columns": {"input": ["image_bytes"]},
            "features": ["luminosity", "contrast", "blur", "entropy"],
            "grayscale": True,
            "normalize": True,
            "laplacian_kernel": "3x3",
        },
    )
    features = processor.compute_features(batch, prev_features={})
    assert "image_bytes_luminosity" in features
    assert len(features["image_bytes_luminosity"]) == 3


if __name__ == "__main__":
    try:
        test_visual_features_runner()
        test_visual_features_direct()
        print("dqm-ml-images smoke test PASSED")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
