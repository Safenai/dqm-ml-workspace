#!/usr/bin/env python3
"""Smoke test: all packages (minus notebooks).

Verifies that metrics, visual features, embeddings, and gap metrics all work.
"""

import io
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


def test_all_metrics_via_api():
    from dqm_ml_core import CompletenessProcessor, ProcessorRunner
    from dqm_ml_images import VisualFeaturesProcessor
    from dqm_ml_pytorch import DomainGapProcessor, ImageEmbeddingProcessor

    runner = ProcessorRunner()

    # 1. Completeness
    df = pd.DataFrame({"col_a": [1, 2, None, 4, 5], "col_b": [1, 2, 3, None, 5]})
    result = runner.run(
        df,
        [
            CompletenessProcessor(
                name="test",
                config={"columns": {"input": ["col_a", "col_b"]}, "include_per_column": True, "include_overall": True},
            )
        ],
    )
    assert "completeness_col_a" in result

    # 2. Visual features
    rng = np.random.default_rng(42)
    images = []
    for _ in range(4):
        img = Image.fromarray(rng.integers(0, 255, (100, 100, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())
    df_img = pd.DataFrame({"image_bytes": images})
    result = runner.run(
        df_img,
        [
            VisualFeaturesProcessor(
                name="test",
                config={
                    "columns": {"input": ["image_bytes"]},
                    "features": ["luminosity", "contrast"],
                    "grayscale": True,
                    "normalize": True,
                    "laplacian_kernel": "3x3",
                },
            )
        ],
    )
    assert "image_bytes_luminosity" in result

    # 3. Embeddings
    images_big = []
    for _ in range(4):
        img = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images_big.append(buf.getvalue())
    df_emb = pd.DataFrame({"image_bytes": images_big})
    result = runner.run(
        df_emb,
        [
            ImageEmbeddingProcessor(
                name="test",
                config={
                    "columns": {"input": ["image_bytes"]},
                    "model": {"arch": "resnet18", "n_layer_feature": -2, "device": "cpu"},
                    "infer": {"batch_size": 2},
                },
            )
        ],
    )
    assert "image_bytes_embedding" in result

    # 4. Gap metrics (with pre-computed embeddings)
    source_emb = rng.standard_normal((50, 128)).astype(np.float32)
    target_emb = rng.standard_normal((50, 128)).astype(np.float32)
    source_df = pd.DataFrame({"embedding": list(source_emb)})
    target_df = pd.DataFrame({"embedding": list(target_emb)})
    result = runner.run_gap(
        source_df,
        target_df,
        DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["embedding"]}, "distance": {"metric": "mmd_linear"}},
        ),
    )
    assert "mmd_linear" in result


def test_all_via_yaml():
    from dqm_ml_job.cli import execute

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test data
        rng2 = np.random.default_rng(42)
        test_data = {
            "col_a": [1, 2, None, 4, 5, 6, 7, None, 9, 10],
            "col_b": [1, 2, 3, None, 5, 6, None, 8, 9, 10],
            "feature": rng2.normal(0, 1, 10).tolist(),
        }
        table = pa.table(test_data)
        parquet_path = tmpdir / "test_data.parquet"
        pq.write_table(table, parquet_path)

        yaml_content = f"""\
dataloaders:
  loaders:
    - name: test_data
      type: parquet
      path: {parquet_path}
      batch_size: 5

metrics:
  processors:
    - name: completeness
      type: completeness
      columns:
        input: ["col_a", "col_b"]
      include_per_column: true
      include_overall: true

  outputs:
    path: {tmpdir}/output_metrics.parquet
"""
        yaml_path = tmpdir / "config.yaml"
        yaml_path.write_text(yaml_content)
        execute(["-p", str(yaml_path)])

        output_file = tmpdir / "output_metrics.parquet"
        assert output_file.exists()
        output_df = pq.read_table(output_file).to_pandas()
        assert "completeness_col_a" in output_df.columns


if __name__ == "__main__":
    try:
        test_all_metrics_via_api()
        test_all_via_yaml()
        print("All-packages smoke test PASSED")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
