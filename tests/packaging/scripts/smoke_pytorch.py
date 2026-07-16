#!/usr/bin/env python3
"""Smoke test: dqm-ml-core + dqm-ml-pytorch + dqm-ml-job.

Verifies embeddings + gap metrics via YAML config through dqm-ml-job CLI.
"""

import io
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def _generate_test_parquet(path: str, num_samples: int = 8, img_size: int = 32) -> None:
    rng = np.random.default_rng(42)
    images = []
    for _ in range(num_samples):
        img = Image.fromarray(rng.integers(0, 255, (img_size, img_size, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())

    table = pa.table(
        {
            "sample_id": np.arange(num_samples, dtype=np.int64),
            "class_name": rng.choice(["cat", "dog", "bird", "car"], num_samples),
            "image_bytes": images,
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_embeddings_yaml():
    from dqm_ml_job.cli import execute

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "images.parquet"
        output_path = Path(tmpdir) / "embeddings.parquet"
        _generate_test_parquet(str(input_path))

        config = {
            "dataloaders": {
                "loaders": [{"name": "images", "type": "parquet", "path": str(input_path), "batch_size": 4}]
            },
            "features": {
                "outputs": {"path": str(output_path)},
                "processors": [
                    {
                        "name": "img_embed",
                        "type": "features_embeddings",
                        "columns": {"input": ["image_bytes"]},
                        "model": {"arch": "resnet18", "n_layer_feature": -2, "device": "cpu"},
                        "infer": {"batch_size": 2, "width": 32, "height": 32},
                    }
                ],
            },
        }
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(yaml.dump(config))
        execute(["-p", str(config_path)])

        result = pq.read_table(output_path)
        assert "image_bytes_embedding" in result.column_names


def test_gap_yaml():
    from dqm_ml_job.cli import execute

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "images.parquet"
        features_output = Path(tmpdir) / "features.parquet"
        gap_output = Path(tmpdir) / "gap.parquet"
        _generate_test_parquet(str(input_path), num_samples=16)

        config = {
            "dataloaders": {
                "loaders": [
                    {
                        "name": "animals",
                        "type": "parquet",
                        "path": str(input_path),
                        "batch_size": 8,
                        "split": {"by": "class_name"},
                    }
                ]
            },
            "features": {
                "outputs": {"path": str(features_output)},
                "processors": [
                    {
                        "name": "embedding",
                        "type": "features_embeddings",
                        "columns": {"input": ["image_bytes"]},
                        "model": {"arch": "resnet18", "n_layer_feature": -2, "device": "cpu"},
                        "infer": {"batch_size": 2, "width": 32, "height": 32},
                    }
                ],
            },
            "gap": {
                "outputs": {"path": str(gap_output), "pairwise": True},
                "processors": [
                    {
                        "name": "fid_gap",
                        "type": "domain_gap",
                        "columns": {"input": ["image_bytes_embedding"]},
                        "distance": {"metric": "fid", "epsilon": 1e-6},
                    }
                ],
            },
        }
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(yaml.dump(config))
        execute(["-p", str(config_path)])

        result = pq.read_table(gap_output)
        assert "fid" in result.column_names
        assert result.column("fid")[0].as_py() >= 0


if __name__ == "__main__":
    try:
        test_embeddings_yaml()
        test_gap_yaml()
        print("dqm-ml-pytorch + dqm-ml-job smoke test PASSED")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
