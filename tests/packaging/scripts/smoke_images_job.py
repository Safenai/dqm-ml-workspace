#!/usr/bin/env python3
"""Smoke test: dqm-ml-core + dqm-ml-images + dqm-ml-job (visual features via YAML pipeline).

Verifies that visual features can be executed through dqm-ml-job CLI with a YAML config.
"""

import io
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


def test_visual_features_via_yaml():
    from dqm_ml_job.cli import execute

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        rng = np.random.default_rng(42)
        images = []
        for _ in range(10):
            img = Image.fromarray(rng.integers(0, 255, (50, 50, 3), dtype=np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(buf.getvalue())

        table = pa.table({"image_bytes": images, "label": [f"img_{i}" for i in range(10)]})
        parquet_path = tmpdir / "test_data.parquet"
        pq.write_table(table, parquet_path)

        yaml_content = f"""\
dataloaders:
  loaders:
    - name: test_data
      type: parquet
      path: {parquet_path}
      batch_size: 5

features:
  processors:
    - name: image_quality
      type: image_features
      columns:
        input: ["image_bytes"]
      features: ["luminosity", "contrast", "blur", "entropy"]
      grayscale: true
      normalize: true
      laplacian_kernel: "3x3"

  outputs:
    path: {tmpdir}/output_features.parquet
"""
        yaml_path = tmpdir / "config.yaml"
        yaml_path.write_text(yaml_content)
        execute(["-p", str(yaml_path)])

        output_file = tmpdir / "output_features.parquet"
        assert output_file.exists(), f"Output file not found: {output_file}"

        output_df = pq.read_table(output_file).to_pandas()
        expected_features = [
            "image_bytes_luminosity",
            "image_bytes_contrast",
            "image_bytes_blur",
            "image_bytes_entropy",
        ]
        for feat in expected_features:
            assert feat in output_df.columns, f"Missing {feat} in output"


if __name__ == "__main__":
    try:
        test_visual_features_via_yaml()
        print("dqm-ml-images + dqm-ml-job smoke test PASSED")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
