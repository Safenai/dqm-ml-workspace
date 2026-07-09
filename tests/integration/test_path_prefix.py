"""Integration tests for sample_path prefix injection.

Tests that the full pipeline correctly resolves relative image paths
using the sample_path prefix configuration.
"""

from pathlib import Path
import shlex

from dqm_ml_job.cli import execute
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture(scope="module")
def prefix_test_data(test_path: str, output_path: str) -> dict:
    """Generate synthetic images and a parquet with only relative filenames.

    Creates:
      - tests/outputs/data/img/prefix_test/*.jpg  (5 synthetic images)
      - tests/outputs/data/path_prefix_test.parquet (image_path = just filenames)

    Returns:
        dict with paths for config generation.
    """
    from PIL import Image, ImageDraw

    gen_path = Path(output_path)
    img_dir = gen_path / "img" / "prefix_test"
    img_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = gen_path / "path_prefix_test.parquet"

    if parquet_path.exists():
        return {
            "img_dir": str(img_dir),
            "parquet": str(parquet_path),
            "output": str(gen_path),
        }

    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for i in range(5):
        img = Image.new("RGB", (100, 70), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        x1, y1 = int(rng.integers(0, 40)), int(rng.integers(0, 20))
        x2, y2 = int(rng.integers(60, 99)), int(rng.integers(50, 69))
        if rng.choice(["ellipse", "rectangle"]) == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=color)
        else:
            draw.rectangle([x1, y1, x2, y2], fill=color)
        fname = f"{i}.jpg"
        img.save(str(img_dir / fname), "JPEG", quality=70)
        rows.append({"sample_id": i, "image_path": fname})

    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    return {
        "img_dir": str(img_dir),
        "parquet": str(parquet_path),
        "output": str(gen_path),
    }


def test_path_prefix_resolves_images(prefix_test_data: dict, test_path: str) -> None:
    """Verify sample_path prefix enables image loading from relative paths."""
    img_dir = prefix_test_data["img_dir"]
    parquet_path = prefix_test_data["parquet"]
    output_dir = prefix_test_data["output"]

    output_pattern = f"{output_dir}/metrics_path_prefix_test_" + "{}-{}.parquet"
    config = f"""---
compute:
  log_level: "warning"
  seed: 42
  threads: 4

dataloaders:
  loaders:
    - name: source_dataset
      type: parquet
      path: {parquet_path}
      sample_path:
        - column: image_path
          prefix: {img_dir}/

features:
  outputs:
    path: {output_pattern}
  processors:
    - name: visual_features
      type: image_features
      columns:
        input: [image_path]
      features: [luminosity]
      grayscale: true
      normalize: true
      histogram:
        bins: 256
      laplacian_kernel: 3x3
"""

    config_path = Path(test_path) / "integration" / "fixtures" / "config" / "generated" / "path_prefix_test.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config)

    execute(shlex.split(f"-p {config_path}"))

    output_file = Path(output_dir) / "metrics_path_prefix_test_source_dataset-0.parquet"
    assert output_file.exists(), f"Output parquet not found: {output_file}"

    table = pq.read_table(output_file)
    assert "image_path_luminosity" in table.column_names, f"luminosity column not in {table.column_names}"

    values = table.column("image_path_luminosity").to_pylist()
    assert len(values) == 5, f"Expected 5 rows, got {len(values)}"

    for i, v in enumerate(values):
        assert v is not None, f"Row {i} has None luminosity value"
        assert isinstance(v, float), f"Row {i} has non-float value: {type(v)}"
        assert 0.0 <= v <= 1.0, f"Row {i} has out-of-range luminosity: {v}"
