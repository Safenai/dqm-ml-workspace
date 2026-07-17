"""Combined pipeline data fixture for topological pipeline integration tests.

Provides a session-scoped fixture generating a single 500-row parquet
with all columns needed to exercise features, metrics, and gap processors
in a single pipeline config.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.utils.seeds import get_test_seed

OUTPUT_DATA = "outputs/data"
_CLASS_NAMES = ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "rabbit", "duck"]
_SOURCES = ["studio", "outdoor", "zoo"]
_SAMPLE_TYPES = ["train", "test", "val"]


def _generate_pipeline_images(img_dir: Path, n: int, rng: np.random.Generator) -> tuple[list[bytes], list[str]]:
    """Generate small RGB images with source-specific pixel distributions."""
    from PIL import Image, ImageDraw

    source_stats = {
        "studio": {"mean": 180, "std": 30},
        "outdoor": {"mean": 120, "std": 50},
        "zoo": {"mean": 100, "std": 40},
    }

    img_dir.mkdir(parents=True, exist_ok=True)
    bytes_list: list[bytes] = []
    path_list: list[str] = []

    for i in range(n):
        source = _SOURCES[rng.integers(0, 3)]
        stats = source_stats[source]
        size = 64
        img = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(img)

        bg = tuple(max(0, min(255, int(stats["mean"] + rng.normal(0, 10)))) for _ in range(3))
        draw.rectangle([0, 0, size - 1, size - 1], fill=bg)

        fg = tuple(max(0, min(255, int(stats["mean"] + rng.normal(0, stats["std"])))) for _ in range(3))
        x1, y1 = int(rng.integers(0, size // 2)), int(rng.integers(0, size // 2))
        x2, y2 = x1 + int(rng.integers(size // 4, size // 2)), y1 + int(rng.integers(size // 4, size // 2))
        if rng.random() < 0.5:
            draw.ellipse([x1, y1, x2, y2], fill=fg)
        else:
            draw.rectangle([x1, y1, x2, y2], fill=fg)

        jpg_path = img_dir / f"pipeline_{i:04d}.jpg"
        img.save(jpg_path, "JPEG", quality=70)
        bytes_list.append(jpg_path.read_bytes())
        path_list.append(str(jpg_path))

    return bytes_list, path_list


@pytest.fixture(scope="session")
def pipeline_data(test_path: str) -> Path:
    """Generate a single 500-row parquet with all pipeline columns.

    Columns:
      - sample_id, class_id, class_name, source, sample_type
      - image_bytes (64x64 JPEG bytes), image_path (path to on-disk JPEG)
      - blur_score, contrast, quality_score (with nulls), brightness, sharpness

    Returns:
        Path to the generated parquet file.
    """
    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    parquet_path = gen_path / "pipeline_data.parquet"

    if parquet_path.exists():
        return parquet_path

    img_dir = gen_path / "img" / "pipeline"
    rng = np.random.default_rng(get_test_seed())
    n = 500

    image_bytes_list, image_path_list = _generate_pipeline_images(img_dir, n, rng)

    blur_score = rng.uniform(0.0, 1.0, n).astype(np.float64)
    contrast = rng.uniform(0.0, 1.0, n).astype(np.float64)
    quality_score = rng.uniform(0.0, 1.0, n).astype(np.float64)
    brightness = rng.uniform(0.0, 1.0, n).astype(np.float64)
    sharpness = rng.uniform(0.0, 1.0, n).astype(np.float64)

    null_blur = rng.random(n) < 0.02
    null_contrast = rng.random(n) < 0.10
    null_quality = rng.random(n) < 0.01

    table = pa.table(
        {
            "sample_id": pa.array(np.arange(n, dtype=np.int64)),
            "class_id": pa.array(rng.integers(0, 10, size=n).astype(np.int64)),
            "class_name": pa.array([_CLASS_NAMES[rng.integers(0, 10)] for _ in range(n)]),
            "source": pa.array([_SOURCES[rng.integers(0, 3)] for _ in range(n)]),
            "sample_type": pa.array(rng.choice(_SAMPLE_TYPES, size=n, p=[0.7, 0.15, 0.15]).tolist()),
            "image_bytes": pa.array(image_bytes_list, type=pa.binary()),
            "image_path": pa.array(image_path_list, type=pa.string()),
            "blur_score": pa.array(blur_score, mask=null_blur),
            "contrast": pa.array(contrast, mask=null_contrast),
            "quality_score": pa.array(quality_score, mask=null_quality),
            "brightness": pa.array(brightness),
            "sharpness": pa.array(sharpness),
        }
    )
    pq.write_table(table, parquet_path)
    return parquet_path
