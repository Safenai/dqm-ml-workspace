"""Stress tests for the visual features processor on edge-case images.

Generates ~20 diverse/extreme JPEG images (uniform, checkerboard, noise,
extreme JPEG artifacts, skewed aspect ratios, tiny images, gradients, etc.)
and verifies all features compute without error and produce valid ranges.
"""

import math
from pathlib import Path
from timeit import default_timer as timer
from typing import Any

from dqm_ml_job.cli import execute
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.fixtures.stress_images import generate_stress_images
import yaml


@pytest.fixture(scope="module")
def behavioral_dir(output_path: Path) -> Path:
    """Create and return a behavioral test artifacts directory.

    Args:
        output_path: Base output directory for test run.

    Returns:
        Path to the behavioral subdirectory (created if needed).
    """
    path = output_path / "behavioral"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_feature_ranges(df: Any, n: int) -> None:
    """Assert that all visual feature columns are within valid ranges."""
    for i in range(n):
        lum = float(df["image_bytes_luminosity"].iloc[i])
        con = float(df["image_bytes_contrast"].iloc[i])
        blur = float(df["image_bytes_blur"].iloc[i])
        ent = float(df["image_bytes_entropy"].iloc[i])

        assert not math.isnan(lum), f"Row {i}: luminosity is NaN"
        assert not math.isnan(con), f"Row {i}: contrast is NaN"
        assert not math.isnan(blur), f"Row {i}: blur is NaN"
        assert not math.isnan(ent), f"Row {i}: entropy is NaN"

        assert not math.isinf(lum), f"Row {i}: luminosity is inf"
        assert not math.isinf(con), f"Row {i}: contrast is inf"
        assert not math.isinf(blur), f"Row {i}: blur is inf"
        assert not math.isinf(ent), f"Row {i}: entropy is inf"

        assert 0.0 <= lum <= 1.0, f"Row {i}: luminosity={lum} outside [0, 1]"
        assert 0.0 <= con <= 10.0, f"Row {i}: contrast={con} outside [0, 10]"
        assert 0.0 <= blur <= 10.0, f"Row {i}: blur={blur} outside [0, 10]"
        assert 0.0 <= ent <= 8.0, f"Row {i}: entropy={ent} outside [0, 8]"


def test_visual_features_stress(
    output_path: Path,
    test_path: str,
    behavioral_dir: Path,
) -> None:
    """Stress test visual features processor on diverse edge-case images.

    Generates 20 synthetic images covering extreme cases (uniform, noise,
    artifacts, aspect ratios, tiny images, gradients) and verifies all
    features compute without error and produce valid output ranges.

    Args:
        output_path: Directory for output parquet files.
        test_path: Path to tests directory (for config generation).
        behavioral_dir: Directory for intermediate test artifacts.
    """
    images = generate_stress_images()
    n = len(images)
    print(f"  Generated {n} stress images")

    parquet_path = behavioral_dir / "vf_stress_data.parquet"
    pq.write_table(
        pa.table({"image_bytes": pa.array(images, type=pa.binary())}),
        parquet_path,
    )

    out_file = output_path / "metrics_vf_stress.parquet"
    config: dict[str, Any] = {
        "compute": {
            "log_level": "debug",
            "seed": 42,
            "progress_bar": True,
            "threads": 4,
        },
        "dataloaders": {
            "loaders": [
                {
                    "name": "source_dataset",
                    "type": "parquet",
                    "path": str(parquet_path),
                    "batch_size": 100,
                },
            ],
        },
        "features": {
            "outputs": {"path": str(out_file)},
            "processors": [
                {
                    "name": "visual_features",
                    "type": "image_features",
                    "columns": {"input": ["image_bytes"]},
                    "features": ["luminosity", "contrast", "blur", "entropy"],
                    "grayscale": True,
                    "normalize": True,
                    "histogram": {"bins": 256},
                    "laplacian_kernel": "3x3",
                },
            ],
        },
    }

    config_dir = Path(test_path) / "integration" / "fixtures" / "config" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "vf_stress.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    start = timer()
    execute(["-p", str(config_path)])
    elapsed = timer() - start
    print(f"  Execution time: {elapsed:.2f}s for {n} images")

    table = pq.read_table(out_file)
    df = table.to_pandas()
    assert len(df) == n, f"Expected {n} rows, got {len(df)}"

    _validate_feature_ranges(df, n)

    print(f"  \u2713 All {n} stress images passed (no NaN/inf, all values in range)")
