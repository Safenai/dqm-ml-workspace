"""Integration test: generated feature columns auto-included with outputs.include.

Tests that when ``features.outputs.include`` specifies only source columns
(e.g. ``[sample_id]``), the feature columns produced by processors are
still automatically written to the output parquet.
"""

from pathlib import Path
import tempfile
from typing import Any

from dqm_ml_job.cli import execute
import pyarrow.parquet as pq
import pytest
import yaml


@pytest.fixture(scope="module")
def config_path(output_path: Path) -> str:
    """Write a temporary visual_features config with include=[sample_id]."""
    config: dict[str, Any] = {
        "compute": {"log_level": "debug", "seed": 42, "threads": 4},
        "dataloaders": {
            "loaders": [
                {
                    "name": "source_dataset",
                    "type": "parquet",
                    "path": str(output_path / "visual_features.parquet"),
                    "batch_size": 10000,
                }
            ]
        },
        "features": {
            "outputs": {
                "path": str(output_path / "metrics_visual_features_include_{}-{}.parquet"),
                "include": ["sample_id"],
            },
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
                }
            ],
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        cfg_path = f.name

    yield cfg_path

    Path(cfg_path).unlink()


def test_visual_features_auto_includes_generated_columns(
    config_path: str,
    output_path: Path,
    visual_features_data: Any,
) -> None:
    """Generated feature columns appear in output even when include omits them."""
    execute(["-p", config_path])

    output_file = output_path / "metrics_visual_features_include_source_dataset-0.parquet"
    table = pq.read_table(output_file)
    columns = table.column_names

    assert "sample_id" in columns
    for col in ["image_bytes_luminosity", "image_bytes_contrast", "image_bytes_blur", "image_bytes_entropy"]:
        assert col in columns, f"Generated feature column '{col}' missing from output"

    assert "class_id" not in columns
    assert "class_name" not in columns
    assert "source" not in columns

    assert table.num_rows == 30

    output_file.unlink()
