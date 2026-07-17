"""Per-feature correctness tests for the visual features processor.

Each test generates known images (black, white, gray, checkerboard, etc.)
embedded as JPEG bytes in a parquet, runs the processor, and asserts correct
feature values.
"""

import io
from pathlib import Path
from timeit import default_timer as timer
from typing import Any

from dqm_ml_job.cli import execute
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.utils.seeds import get_test_seed
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


def _bytes_parquet(images: list[bytes], case_name: str, behavioral_dir: Path) -> Path:
    """Write image bytes to a parquet file for visual features testing.

    Args:
        images: List of JPEG-encoded image bytes.
        case_name: Identifier for this test case (used in filename).
        behavioral_dir: Directory where the parquet file will be written.

    Returns:
        Path to the created parquet file.
    """
    path = behavioral_dir / f"vf_data_{case_name}.parquet"
    pq.write_table(pa.table({"image_bytes": pa.array(images, type=pa.binary())}), path)
    return path


def _run_vf_job(
    parquet_path: Path,
    test_name: str,
    output_path: Path,
    test_path: str,
) -> dict[str, float]:
    """Run visual features processor job and extract feature values.

    Creates a temporary YAML config, executes the pipeline via CLI, and parses
    the output parquet to return visual feature values for each test image.

    Args:
        parquet_path: Path to input parquet file with image_bytes column.
        test_name: Identifier for this test case (used in config/output names).
        output_path: Directory where output parquet will be written.
        test_path: Path to tests directory (for config file generation).

    Returns:
        Dictionary mapping feature names (luminosity, contrast, blur, entropy)
        to their computed values for the first image.
    """
    config_name = f"vf_prop_{test_name}"
    out_file = output_path / f"metrics_{config_name}.parquet"

    config: dict[str, Any] = {
        "compute": {
            "log_level": "debug",
            "seed": get_test_seed(),
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
                    "normalize": False,
                    "histogram": {"bins": 256},
                    "laplacian_kernel": "3x3",
                },
            ],
        },
    }

    config_dir = Path(test_path) / "integration" / "fixtures" / "config" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config_name}.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    start = timer()
    execute(["-p", str(config_path)])
    elapsed = timer() - start
    print(f"  [{test_name}] {elapsed:.2f}s")

    table = pq.read_table(out_file)
    df = table.to_pandas()
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"

    return {
        "luminosity": float(df["image_bytes_luminosity"].iloc[0]),
        "contrast": float(df["image_bytes_contrast"].iloc[0]),
        "blur": float(df["image_bytes_blur"].iloc[0]),
        "entropy": float(df["image_bytes_entropy"].iloc[0]),
    }


def _jpg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    """Encode a PIL Image as JPEG bytes.

    Args:
        img: PIL Image to encode.
        quality: JPEG quality setting 1-100 (default: 95).

    Returns:
        JPEG-encoded bytes of the image.
    """
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _png_bytes(img: Image.Image) -> bytes:
    """Encode a PIL Image as PNG bytes (lossless).

    Args:
        img: PIL Image to encode.

    Returns:
        PNG-encoded bytes of the image.
    """
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_luminosity_black(output_path: Path, test_path: str, behavioral_dir: Path) -> None:
    """Test luminosity feature on pure black image (expected ~0.0)."""
    img = Image.new("RGB", (100, 70), (0, 0, 0))
    pq_path = _bytes_parquet([_jpg_bytes(img)], "luminosity_black", behavioral_dir)
    result = _run_vf_job(pq_path, "luminosity_black", output_path, test_path)
    assert result["luminosity"] == pytest.approx(0.0, abs=0.02), f"black luminosity = {result['luminosity']}"


def test_luminosity_white(output_path: Path, test_path: str, behavioral_dir: Path) -> None:
    """Test luminosity feature on pure white image (expected ~1.0)."""
    img = Image.new("RGB", (100, 70), (255, 255, 255))
    pq_path = _bytes_parquet([_jpg_bytes(img)], "luminosity_white", behavioral_dir)
    result = _run_vf_job(pq_path, "luminosity_white", output_path, test_path)
    assert result["luminosity"] == pytest.approx(1.0, abs=0.02), f"white luminosity = {result['luminosity']}"


def test_luminosity_gray(output_path: Path, test_path: str, behavioral_dir: Path) -> None:
    """Test luminosity feature on mid-gray image (expected ~0.5)."""
    img = Image.new("RGB", (100, 70), (128, 128, 128))
    pq_path = _bytes_parquet([_jpg_bytes(img)], "luminosity_gray", behavioral_dir)
    result = _run_vf_job(pq_path, "luminosity_gray", output_path, test_path)
    assert result["luminosity"] == pytest.approx(0.5, abs=0.02), f"gray luminosity = {result['luminosity']}"


def test_contrast_uniform(output_path: Path, test_path: str, behavioral_dir: Path) -> None:
    """Test contrast feature on uniform image (expected ~0.0)."""
    img = Image.new("RGB", (100, 70), (128, 128, 128))
    pq_path = _bytes_parquet([_jpg_bytes(img)], "contrast_uniform", behavioral_dir)
    result = _run_vf_job(pq_path, "contrast_uniform", output_path, test_path)
    assert result["contrast"] == pytest.approx(0.0, abs=0.02), f"uniform contrast = {result['contrast']}"


def test_contrast_checkerboard(output_path: Path, test_path: str, behavioral_dir: Path) -> None:
    """Test contrast feature on checkerboard pattern (expected > 0.4)."""
    img = Image.new("L", (100, 70), 0)
    pixels = img.load()
    for x in range(100):
        for y in range(70):
            if (x // 8 + y // 8) % 2 == 0:
                pixels[x, y] = 255
    rgb = img.convert("RGB")
    pq_path = _bytes_parquet([_jpg_bytes(rgb)], "contrast_checkerboard", behavioral_dir)
    result = _run_vf_job(pq_path, "contrast_checkerboard", output_path, test_path)
    assert result["contrast"] > 0.4, f"checkerboard contrast = {result['contrast']} <= 0.4"


def test_blur_sharp_vs_blurry(output_path: Path, test_path: str, behavioral_dir: Path) -> None:
    img = Image.new("RGB", (100, 70), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 90, 60], fill=(255, 255, 255))
    sharp_bytes = _jpg_bytes(img)

    blurry = img.filter(ImageFilter.GaussianBlur(radius=4))
    blurry_bytes = _jpg_bytes(blurry)

    pq_path_multi = behavioral_dir / "vf_data_blur_pair.parquet"
    pq.write_table(
        pa.table({"image_bytes": pa.array([sharp_bytes, blurry_bytes], type=pa.binary())}),
        pq_path_multi,
    )

    config_name = "vf_prop_blur_pair"
    out_file = output_path / "metrics_vf_prop_blur_pair.parquet"
    config: dict[str, Any] = {
        "compute": {"log_level": "debug", "seed": get_test_seed(), "progress_bar": True, "threads": 4},
        "dataloaders": {
            "loaders": [
                {
                    "name": "source_dataset",
                    "type": "parquet",
                    "path": str(pq_path_multi),
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
                    "normalize": False,
                    "histogram": {"bins": 256},
                    "laplacian_kernel": "3x3",
                },
            ],
        },
    }
    config_dir = Path(test_path) / "integration" / "fixtures" / "config" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    with (config_dir / f"{config_name}.yaml").open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    execute(["-p", str(config_dir / f"{config_name}.yaml")])
    table = pq.read_table(out_file)
    df = table.to_pandas()
    assert len(df) == 2, f"Expected 2 rows, got {len(df)}"

    blur_sharp = float(df["image_bytes_blur"].iloc[0])
    blur_blurry = float(df["image_bytes_blur"].iloc[1])
    print(f"  blur_sharp={blur_sharp:.4f}  blur_blurry={blur_blurry:.4f}")

    assert blur_sharp > blur_blurry, f"sharp blur ({blur_sharp}) <= blurry blur ({blur_blurry})"


def test_entropy_random(output_path: Path, test_path: str, behavioral_dir: Path) -> None:
    """Test entropy feature on random noise image (expected > 5.0)."""
    rng = np.random.default_rng(get_test_seed())
    arr = rng.integers(0, 256, (70, 100), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    pq_path = _bytes_parquet([_png_bytes(img)], "entropy_random", behavioral_dir)
    result = _run_vf_job(pq_path, "entropy_random", output_path, test_path)
    assert result["entropy"] > 5.0, f"random entropy = {result['entropy']} <= 5.0"


def test_entropy_uniform(output_path: Path, test_path: str, behavioral_dir: Path) -> None:
    """Test entropy feature on uniform image (expected ~0.0)."""
    img = Image.new("RGB", (100, 70), (128, 128, 128))
    pq_path = _bytes_parquet([_jpg_bytes(img)], "entropy_uniform", behavioral_dir)
    result = _run_vf_job(pq_path, "entropy_uniform", output_path, test_path)
    assert result["entropy"] == pytest.approx(0.0, abs=0.05), f"uniform entropy = {result['entropy']}"
