"""Property-based tests for the domain gap metric processor.

Tests verify monotonic gap ordering across controlled dissimilarity levels
using the full image pipeline (PIL → parquet → features_embeddings → domain_gap).
"""

from pathlib import Path
from timeit import default_timer as timer
from typing import Any

from dqm_ml_job.cli import execute
import numpy as np
from PIL import Image, ImageDraw
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


@pytest.fixture(scope="session")
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


_N_IMAGES = 50


def _generate_controlled_images(
    output_dir: Path,
    seed: int = 42,
    n: int = _N_IMAGES,
) -> dict[str, tuple[list[Path], list[Path]]]:
    """Generate controlled synthetic image pairs for domain gap testing.

    Creates 4 levels of increasing domain shift:
    1. same_domain: Identical source and target images
    2. colour_shift: Same geometry, shifted hue
    3. shape_shift: Same colour, circles vs squares
    4. noise_shift: Same geometry/colour, added noise to target

    All images are 64x64 JPEG with consistent coordinate seeds for
    reproducible geometric alignment across cases.

    Args:
        output_dir: Directory where generated images will be saved.
        seed: Random seed for reproducible generation (default: 42).
        n: Number of image pairs per case (default: 50).

    Returns:
        Dictionary mapping case names to tuples of (source_paths, target_paths).
        Each path list contains n Path objects pointing to generated JPEGs.
    """
    rng = np.random.default_rng(seed)
    img_size = 64
    img_dir = output_dir / "domain_gap_prop_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Generate base coordinates shared across all cases
    cx_list = rng.integers(5, img_size - 5, n)
    cy_list = rng.integers(5, img_size - 5, n)
    rx_list = rng.integers(3, img_size // 3, n)
    ry_list = rng.integers(3, img_size // 3, n)

    # Per-image jitter to ensure non-degenerate embedding variances
    pos_jitter = rng.integers(-1, 2, (n, 4))  # (d_cx, d_cy, d_rx, d_ry)
    bg_jitter = rng.integers(-2, 3, (n, 3))  # (d_r, d_g, d_b)
    fill_jitter = rng.integers(-5, 6, (n, 3))

    cases: dict[str, tuple[list[Path], list[Path]]] = {}

    # 1. same_domain: source and target are identical blue circles
    src_same: list[Path] = []
    tgt_same: list[Path] = []
    for i in range(n):
        cx = int(cx_list[i]) + int(pos_jitter[i, 0])
        cy = int(cy_list[i]) + int(pos_jitter[i, 1])
        rx = int(rx_list[i]) + int(pos_jitter[i, 2])
        ry = int(ry_list[i]) + int(pos_jitter[i, 3])
        bg = (5 + int(bg_jitter[i, 0]), 5 + int(bg_jitter[i, 1]), 15 + int(bg_jitter[i, 2]))
        fill = (30 + int(fill_jitter[i, 0]), 100 + int(fill_jitter[i, 1]), 200 + int(fill_jitter[i, 2]))
        img = Image.new("RGB", (img_size, img_size), bg)
        draw = ImageDraw.Draw(img)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill)
        sp = img_dir / f"same_src_{i:04d}.jpg"
        img.save(sp, "JPEG", quality=70)
        src_same.append(sp)

        tp = img_dir / f"same_tgt_{i:04d}.jpg"
        img.copy().save(tp, "JPEG", quality=70)
        tgt_same.append(tp)
    cases["same_domain"] = (src_same, tgt_same)

    # 2. colour_shift: same geometry, slight hue shift
    src_colour: list[Path] = []
    tgt_colour: list[Path] = []
    for i in range(n):
        cx = int(cx_list[i]) + int(pos_jitter[i, 0])
        cy = int(cy_list[i]) + int(pos_jitter[i, 1])
        rx = int(rx_list[i]) + int(pos_jitter[i, 2])
        ry = int(ry_list[i]) + int(pos_jitter[i, 3])
        bg = (5 + int(bg_jitter[i, 0]), 5 + int(bg_jitter[i, 1]), 15 + int(bg_jitter[i, 2]))

        s_img = Image.new("RGB", (img_size, img_size), bg)
        s_draw = ImageDraw.Draw(s_img)
        s_fill = (30 + int(fill_jitter[i, 0]), 100 + int(fill_jitter[i, 1]), 200 + int(fill_jitter[i, 2]))
        s_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=s_fill)
        sp = img_dir / f"colour_src_{i:04d}.jpg"
        s_img.save(sp, "JPEG", quality=70)
        src_colour.append(sp)

        t_img = Image.new("RGB", (img_size, img_size), bg)
        t_draw = ImageDraw.Draw(t_img)
        t_fill = (50 + int(fill_jitter[i, 0]), 120 + int(fill_jitter[i, 1]), 220 + int(fill_jitter[i, 2]))
        t_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=t_fill)
        tp = img_dir / f"colour_tgt_{i:04d}.jpg"
        t_img.save(tp, "JPEG", quality=70)
        tgt_colour.append(tp)
    cases["colour_shift"] = (src_colour, tgt_colour)

    # 3. shape_shift: same colour, circles → squares
    src_shape: list[Path] = []
    tgt_shape: list[Path] = []
    for i in range(n):
        cx = int(cx_list[i]) + int(pos_jitter[i, 0])
        cy = int(cy_list[i]) + int(pos_jitter[i, 1])
        rx = int(rx_list[i]) + int(pos_jitter[i, 2])
        ry = int(ry_list[i]) + int(pos_jitter[i, 3])
        bg = (5 + int(bg_jitter[i, 0]), 5 + int(bg_jitter[i, 1]), 15 + int(bg_jitter[i, 2]))

        s_img = Image.new("RGB", (img_size, img_size), bg)
        s_draw = ImageDraw.Draw(s_img)
        s_fill = (30 + int(fill_jitter[i, 0]), 100 + int(fill_jitter[i, 1]), 200 + int(fill_jitter[i, 2]))
        s_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=s_fill)
        sp = img_dir / f"shape_src_{i:04d}.jpg"
        s_img.save(sp, "JPEG", quality=70)
        src_shape.append(sp)

        t_img = Image.new("RGB", (img_size, img_size), bg)
        t_draw = ImageDraw.Draw(t_img)
        t_fill = (30 + int(fill_jitter[i, 0]), 100 + int(fill_jitter[i, 1]), 200 + int(fill_jitter[i, 2]))
        t_draw.rectangle([cx - rx, cy - ry, cx + rx, cy + ry], fill=t_fill)
        tp = img_dir / f"shape_tgt_{i:04d}.jpg"
        t_img.save(tp, "JPEG", quality=70)
        tgt_shape.append(tp)
    cases["shape_shift"] = (src_shape, tgt_shape)

    # 4. full_shift: different shapes and colours
    src_full: list[Path] = []
    tgt_full: list[Path] = []
    for i in range(n):
        cx = int(cx_list[i]) + int(pos_jitter[i, 0])
        cy = int(cy_list[i]) + int(pos_jitter[i, 1])
        rx = int(rx_list[i]) + int(pos_jitter[i, 2])
        ry = int(ry_list[i]) + int(pos_jitter[i, 3])

        s_bg = (5 + int(bg_jitter[i, 0]), 5 + int(bg_jitter[i, 1]), 15 + int(bg_jitter[i, 2]))
        s_img = Image.new("RGB", (img_size, img_size), s_bg)
        s_draw = ImageDraw.Draw(s_img)
        s_fill = (30 + int(fill_jitter[i, 0]), 100 + int(fill_jitter[i, 1]), 200 + int(fill_jitter[i, 2]))
        s_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=s_fill)
        sp = img_dir / f"full_src_{i:04d}.jpg"
        s_img.save(sp, "JPEG", quality=70)
        src_full.append(sp)

        t_bg = (15 + int(bg_jitter[i, 0]), 5 + int(bg_jitter[i, 1]), 5 + int(bg_jitter[i, 2]))
        t_img = Image.new("RGB", (img_size, img_size), t_bg)
        t_draw = ImageDraw.Draw(t_img)
        t_fill = (200 + int(fill_jitter[i, 0]), 50 + int(fill_jitter[i, 1]), 30 + int(fill_jitter[i, 2]))
        t_draw.rectangle([cx - rx, cy - ry, cx + rx, cy + ry], fill=t_fill)
        tp = img_dir / f"full_tgt_{i:04d}.jpg"
        t_img.save(tp, "JPEG", quality=70)
        tgt_full.append(tp)
    cases["full_shift"] = (src_full, tgt_full)

    return cases


def _write_domain_gap_parquet(
    paths: list[Path],
    save_path: Path,
) -> None:
    """Write image paths to a parquet file for domain gap testing.

    Args:
        paths: List of Path objects pointing to image files.
        save_path: Output path for the parquet file.
    """
    path_strs = [str(p) for p in paths]
    table = pa.table({"image_path": pa.array(path_strs)})
    pq.write_table(table, save_path)


class TestDomainGapProperties:
    """Group of property tests for domain gap using full image pipeline."""

    @staticmethod
    @pytest.fixture(scope="class")
    def case_parquets(
        behavioral_dir: Path,
    ) -> dict[str, tuple[Path, Path]]:
        """Generate parquet pairs for each case, cached per class."""
        images = _generate_controlled_images(behavioral_dir)
        parquets: dict[str, tuple[Path, Path]] = {}
        for case_name, (src_paths, tgt_paths) in images.items():
            src_pq = behavioral_dir / f"dg_{case_name}_source.parquet"
            tgt_pq = behavioral_dir / f"dg_{case_name}_target.parquet"
            if not src_pq.exists():
                _write_domain_gap_parquet(src_paths, src_pq)
            if not tgt_pq.exists():
                _write_domain_gap_parquet(tgt_paths, tgt_pq)
            parquets[case_name] = (src_pq, tgt_pq)
        return parquets

    def _run_domain_gap(
        self,
        source_parquet: Path,
        target_parquet: Path,
        metric: str,
        case_name: str,
        output_path: Path,
        test_path: str,
    ) -> float:
        """Run full domain gap pipeline (images -> embeddings -> distance) and return metric value.

        Creates a temporary YAML config with the complete pipeline:
        - Two dataloaders (source_dataset, target_dataset) with sample_path prefix
        - features_embeddings processor (ResNet18, layer -2)
        - domain_gap processor with specified distance metric

        Args:
            source_parquet: Parquet with source image paths.
            target_parquet: Parquet with target image paths.
            metric: Distance metric to compute (mmd_linear, wasserstein_1d, klmvn_diag).
            case_name: Identifier for this test case (used in config/output names).
            output_path: Directory where output parquet will be written.
            test_path: Path to tests directory (for config file generation).

        Returns:
            Computed domain gap metric value as float.
        """
        config_name = f"dg_prop_{case_name}_{metric}"
        out_file = output_path / f"metrics_{config_name}.parquet"

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
                        "path": str(source_parquet),
                        "batch_size": 50,
                        "sample_path": [{"column": "image_path"}],
                    },
                    {
                        "name": "target_dataset",
                        "type": "parquet",
                        "path": str(target_parquet),
                        "batch_size": 50,
                        "sample_path": [{"column": "image_path"}],
                    },
                ],
            },
            "features": {
                "processors": [
                    {
                        "name": "image_embedding",
                        "type": "features_embeddings",
                        "columns": {"input": ["image_path"]},
                        "model": {
                            "arch": "resnet18",
                            "n_layer_feature": -2,
                            "device": "cpu",
                        },
                        "infer": {
                            "batch_size": 10,
                            "width": 64,
                            "height": 64,
                            "norm_mean": [0.485, 0.456, 0.406],
                            "norm_std": [0.229, 0.224, 0.225],
                        },
                    },
                ],
            },
            "gap": {
                "outputs": {"path": str(out_file)},
                "processors": [
                    {
                        "name": "domain_gap",
                        "type": "domain_gap",
                        "columns": {"input": ["image_path_embedding"]},
                        "distance": {
                            "metric": metric,
                            "klmvn_var_eps": 0.01 if metric == "klmvn_diag" else 0.0,
                        },
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
        print(f"  [{case_name}/{metric}] {elapsed:.2f}s")

        table = pq.read_table(out_file)
        df = table.to_pandas()
        assert metric in df.columns, f"Column '{metric}' not found in output"
        val = float(df[metric].iloc[0])
        return val

    @pytest.mark.parametrize("metric", ["mmd_linear", "wasserstein_1d", "klmvn_diag"])
    def test_domain_gap_monotonic(
        self,
        metric: str,
        case_parquets: dict[str, tuple[Path, Path]],
        output_path: Path,
        test_path: str,
    ) -> None:
        cases = ["same_domain", "colour_shift", "shape_shift", "full_shift"]
        values: list[float] = []

        for case_name in cases:
            src_pq, tgt_pq = case_parquets[case_name]
            val = self._run_domain_gap(src_pq, tgt_pq, metric, case_name, output_path, test_path)
            values.append(val)
            print(f"    {case_name}: {metric} = {val:.6f}")

        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"Monotonicity violated for {metric}: "
                f"{cases[i]} ({values[i]:.6f}) > {cases[i + 1]} ({values[i + 1]:.6f})"
            )

        print(f"  ✓ {metric} monotonic: {dict(zip(cases, values, strict=True))}")
