"""Generate synthetic parquet data for DQM-ML examples.

Produces two datasets:

1. ``large_test_2m.parquet``  (2M rows)
   Tabular metadata with quality scores, class labels, and NaN values.

2. ``samples_with_images.parquet``  (1200 rows)
    Small synthetic images (32x32 PNG bytes) with metadata for visual-features,
   embeddings, and domain-gap examples.

Usage::

    python examples/script/generate_data.py

Run once before using any config in ``examples/config/scenario/``.
"""

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
_SEED = 42

# ── Tabular dataset ──────────────────────────────────────────────────

_NROWS = 2_000_000


def _generate_tabular_dataset() -> None:
    path = _OUTPUT_DIR / "large_test_2m.parquet"
    rng = np.random.default_rng(_SEED)

    # imbalanced class distribution
    class_names = (
        ["bird"] * 4  # 40 %
        + ["cat", "dog", "horse"] * 1  # 30 %
        + ["cow", "sheep", "pig", "chicken", "fish", "goat"]  # 30 %
    )
    class_name = rng.choice(class_names, size=_NROWS)

    # imbalanced sample-type split
    sample_type = rng.choice(
        ["train", "test", "val"],
        size=_NROWS,
        p=[0.70, 0.20, 0.10],
    )

    # Beta-based quality scores — clearly non-normal distributions
    def _beta(a: float, b: float, size: int) -> np.ndarray:
        return rng.beta(a, b, size=size).astype(np.float64)

    brightness = _beta(2.0, 2.0, _NROWS)  # symmetric mound, not quite normal
    blur_score = _beta(0.5, 0.5, _NROWS)  # U-shaped (bimodal)
    contrast = _beta(2.0, 5.0, _NROWS)  # right-skewed
    sharpness = _beta(0.8, 0.8, _NROWS)  # plateau / slightly U-shaped

    # quality_score: mixture of two betas
    half = _NROWS // 2
    qs_a = rng.beta(0.8, 0.8, size=half)
    qs_b = rng.beta(2.0, 5.0, size=_NROWS - half)
    quality_score = np.concatenate([qs_a, qs_b]).astype(np.float64)
    rng.shuffle(quality_score)

    data = {
        "sample_id": np.arange(_NROWS, dtype=np.int64),
        "class_id": np.array([class_names.index(c) for c in class_name], dtype=np.int64),
        "class_name": class_name,
        "image_size": rng.integers(256, 1024, size=_NROWS, dtype=np.int64),
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "sharpness": sharpness,
        "quality_score": quality_score,
        "file_name": np.array([f"img_{i:07d}.jpg" for i in range(_NROWS)]),
        "sample_type": sample_type,
        "transform_id": rng.integers(0, 10, size=_NROWS, dtype=np.int64),
    }

    # per-column NaN rates: 0 %, 2 %, 12 %, 4 %, 1.5 %
    # Use explicit mask so NaN becomes a true Arrow null (not float NaN)
    nan_spec = {
        "brightness": 0.00,
        "blur_score": 0.02,
        "contrast": 0.12,
        "sharpness": 0.04,
        "quality_score": 0.015,
    }
    for col, rate in nan_spec.items():
        arr = data[col].copy()
        if rate > 0:
            mask = rng.random(_NROWS) < rate
            data[col] = pa.array(arr, mask=mask)
        else:
            data[col] = pa.array(arr)

    pq.write_table(pa.table(data), path)
    print(f"Wrote {_NROWS:,} rows to {path}  ({path.stat().st_size / 1e6:.0f} MB)")


# ── Image dataset ────────────────────────────────────────────────────

_IMG_ROWS = 1200
_IMG_SIZE = 32

# Source-specific pixel distributions (more distinct for clearer signal)
_SOURCE_PARAMS = {
    "zoo": {"mean": 0.50, "std": 0.30, "bias": np.array([0.00, 0.00, 0.00])},
    "safari": {"mean": 0.48, "std": 0.32, "bias": np.array([-0.03, 0.05, -0.03])},
    "reserve": {"mean": 0.28, "std": 0.28, "bias": np.array([0.05, -0.02, -0.05])},
}

# Per-class channel boost for Wasserstein signal
_CLASS_BIAS = {
    "elephant": np.array([0.05, 0.05, 0.05]),
    "lion": np.array([0.10, 0.00, 0.00]),
    "giraffe": np.array([0.00, 0.03, 0.00]),
    "zebra": np.array([0.00, 0.00, 0.10]),
}

# Per-source null rates for quality_score (makes completeness interesting)
_SOURCE_NULL_RATES = {
    "zoo": 0.02,
    "safari": 0.08,
    "reserve": 0.15,
}


def _make_synthetic_image(rng: np.random.Generator, source: str, class_name: str) -> bytes:
    params = _SOURCE_PARAMS[source]
    arr = rng.normal(params["mean"], params["std"], size=(_IMG_SIZE, _IMG_SIZE, 3))
    arr = arr.astype(np.float32)
    arr += params["bias"]
    arr += _CLASS_BIAS[class_name]
    arr = np.clip(arr, 0, 1)
    arr_uint8 = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _generate_image_dataset() -> None:
    path = _OUTPUT_DIR / "samples_with_images.parquet"
    rng = np.random.default_rng(_SEED + 1)

    class_names = list(_CLASS_BIAS.keys())
    sources = list(_SOURCE_PARAMS.keys())

    sample_ids = np.arange(_IMG_ROWS, dtype=np.int64)
    chosen_classes = rng.choice(class_names, size=_IMG_ROWS)
    chosen_sources = rng.choice(sources, size=_IMG_ROWS)

    image_bytes = [_make_synthetic_image(rng, s, c) for s, c in zip(chosen_sources, chosen_classes, strict=True)]

    quality_score = rng.random(_IMG_ROWS).astype(np.float64)
    null_mask = np.zeros(_IMG_ROWS, dtype=bool)
    for source, rate in _SOURCE_NULL_RATES.items():
        null_mask |= (chosen_sources == source) & (rng.random(_IMG_ROWS) < rate)
    quality_score = pa.array(quality_score, mask=null_mask)

    data = {
        "sample_id": sample_ids,
        "class_id": np.array([class_names.index(c) for c in chosen_classes], dtype=np.int64),
        "class_name": chosen_classes,
        "source": chosen_sources,
        "quality_score": quality_score,
        "image_bytes": pa.array(image_bytes, type=pa.binary()),
    }

    pq.write_table(pa.table(data), path)
    print(f"Wrote {_IMG_ROWS:,} rows to {path}  ({path.stat().st_size / 1e3:.0f} KB)")


# ── Entry point ──────────────────────────────────────────────────────


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _generate_tabular_dataset()
    _generate_image_dataset()


if __name__ == "__main__":
    main()
