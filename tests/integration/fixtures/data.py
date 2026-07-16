"""Data fixtures for DQM-ML tests.

This module provides fixtures for generating and loading test data.
"""

import io
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.utils.files import write_path_list_to_parquet
from tests.utils.plots import plot_histograms
from tests.utils.seeds import get_test_seed

OUTPUT_PLOTS = "outputs/plots"
OUTPUT_DATA = "outputs/data"

_rng = np.random.default_rng(get_test_seed())


def _get_test_path() -> str:
    """Get the tests directory path."""
    return str(Path(__file__).parent.parent.resolve()) + "/"


def _get_fiftyone():
    import fiftyone.zoo as foz

    return foz


def _generate_synthetic_domain_images(
    output_dir: Path,
    n_per_set: int = 250,
    img_size: int = 64,
    seed: int = get_test_seed(),
) -> tuple[list[Path], list[str], list[Path], list[str]]:
    """Generate synthetic images for domain gap testing.

    Source images use blue-toned color palettes, target images use
    red-toned palettes.  Each image contains a random geometric shape
    on a dark background, providing enough spatial structure for
    ResNet18 / InceptionV3 embeddings to be meaningfully differentiated.

    Returns:
        Tuple of (source_paths, source_classes, target_paths, target_classes).
    """
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    img_dir = output_dir / "synthetic_domain_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    class_names = [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ]

    source_paths: list[Path] = []
    source_classes: list[str] = []
    target_paths: list[Path] = []
    target_classes: list[str] = []

    for i in range(n_per_set):
        cx = int(rng.integers(5, img_size - 5))
        cy = int(rng.integers(5, img_size - 5))
        rx = int(rng.integers(3, img_size // 3))
        ry = int(rng.integers(3, img_size // 3))

        # Source: blue-toned ellipse on dark-blue background
        s_img = Image.new("RGB", (img_size, img_size), (5, 5, 15))
        s_draw = ImageDraw.Draw(s_img)
        sr = int(rng.integers(0, 60))
        sg = int(rng.integers(40, 140))
        sb = int(rng.integers(140, 255))
        s_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(sr, sg, sb))
        s_path = img_dir / f"source_{i:04d}.jpg"
        s_img.save(s_path, "JPEG", quality=70)
        source_paths.append(s_path)
        source_classes.append(class_names[i % len(class_names)])

        # Target: red-toned ellipse on dark-red background
        t_img = Image.new("RGB", (img_size, img_size), (15, 5, 5))
        t_draw = ImageDraw.Draw(t_img)
        tr = int(rng.integers(140, 255))
        tg = int(rng.integers(0, 80))
        tb = int(rng.integers(0, 60))
        t_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(tr, tg, tb))
        t_path = img_dir / f"target_{i:04d}.jpg"
        t_img.save(t_path, "JPEG", quality=70)
        target_paths.append(t_path)
        target_classes.append(class_names[i % len(class_names)])

    return source_paths, source_classes, target_paths, target_classes


@pytest.fixture(scope="session")
def coco_data(test_path: str) -> list[Path]:
    """Generate synthetic image dataset for domain gap tests.

    Creates structured synthetic images (colored geometric shapes)
    organised into source and target sets with different colour palettes,
    along with class and domain metadata in parquet files.

    Args:
        test_path: Path to the tests directory.

    Returns:
        List containing paths to source and target parquet files.
    """
    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    source_path = Path(gen_path) / "source_1000.parquet"
    target_path = Path(gen_path) / "target_1000.parquet"

    if source_path.exists() and target_path.exists():
        source_table = pq.read_table(source_path)
        if source_table.num_rows > 0 and "domain" in source_table.column_names:
            sample_path = source_table.column("image_path")[0].as_py()
            if Path(sample_path).exists():
                print("Parquet found, images available, no need to recreate")
                return [source_path, target_path]
        print("Parquet found but images missing or domain column missing, regenerating")

    source_paths, source_classes, target_paths, target_classes = _generate_synthetic_domain_images(gen_path)

    domain_rng = np.random.default_rng(get_test_seed())
    source_domains = ["indoor" if domain_rng.random() < 0.5 else "outdoor" for _ in source_paths]
    target_domains = ["indoor" if domain_rng.random() < 0.5 else "outdoor" for _ in target_paths]

    write_path_list_to_parquet(source_paths, source_path, source_classes, domain=source_domains)
    write_path_list_to_parquet(target_paths, target_path, target_classes, domain=target_domains)

    return [source_path, target_path]


@pytest.fixture(scope="session")
def coco_csv(coco_data: list[Path]) -> list[Path]:
    """Generate CSV versions of the source/target parquet files."""
    csv_paths = []
    for parquet_path in coco_data:
        csv_path = parquet_path.with_suffix(".csv")
        if not csv_path.exists():
            table = pq.read_table(parquet_path)
            table.to_pandas().to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
    return csv_paths


@pytest.fixture(scope="session")
def coco_data_real(test_path: str) -> list[Path]:
    """Generate real COCO-2017 dataset for benchmark domain gap tests.

    Downloads COCO-2017 dataset via fiftyone and creates source/target
    parquet files with class and domain metadata.

    Args:
        test_path: Path to the tests directory.

    Returns:
        List containing paths to source and target parquet files.
    """
    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    source_path = Path(gen_path) / "source_1000_real.parquet"
    target_path = Path(gen_path) / "target_1000_real.parquet"

    if source_path.exists() and target_path.exists():
        source_table = pq.read_table(source_path)
        if source_table.num_rows > 0 and "domain" in source_table.column_names:
            sample_path = source_table.column("image_path")[0].as_py()
            if Path(sample_path).exists():
                print("Real COCO parquet found, images available, no need to recreate")
                return [source_path, target_path]
        print("Real COCO parquet found but images missing or missing domain column, re-downloading")

    foz = _get_fiftyone()
    foz.download_zoo_dataset(
        "coco-2017",
        splits=["train"],
        classes=[
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
        ],
        max_samples=2000,
    )
    dataset_path = Path.home() / "fiftyone" / "coco-2017" / "train" / "data"
    annotations_path = Path.home() / "fiftyone" / "coco-2017" / "raw" / "instances_train2017.json"

    import json

    with Path(annotations_path).open() as f:
        coco_raw = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco_raw["categories"]}
    image_to_class = {}
    for ann in coco_raw["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id not in image_to_class:
            image_to_class[img_id] = categories[cat_id]

    files = sorted(Path(dataset_path).glob("*.jpg"))

    source_files = files[: len(files) // 2]
    target_files = files[len(files) // 2 :]

    source_classes = []
    source_paths = []
    for f in source_files:
        img_id = int(f.stem)
        class_name = image_to_class.get(img_id, "unknown")
        source_classes.append(class_name)
        source_paths.append(f)

    target_classes = []
    target_paths = []
    for f in target_files:
        img_id = int(f.stem)
        class_name = image_to_class.get(img_id, "unknown")
        target_classes.append(class_name)
        target_paths.append(f)

    domain_rng = np.random.default_rng(get_test_seed())
    source_domains = ["indoor" if domain_rng.random() < 0.5 else "outdoor" for _ in source_files]
    target_domains = ["indoor" if domain_rng.random() < 0.5 else "outdoor" for _ in target_files]

    write_path_list_to_parquet(source_paths, source_path, source_classes, domain=source_domains)
    write_path_list_to_parquet(target_paths, target_path, target_classes, domain=target_domains)

    return [source_path, target_path]


@pytest.fixture(scope="session")
def coco_data_500(coco_data_real: list[Path], output_path: Path) -> None:
    """Create 500-image parquet fixtures from the real COCO-1000 parquets.

    Args:
        coco_data_real: Fixture providing paths to source_1000_real.parquet
            and target_1000_real.parquet.
        output_path: Path to test output data directory.
    """
    source_500 = output_path / "source_500.parquet"
    target_500 = output_path / "target_500.parquet"

    if source_500.exists() and target_500.exists():
        return

    for src, dst in [(coco_data_real[0], source_500), (coco_data_real[1], target_500)]:
        table = pq.read_table(src).slice(0, 500)
        pq.write_table(table, dst)


@pytest.fixture(scope="session")
def completeness_data(test_path: str) -> None:
    """Generate synthetic completeness test data.

    Creates a parquet file with 1000 rows of 4 double columns,
    including ~5% null values in some columns to exercise the
    completeness metric.

    Args:
        test_path: Path to the tests directory.
    """
    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    path = Path(gen_path) / "completeness.parquet"

    if path.exists():
        return

    rng = np.random.default_rng(get_test_seed())
    n = 1000
    col1 = rng.random(n).astype(float)
    col3 = rng.random(n).astype(float)
    col6 = rng.random(n).astype(float)
    col9 = rng.random(n).astype(float)

    # Inject ~7% nulls via explicit mask so NaNs become true Arrow nulls
    mask1 = rng.random(n) < 0.07
    mask3 = rng.random(n) < 0.07
    mask6 = rng.random(n) < 0.07

    table = pa.table(
        {
            "column_1": pa.array(col1, mask=mask1),
            "column_3": pa.array(col3, mask=mask3),
            "column_6": pa.array(col6, mask=mask6),
            "column_9": pa.array(col9),
        }
    )
    pq.write_table(table, path)


@pytest.fixture(scope="session")
def domain_gap_bytes_data(test_path: str) -> None:
    """Generate synthetic bytes parquets for domain gap wasserstein_bytes tests.

    Creates source_bytes.parquet and target_bytes.parquet with 5 image
    byte-blobs each, generated from synthetic domain images.

    Args:
        test_path: Path to the tests directory.
    """
    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    src_path = Path(gen_path) / "source_bytes.parquet"
    tgt_path = Path(gen_path) / "target_bytes.parquet"

    if src_path.exists() and tgt_path.exists():
        return

    img_dir = gen_path / "domain_gap_bytes_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    src_paths, _, tgt_paths, _ = _generate_synthetic_domain_images(
        img_dir,
        n_per_set=5,
        img_size=64,
        seed=get_test_seed(),
    )

    pq.write_table(
        pa.table({"image_bytes": [p.read_bytes() for p in src_paths]}),
        src_path,
    )
    pq.write_table(
        pa.table({"image_bytes": [p.read_bytes() for p in tgt_paths]}),
        tgt_path,
    )


@pytest.fixture(scope="session")
def visual_features_data(test_path: str) -> None:
    """Generate synthetic visual features test data.

    Creates:
      - visual_features.parquet     (30 rows, image_bytes column)
      - visual_features_path.parquet (30 rows, image_path column)
      - img/features/*.jpg           (30 synthetic JPEGs)

    Args:
        test_path: Path to the tests directory.
    """
    from PIL import Image, ImageDraw

    gen_path = Path(test_path) / OUTPUT_DATA
    img_dir = gen_path / "img" / "features"
    Path.mkdir(img_dir, exist_ok=True, parents=True)
    bytes_path = gen_path / "visual_features.parquet"
    path_path = gen_path / "visual_features_path.parquet"

    if bytes_path.exists() and path_path.exists():
        return

    rng = np.random.default_rng(get_test_seed())
    class_names = ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "rabbit", "duck"]
    sources = ["studio", "zoo", "outdoor"]
    rows_bytes: list[dict[str, Any]] = []
    rows_path: list[dict[str, Any]] = []

    for i in range(30):
        img = Image.new("RGB", (100, 70), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        shape = rng.choice(["ellipse", "rectangle"])
        x1, y1 = int(rng.integers(0, 40)), int(rng.integers(0, 20))
        x2, y2 = int(rng.integers(60, 99)), int(rng.integers(50, 69))
        if shape == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=color)
        else:
            draw.rectangle([x1, y1, x2, y2], fill=color)
        fname = f"{i}-100x70.jpg"
        jpg_path = img_dir / fname
        img.save(jpg_path, "JPEG", quality=70)
        class_id = i % 10
        base: dict[str, Any] = {
            "sample_id": i,
            "class_id": class_id,
            "class_name": class_names[class_id],
            "source": sources[i % 3],
        }
        rows_bytes.append({**base, "image_bytes": jpg_path.read_bytes()})
        rows_path.append({**base, "image_path": str(jpg_path)})

    pq.write_table(pa.Table.from_pylist(rows_bytes), bytes_path)
    pq.write_table(pa.Table.from_pylist(rows_path), path_path)


@pytest.fixture(scope="session")
def uniform_dist(test_path: str) -> Any:
    """Generate uniform distribution test data.

    Creates parquet file with uniformly distributed data and generates
    histogram plots.

    Args:
        test_path: Path to the tests directory.

    Returns:
        None.
    """
    plot_path = Path(test_path) / OUTPUT_PLOTS
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / f"{OUTPUT_DATA}/uniform_distribution.parquet"

    data_1 = _rng.uniform(0, 0.05, 1000000)
    data_2 = _rng.uniform(0.1, 1, 1000000)
    data_3 = _rng.uniform(0.2, 2, 1000000)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / OUTPUT_DATA,
        "uniform_distribution.parquet",
    )


@pytest.fixture(scope="session")
def not_uniform_dist(test_path: str) -> Any:
    """Generate non-uniform distribution test data.

    Creates parquet file with non-uniformly distributed data and generates
    histogram plots.

    Args:
        test_path: Path to the tests directory.

    Returns:
        None.
    """
    plot_path = Path(test_path) / OUTPUT_PLOTS
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / f"{OUTPUT_DATA}/not_uniform_distribution.parquet"

    a = _rng.uniform(0, 0.05, 500000)
    b = _rng.uniform(0.05, 2, 500000)
    data_1 = np.concatenate((a, b), axis=None)
    a = _rng.uniform(0.1, 1, 200000)
    b = _rng.uniform(0.1, 3, 800000)
    data_2 = np.concatenate((a, b), axis=None)
    a = _rng.uniform(0.2, 2, 200000)
    b = _rng.uniform(0.2, 3, 600000)
    c = _rng.uniform(0.2, 2, 200000)
    data_3 = np.concatenate((a, b, c), axis=None)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / OUTPUT_DATA,
        "not_uniform_distribution.parquet",
    )


@pytest.fixture(scope="session")
def diversity_data(test_path: str) -> Any:
    """Generate synthetic diversity test data.

    Creates parquet file with integer categorical data drawn from
    uniform discrete distributions at different uniqueness levels
    for streaming pipeline tests.

    Args:
        test_path: Path to the tests directory.

    Returns:
        None.
    """
    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    path = Path(gen_path) / "diversity.parquet"

    if path.exists():
        return

    rng = np.random.default_rng(get_test_seed())
    n = 200
    pa_table = pa.table(
        {
            "column_2": rng.integers(0, 40, size=n).astype(float),
            "column_4": rng.integers(0, 20, size=n).astype(float),
            "column_6": rng.integers(0, 8, size=n).astype(float),
        }
    )
    pq.write_table(pa_table, path)


@pytest.fixture(scope="session")
def normal_dist(test_path: str) -> Any:
    """Generate normal distribution test data.

    Creates parquet file with normally distributed data and generates
    histogram plots.

    Args:
        test_path: Path to the tests directory.

    Returns:
        None.
    """
    plot_path = Path(test_path) / OUTPUT_PLOTS
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / f"{OUTPUT_DATA}/normal_distribution.parquet"

    data_1 = _rng.normal(0, 0.5, 1000000)
    data_2 = _rng.normal(0, 5, 1000000)
    data_3 = _rng.normal(0, 50, 1000000)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / OUTPUT_DATA,
        "normal_distribution.parquet",
    )


@pytest.fixture(scope="session")
def not_normal_dist(test_path: str) -> Any:
    """Generate non-normal distribution test data.

    Creates parquet file with non-normally distributed data (bimodal) and
    generates histogram plots.

    Args:
        test_path: Path to the tests directory.

    Returns:
        None.
    """
    plot_path = Path(test_path) / OUTPUT_PLOTS
    Path.mkdir(plot_path, exist_ok=True, parents=True)
    path = Path(test_path) / f"{OUTPUT_DATA}/not_normal_distribution.parquet"

    a = _rng.normal(0, 0.5, 500000)
    b = _rng.normal(5, 0.5, 500000)
    data_1 = np.concatenate((a, b), axis=None)
    a = _rng.normal(0, 5, 500000)
    b = _rng.normal(50, 5, 500000)
    data_2 = np.concatenate((a, b), axis=None)
    a = _rng.normal(0, 50, 500000)
    b = _rng.normal(500, 50, 500000)
    data_3 = np.concatenate((a, b), axis=None)
    pa_table = pa.table({"data_1": data_1, "data_2": data_2, "data_3": data_3})
    pq.write_table(pa_table, path)

    plot_histograms(
        plot_path,
        ["data_1", "data_2", "data_3"],
        Path(test_path) / OUTPUT_DATA,
        "not_normal_distribution.parquet",
    )


@pytest.fixture(scope="session")
def full_story_data(test_path: str) -> Path:
    """Generate synthetic image dataset for full-story pipeline integration test.

    Mirrors ``examples/script/generate_data.py:_generate_image_dataset``
    to produce 1200 rows of synthetic 32x32 PNG images with:
      - 3 sources (safari, reserve, zoo) with distinct pixel distributions
      - 4 classes (elephant, lion, giraffe, zebra) with per-class channel biases
      - quality_score with source-specific null rates (2/8/15 %)

    Returns:
        Path to the generated parquet file.
    """
    from PIL import Image

    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    parquet_path = gen_path / "full_story_data.parquet"

    if parquet_path.exists():
        return parquet_path

    n = 1200
    img_size = 32
    rng = np.random.default_rng(get_test_seed())

    class_names = ["elephant", "lion", "giraffe", "zebra"]
    sources = ["safari", "reserve", "zoo"]

    source_params = {
        "zoo": {"mean": 0.50, "std": 0.30, "bias": np.array([0.00, 0.00, 0.00])},
        "safari": {"mean": 0.48, "std": 0.32, "bias": np.array([-0.03, 0.05, -0.03])},
        "reserve": {"mean": 0.28, "std": 0.28, "bias": np.array([0.05, -0.02, -0.05])},
    }
    class_bias = {
        "elephant": np.array([0.05, 0.05, 0.05]),
        "lion": np.array([0.10, 0.00, 0.00]),
        "giraffe": np.array([0.00, 0.03, 0.00]),
        "zebra": np.array([0.00, 0.00, 0.10]),
    }
    source_null_rates = {"zoo": 0.02, "safari": 0.08, "reserve": 0.15}

    chosen_classes = rng.choice(class_names, size=n)
    chosen_sources = rng.choice(sources, size=n)

    image_bytes_list: list[bytes] = []
    for s, c in zip(chosen_sources, chosen_classes, strict=True):
        params = source_params[s]
        arr = rng.normal(params["mean"], params["std"], size=(img_size, img_size, 3))
        arr = arr.astype(np.float32)
        arr += params["bias"]
        arr += class_bias[c]
        arr = np.clip(arr, 0, 1)
        arr_uint8 = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr_uint8, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes_list.append(buf.getvalue())

    quality_score = rng.random(n).astype(np.float64)
    null_mask = np.zeros(n, dtype=bool)
    for source, rate in source_null_rates.items():
        null_mask |= (chosen_sources == source) & (rng.random(n) < rate)

    table = pa.table(
        {
            "sample_id": pa.array(np.arange(n, dtype=np.int64)),
            "class_id": pa.array(np.array([class_names.index(c) for c in chosen_classes], dtype=np.int64)),
            "class_name": pa.array(chosen_classes),
            "source": pa.array(chosen_sources),
            "quality_score": pa.array(quality_score, mask=null_mask),
            "image_bytes": pa.array(image_bytes_list, type=pa.binary()),
        }
    )
    pq.write_table(table, parquet_path)
    return parquet_path


@pytest.fixture(scope="session")
def batch_invariance_data(test_path: str) -> Path:
    """1000-row parquet for domain-gap batch-size invariance tests.

    Only contains columns needed by domain-gap processors:``source``,
    ``class_name`` and ``image_bytes``.  Images are generated in-memory
    (no on-disk files).

    Returns:
        Path to the generated parquet file.
    """
    from PIL import Image, ImageDraw

    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    parquet_path = gen_path / "batch_invariance_data.parquet"

    if parquet_path.exists():
        return parquet_path

    rng = np.random.default_rng(get_test_seed())
    n = 1000
    sources = ["safari", "reserve"]
    class_names = ["elephant", "lion"]

    chosen_sources = rng.choice(sources, size=n, p=[0.6, 0.4])
    chosen_classes = rng.choice(class_names, size=n, p=[0.5, 0.5])

    image_bytes_list: list[bytes] = []
    for _ in range(n):
        bg = tuple(int(rng.integers(0, 40)) for _ in range(3))
        fg = tuple(int(rng.integers(140, 255)) for _ in range(3))
        img = Image.new("RGB", (32, 32), bg)
        draw = ImageDraw.Draw(img)
        cx, cy = int(rng.integers(5, 27)), int(rng.integers(5, 27))
        rx, ry = int(rng.integers(3, 12)), int(rng.integers(3, 12))
        if rng.random() < 0.5:
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fg)
        else:
            draw.rectangle([cx - rx, cy - ry, cx + rx, cy + ry], fill=fg)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes_list.append(buf.getvalue())

    table = pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "source": pa.array(chosen_sources),
            "class_name": pa.array(chosen_classes),
            "image_bytes": pa.array(image_bytes_list, type=pa.binary()),
        }
    )
    pq.write_table(table, parquet_path)
    return parquet_path


@pytest.fixture(scope="session")
def large_tabular_data(test_path: str) -> Path:
    """1M-row parquet for completeness / diversity / representativeness tests.

    Contains numeric columns for completeness (with ~7% nulls in some),
    categorical columns for diversity, and float columns for
    representativeness.  No image bytes — no embedding loading needed.

    Returns:
        Path to the generated parquet file.
    """
    gen_path = Path(test_path) / OUTPUT_DATA
    Path.mkdir(gen_path, exist_ok=True, parents=True)
    parquet_path = gen_path / "large_tabular_data.parquet"

    if parquet_path.exists():
        return parquet_path

    rng = np.random.default_rng(get_test_seed())
    n = 1_000_000
    sources = ["safari", "reserve", "zoo"]
    class_names = ["elephant", "lion", "giraffe", "zebra"]

    chosen_sources = rng.choice(sources, size=n, p=[0.4, 0.35, 0.25])
    chosen_classes = rng.choice(class_names, size=n, p=[0.3, 0.3, 0.2, 0.2])

    # -- completeness columns (matches completeness_data convention) --
    col1 = rng.random(n).astype(float)
    col3 = rng.random(n).astype(float)
    col6 = rng.random(n).astype(float)
    col9 = rng.random(n).astype(float)
    mask1 = rng.random(n) < 0.07
    mask3 = rng.random(n) < 0.07
    mask6 = rng.random(n) < 0.07

    # -- representativeness synthetic VF feature columns (generic names) --
    vf_lum = rng.uniform(0, 1, n).astype(float)
    vf_contrast = rng.exponential(0.5, n).astype(float)
    vf_blur = rng.normal(0, 0.3, n).astype(float)
    vf_entropy = rng.uniform(0, 8, n).astype(float)

    table = pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "source": pa.array(chosen_sources),
            "class_name": pa.array(chosen_classes),
            "column_1": pa.array(col1, mask=mask1),
            "column_3": pa.array(col3, mask=mask3),
            "column_6": pa.array(col6, mask=mask6),
            "column_9": pa.array(col9),
            "vf_luminosity": pa.array(vf_lum),
            "vf_contrast": pa.array(vf_contrast),
            "vf_blur": pa.array(vf_blur),
            "vf_entropy": pa.array(vf_entropy),
        }
    )
    pq.write_table(table, parquet_path)
    return parquet_path
