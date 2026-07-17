"""Property-based tests for the representativeness metric processor.

Tests verify GRTE ordering across controlled distribution families
without hardcoded expected.yaml values.  Each test case generates a
fresh 100k-row parquet, runs the representativeness processor, and
asserts qualitative ranges on GRTE output.
"""

from pathlib import Path
from timeit import default_timer as timer
from typing import Any

from dqm_ml_job.cli import execute
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.utils.seeds import get_test_seed
import yaml

_rng = np.random.default_rng(get_test_seed())


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


def _write_repr_parquet(
    path: Path,
    col_fns: dict[str, Any],
    n: int = 100_000,
) -> None:
    """Write a parquet file with generated column data for representativeness tests.

    Args:
        path: Output path for the parquet file.
        col_fns: Dictionary mapping column names to generator functions.
            Each function takes an integer n and returns an np.ndarray of length n.
        n: Number of rows to generate (default: 100,000).
    """
    data: dict[str, np.ndarray] = {}
    for name, fn in col_fns.items():
        data[name] = fn(n)
    pq.write_table(pa.table(data), path)


def _run_repr_job(
    parquet_path: Path,
    distribution: str,
    case_name: str,
    output_path: Path,
    test_path: str,
) -> dict[str, float]:
    """Run a representativeness processor job and extract GRTE values.

    Creates a temporary YAML config, executes the pipeline via CLI, and parses
    the output parquet to return GRTE values for each test column.

    Args:
        parquet_path: Path to input parquet file with test data.
        distribution: Distribution type for representativeness processor
            (e.g., "normal", "uniform", "not_normal").
        case_name: Identifier for this test case (used in config/output names).
        output_path: Directory where output parquet will be written.
        test_path: Path to tests directory (for config file generation).

    Returns:
        Dictionary mapping column names (data_1, data_2, data_3) to their
        computed GRTE values.
    """
    config_name = f"repr_prop_{case_name}"
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
                    "batch_size": 100_000,
                },
            ],
        },
        "metrics": {
            "outputs": {"path": str(out_file)},
            "processors": [
                {
                    "name": "representativeness",
                    "type": "representativeness",
                    "metrics": [
                        "chi-square",
                        "grte",
                        "shannon-entropy",
                        "kolmogorov-smirnov",
                    ],
                    "ks": {"sample_size": 100_000, "sample_divisor": 1},
                    "columns": {"input": ["data_1", "data_2", "data_3"]},
                    "histogram": {"bins": 10},
                    "distribution": distribution,
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
    print(f"[{case_name}] Execution time: {elapsed:.2f}s")

    table = pq.read_table(out_file)
    df = table.to_pandas()
    source = df[df["selection"] == "source_dataset"]
    assert len(source) == 1, f"Expected 1 source_dataset row, got {len(source)}"

    grte_values: dict[str, float] = {}
    for col_name in ["data_1", "data_2", "data_3"]:
        key = f"{col_name}_grte_grte_value"
        val = source[key].iloc[0]
        grte_values[col_name] = float(val)
    return grte_values


_N = 100_000

NORMAL_CASES: list[tuple[str, dict[str, Any], str, float | None, float | None]] = [
    (
        "pure_normal",
        {
            "data_1": lambda n: _rng.normal(0, 1, n),
            "data_2": lambda n: _rng.normal(0, 5, n),
            "data_3": lambda n: _rng.normal(0, 50, n),
        },
        "normal",
        0.7,
        None,
    ),
    (
        "close_normal",
        {
            "data_1": lambda n: _rng.normal(0, 1, n) + _rng.uniform(-1, 1, n),
            "data_2": lambda n: _rng.normal(0, 5, n) + _rng.uniform(-1, 1, n),
            "data_3": lambda n: _rng.normal(0, 50, n) + _rng.uniform(-1, 1, n),
        },
        "normal",
        0.3,
        None,
    ),
    (
        "far_normal",
        {
            "data_1": lambda n: _rng.exponential(1, n),
            "data_2": lambda n: _rng.exponential(2, n),
            "data_3": lambda n: _rng.uniform(0, 5, n),
        },
        "normal",
        None,
        None,
    ),
    (
        "uniform_data",
        {
            "data_1": lambda n: _rng.uniform(0, 0.05, n),
            "data_2": lambda n: _rng.uniform(0.1, 1, n),
            "data_3": lambda n: _rng.uniform(0.2, 2, n),
        },
        "normal",
        None,
        None,
    ),
]

UNIFORM_CASES: list[tuple[str, dict[str, Any], str, float | None, float | None]] = [
    (
        "pure_uniform",
        {
            "data_1": lambda n: _rng.uniform(0, 0.05, n),
            "data_2": lambda n: _rng.uniform(0.1, 1, n),
            "data_3": lambda n: _rng.uniform(0.2, 2, n),
        },
        "uniform",
        0.7,
        None,
    ),
    (
        "close_uniform",
        {
            "data_1": lambda n: _rng.uniform(0, 0.05, n) + _rng.normal(0, 0.002, n),
            "data_2": lambda n: _rng.uniform(0.1, 1, n) + _rng.normal(0, 0.02, n),
            "data_3": lambda n: _rng.uniform(0.2, 2, n) + _rng.normal(0, 0.04, n),
        },
        "uniform",
        0.3,
        None,
    ),
    (
        "far_uniform",
        {
            "data_1": lambda n: _rng.normal(0, 1, n),
            "data_2": lambda n: _rng.exponential(2, n),
            "data_3": lambda n: _rng.normal(0, 50, n),
        },
        "uniform",
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    ("case_name", "col_fns", "distribution", "grte_min", "grte_max"),
    NORMAL_CASES + UNIFORM_CASES,
)
def test_repr_grte_ranges(
    case_name: str,
    col_fns: dict[str, Any],
    distribution: str,
    grte_min: float | None,
    grte_max: float | None,
    output_path: Path,
    test_path: str,
    behavioral_dir: Path,
) -> None:
    parquet_path = behavioral_dir / f"repr_{case_name}.parquet"
    _write_repr_parquet(parquet_path, col_fns, n=_N)

    grte_values = _run_repr_job(parquet_path, distribution, case_name, output_path, test_path)

    for col_name, val in grte_values.items():
        print(f"  GRTE({col_name}) = {val:.4f}")
        if grte_min is not None:
            assert val >= grte_min, f"GRTE for {col_name} = {val:.4f} < min {grte_min} in case '{case_name}'"
        if grte_max is not None:
            assert val <= grte_max, f"GRTE for {col_name} = {val:.4f} > max {grte_max} in case '{case_name}'"


def test_repr_grte_ordering(
    output_path: Path,
    test_path: str,
    behavioral_dir: Path,
) -> None:
    """Assert pure_normal has the highest GRTE (best self-fit)."""
    cases = [
        ("pure_normal", NORMAL_CASES[0][1], "normal"),
        ("close_normal", NORMAL_CASES[1][1], "normal"),
        ("far_normal", NORMAL_CASES[2][1], "normal"),
        ("uniform_data", NORMAL_CASES[3][1], "normal"),
    ]

    means: list[float] = []
    names: list[str] = []
    for name, fns, dist in cases:
        path = behavioral_dir / f"repr_{name}.parquet"
        _write_repr_parquet(path, fns, n=_N)
        grte = _run_repr_job(path, dist, name, output_path, test_path)
        mean_val = float(np.mean(list(grte.values())))
        means.append(mean_val)
        names.append(name)
        print(f"  {name}: mean GRTE = {mean_val:.4f}")

    pure_mean = means[0]
    for name, val in zip(names[1:], means[1:], strict=True):
        assert pure_mean + 1e-4 >= val, f"pure_normal GRTE ({pure_mean:.6f}) < {name} GRTE ({val:.6f})"
