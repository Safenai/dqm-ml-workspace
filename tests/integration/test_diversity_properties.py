"""Property-based tests for the diversity metric processor.

Tests verify diversity indices (Gini-Simpson, Richness, Shannon, Simpson)
on controlled categorical compositions without hardcoded expected.yaml values.
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


def _write_div_parquet(path: Path, values: list[float]) -> None:
    """Write a parquet file with a single column for diversity tests.

    Args:
        path: Output path for the parquet file.
        values: List of float values to write as the "col" column.
    """
    pq.write_table(pa.table({"col": pa.array(values, type=pa.float64())}), path)


def _run_div_job(
    parquet_path: Path,
    case_name: str,
    output_path: Path,
    test_path: str,
) -> dict[str, float]:
    """Run a diversity processor job and extract metric values.

    Creates a temporary YAML config, executes the pipeline via CLI, and parses
    the output parquet to return diversity metric values.

    Args:
        parquet_path: Path to input parquet file with test data.
        case_name: Identifier for this test case (used in config/output names).
        output_path: Directory where output parquet will be written.
        test_path: Path to tests directory (for config file generation).

    Returns:
        Dictionary mapping metric names (simpson, gini, shannon, richness)
        to their computed values.
    """
    config_name = f"div_prop_{case_name}"
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
                    "batch_size": 10_000,
                },
            ],
        },
        "metrics": {
            "outputs": {"path": str(out_file)},
            "processors": [
                {
                    "name": "diversity",
                    "type": "diversity",
                    "columns": {"input": ["col"]},
                    "metrics": ["simpson", "gini", "shannon", "richness"],
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
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"

    results: dict[str, float] = {}
    for metric in ["simpson", "gini", "shannon", "richness"]:
        key = f"col_{metric}"
        results[metric] = float(df[key].iloc[0])
    return results


@pytest.mark.parametrize(
    ("case_name", "values", "expected"),
    [
        pytest.param(
            "all_same",
            [0.0] * 1000,
            {"gini_simpson": 0.0, "richness": 1},
            id="all_same",
        ),
        pytest.param(
            "two_balanced",
            [0.0] * 500 + [1.0] * 500,
            {"gini_simpson": 0.5, "richness": 2},
            id="two_balanced",
        ),
        pytest.param(
            "two_skewed",
            [0.0] * 900 + [1.0] * 100,
            {"gini_simpson": 0.18, "richness": 2},
            id="two_skewed",
        ),
        pytest.param(
            "many_balanced",
            list(map(float, np.random.default_rng(get_test_seed()).integers(0, 100, 1000))),
            {"gini_simpson": 0.95, "richness": 90},
            id="many_balanced",
        ),
        pytest.param(
            "many_skewed",
            [0.0] * 900 + list(map(float, np.random.default_rng(get_test_seed() + 1).integers(1, 11, 100))),
            {"gini_simpson": 0.19, "richness": 11},
            id="many_skewed",
        ),
    ],
)
def test_div_properties(
    case_name: str,
    values: list[float],
    expected: dict[str, Any],
    output_path: Path,
    test_path: str,
    behavioral_dir: Path,
) -> None:
    parquet_path = behavioral_dir / f"div_{case_name}.parquet"
    _write_div_parquet(parquet_path, values)

    results = _run_div_job(parquet_path, case_name, output_path, test_path)

    gs = results["simpson"]
    r = results["richness"]

    print(f"  simpson={gs:.4f}  gini={results['gini']:.4f}  shannon={results['shannon']:.4f}  richness={r}")

    assert gs == pytest.approx(expected["gini_simpson"], abs=0.05), (
        f"simpson={gs:.4f} != expected {expected['gini_simpson']:.4f}"
    )
    assert r >= expected["richness"] * 0.85, (
        f"richness={r} < {expected['richness'] * 0.85} (85% of expected {expected['richness']})"
    )
    assert r <= expected["richness"] * 1.15, (
        f"richness={r} > {expected['richness'] * 1.15} (115% of expected {expected['richness']})"
    )

    assert 0.0 <= results["shannon"] <= 10.0, f"shannon={results['shannon']} outside [0, 10]"

    assert 0.0 <= results["gini"] <= 1.0, f"gini={results['gini']} outside [0, 1]"
