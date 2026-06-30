"""Integration tests verifying root-key ordering invariance.

The pipeline's topological sort should produce identical output
regardless of the order of top-level keys (dataloaders, features, metrics, gap).
"""

from pathlib import Path
from timeit import default_timer as timer

from dqm_ml_job.cli import execute
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.utils.pipeline_configs import (
    _make_completeness,
    _make_domain_gap,
    _make_features_embeddings,
    build_pipeline_config,
    make_loader,
)

_ROOT_KEY_ORDERS = [
    ["dataloaders", "features", "metrics", "gap"],
    ["dataloaders", "metrics", "features", "gap"],
    ["dataloaders", "gap", "features", "metrics"],
    ["dataloaders", "metrics", "gap", "features"],
    ["gap", "metrics", "features", "dataloaders"],
]


@pytest.fixture(scope="session")
def reference_pipeline_tables(pipeline_data: Path, output_path: Path) -> tuple[pa.Table, pa.Table]:
    """Run the reference ordering pipeline once and return (metrics_table, gap_table).

    Session-scoped so all parametrizations share the same reference,
    regardless of test execution order (``pytest-randomly``).
    """
    config_name = "ordering_" + "_".join(_ROOT_KEY_ORDERS[0])
    processors = [
        _make_features_embeddings(input_col="image_bytes"),
        _make_domain_gap(input_col="image_bytes_embedding", metric="mmd_linear"),
        _make_completeness(columns=["blur_score", "contrast", "quality_score"]),
    ]

    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=processors,
        root_key_order=_ROOT_KEY_ORDERS[0],
        output_dir=str(output_path),
        config_name=config_name,
        loaders=[
            make_loader(
                data_path=str(pipeline_data),
                name="outdoor_animals",
                batch_size=100,
                filters=[{"column": "source", "values": ["outdoor"]}],
            ),
            make_loader(
                data_path=str(pipeline_data),
                name="zoo_animals",
                batch_size=100,
                filters=[{"column": "source", "values": ["zoo"]}],
            ),
        ],
    )

    execute(["-p", str(config_path)])

    metrics_path = output_path / f"{config_name}_metrics.parquet"
    gap_path = output_path / f"{config_name}_gap.parquet"

    assert metrics_path.exists(), f"Missing reference metrics: {metrics_path}"
    assert gap_path.exists(), f"Missing reference gap: {gap_path}"

    metrics_table = pq.read_table(metrics_path)
    gap_table = pq.read_table(gap_path)

    assert metrics_table.num_rows > 0, "Reference metrics output is empty"
    assert gap_table.num_rows > 0, "Reference gap output is empty"

    return metrics_table, gap_table


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "root_key_order",
    _ROOT_KEY_ORDERS,
    ids=[
        "dataloaders_features_metrics_gap",
        "dataloaders_metrics_features_gap",
        "dataloaders_gap_features_metrics",
        "dataloaders_metrics_gap_features",
        "gap_metrics_features_dataloaders",
    ],
)
def test_pipeline_ordering_invariance(
    pipeline_data: Path,
    output_path: Path,
    root_key_order: list[str],
    reference_pipeline_tables: tuple[pa.Table, pa.Table],
) -> None:
    """Verify that all root key orderings produce identical gap + metrics output."""
    ref_metrics_table, ref_gap_table = reference_pipeline_tables
    config_name = "ordering_" + "_".join(root_key_order)
    processors = [
        _make_features_embeddings(input_col="image_bytes"),
        _make_domain_gap(input_col="image_bytes_embedding", metric="mmd_linear"),
        _make_completeness(columns=["blur_score", "contrast", "quality_score"]),
    ]

    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=processors,
        root_key_order=root_key_order,
        output_dir=str(output_path),
        config_name=config_name,
        loaders=[
            make_loader(
                data_path=str(pipeline_data),
                name="outdoor_animals",
                batch_size=100,
                filters=[{"column": "source", "values": ["outdoor"]}],
            ),
            make_loader(
                data_path=str(pipeline_data),
                name="zoo_animals",
                batch_size=100,
                filters=[{"column": "source", "values": ["zoo"]}],
            ),
        ],
    )

    start = timer()
    execute(["-p", str(config_path)])
    elapsed = timer() - start
    print(f"Execution time ({config_name}): {elapsed:.2f}s")

    metrics_path = output_path / f"{config_name}_metrics.parquet"
    gap_path = output_path / f"{config_name}_gap.parquet"

    assert metrics_path.exists(), f"Missing metrics output: {metrics_path}"
    assert gap_path.exists(), f"Missing gap output: {gap_path}"

    metrics_table = pq.read_table(metrics_path)
    gap_table = pq.read_table(gap_path)

    assert metrics_table.num_rows > 0, "Metrics output is empty"
    assert gap_table.num_rows > 0, "Gap output is empty"

    assert metrics_table == ref_metrics_table, f"Metrics output differs for ordering {root_key_order}"
    assert gap_table == ref_gap_table, f"Gap output differs for ordering {root_key_order}"
