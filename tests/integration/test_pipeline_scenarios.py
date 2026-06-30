"""Integration tests for real data quality scenarios from docs/metrics/*.md.

Each scenario configures a realistic subset of processors to verify
specific data quality use cases end-to-end.
"""

from pathlib import Path
from timeit import default_timer as timer

from dqm_ml_job.cli import execute
import pyarrow.parquet as pq
import pytest
from tests.utils.pipeline_configs import (
    build_pipeline_config,
    make_loader,
    scenario_acquisition_drift,
    scenario_class_imbalance,
    scenario_feature_selection_assist,
    scenario_multi_source_vf_diversity,
    scenario_overview_chain,
    scenario_preprocessing_sanity,
    scenario_quality_gate,
    scenario_train_test_drift,
)

_ROOT_KEY_ORDERS = [
    ["dataloaders", "features", "metrics", "gap"],
    ["dataloaders", "metrics", "features", "gap"],
    ["dataloaders", "gap", "features", "metrics"],
    ["dataloaders", "metrics", "gap", "features"],
    ["gap", "metrics", "features", "dataloaders"],
]


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
def test_scenario_overview_chain(
    pipeline_data: Path,
    output_path: Path,
    root_key_order: list[str],
) -> None:
    """Full 8-step chain from examples/overview.md with varying root key orders."""
    processors, base_name = scenario_overview_chain(str(pipeline_data))
    config_name = f"{base_name}_" + "_".join(root_key_order)

    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=processors,
        root_key_order=root_key_order,
        output_dir=str(output_path),
        config_name=config_name,
        loaders=[
            make_loader(
                data_path=str(pipeline_data),
                batch_size=100,
                split={"by": "source", "values": ["studio", "outdoor", "zoo"]},
            )
        ],
    )

    start = timer()
    execute(["-p", str(config_path)])
    elapsed = timer() - start
    print(f"Execution time ({base_name}): {elapsed:.2f}s")

    features_path = output_path / f"{config_name}_features.parquet"
    metrics_path = output_path / f"{config_name}_metrics.parquet"
    gap_path = output_path / f"{config_name}_gap.parquet"

    assert features_path.exists(), f"Missing features: {features_path}"
    assert metrics_path.exists(), f"Missing metrics: {metrics_path}"
    assert gap_path.exists(), f"Missing gap: {gap_path}"

    features_table = pq.read_table(features_path)
    metrics_table = pq.read_table(metrics_path)
    gap_table = pq.read_table(gap_path)

    assert features_table.num_rows > 0
    assert metrics_table.num_rows > 0
    assert gap_table.num_rows > 0

    df_metrics = metrics_table.to_pandas()
    assert any(c.startswith("completeness_") for c in df_metrics.columns)
    assert any(c.endswith("_simpson") for c in df_metrics.columns)
    assert any("_chi-square_" in c for c in df_metrics.columns)

    df_gap = gap_table.to_pandas()
    assert "mmd_linear" in df_gap.columns
    assert df_gap["mmd_linear"].notna().any()

    df_features = features_table.to_pandas()
    assert "image_bytes_contrast" in df_features.columns


@pytest.mark.timeout(300)
def test_scenario_quality_gate(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Completeness as a quality gate — verify null columns score < 1.0."""
    processors, base_name = scenario_quality_gate(str(pipeline_data))
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=processors,
        root_key_order=["dataloaders", "metrics"],
        output_dir=str(output_path),
        config_name=base_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    table = pq.read_table(output_path / f"{base_name}_metrics.parquet")
    df = table.to_pandas()

    assert "completeness_overall" in df.columns
    overall = df["completeness_overall"].iloc[0]
    assert overall < 1.0, f"Expected completeness_overall < 1.0, got {overall}"
    assert overall > 0.8, f"Expected completeness_overall > 0.8, got {overall}"

    assert "completeness_contrast" in df.columns
    assert df["completeness_contrast"].iloc[0] < 1.0, "contrast has 10% nulls, expected < 1.0"
    assert "completeness_brightness" in df.columns
    assert df["completeness_brightness"].iloc[0] == pytest.approx(1.0), "brightness has no nulls, expected 1.0"


@pytest.mark.timeout(300)
def test_scenario_class_imbalance(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Diversity by class_name — verify indices in [0,1]."""
    processors, base_name = scenario_class_imbalance(str(pipeline_data))
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=processors,
        root_key_order=["dataloaders", "metrics"],
        output_dir=str(output_path),
        config_name=base_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    table = pq.read_table(output_path / f"{base_name}_metrics.parquet")
    df = table.to_pandas()

    diversity_cols = [c for c in df.columns if c.endswith("_simpson")]
    assert len(diversity_cols) > 0

    for col in diversity_cols:
        values = df[col].dropna()
        if len(values) > 0:
            assert values.between(0, 1).all(), f"{col} values outside [0, 1]"


@pytest.mark.timeout(300)
def test_scenario_preprocessing_sanity(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Representativeness on raw columns — verify scores in [0,1]."""
    processors, base_name = scenario_preprocessing_sanity(str(pipeline_data))
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=processors,
        root_key_order=["dataloaders", "metrics"],
        output_dir=str(output_path),
        config_name=base_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    table = pq.read_table(output_path / f"{base_name}_metrics.parquet")
    df = table.to_pandas()

    repr_cols = [c for c in df.columns if "_chi-square_p_value" in c]
    assert len(repr_cols) > 0

    for col in repr_cols:
        values = df[col].dropna()
        if len(values) > 0:
            assert values.between(0, 1).all(), f"{col} values outside [0, 1]"


@pytest.mark.timeout(600)
def test_scenario_train_test_drift(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Embeddings -> domain_gap(split by sample_type) — gap > 0 between train/test."""
    processors, base_name = scenario_train_test_drift(str(pipeline_data))
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=processors,
        root_key_order=["dataloaders", "features", "gap"],
        output_dir=str(output_path),
        config_name=base_name,
        loaders=[
            make_loader(
                data_path=str(pipeline_data),
                batch_size=100,
                split={"by": "sample_type", "values": ["train", "test", "val"]},
            )
        ],
    )
    execute(["-p", str(config_path)])

    table = pq.read_table(output_path / f"{base_name}_gap.parquet")
    df = table.to_pandas()

    assert "mmd_linear" in df.columns
    assert df["mmd_linear"].notna().any()
    assert df["mmd_linear"].iloc[0] > 0, "Expected positive gap between train/test"


@pytest.mark.timeout(600)
def test_scenario_acquisition_drift(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Embeddings -> domain_gap(split by source) — gap > 0 between sources."""
    processors, base_name = scenario_acquisition_drift(str(pipeline_data))
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=processors,
        root_key_order=["dataloaders", "features", "gap"],
        output_dir=str(output_path),
        config_name=base_name,
        loaders=[
            make_loader(
                data_path=str(pipeline_data),
                batch_size=100,
                split={"by": "source", "values": ["studio", "outdoor", "zoo"]},
            )
        ],
    )
    execute(["-p", str(config_path)])

    table = pq.read_table(output_path / f"{base_name}_gap.parquet")
    df = table.to_pandas()

    assert "mmd_linear" in df.columns
    assert "selection_source" in df.columns
    assert "selection_target" in df.columns
    assert df["mmd_linear"].notna().any()
    assert df["mmd_linear"].iloc[0] > 0, "Expected positive gap between different sources"


@pytest.mark.timeout(600)
def test_scenario_multi_source_vf_diversity(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Visual features -> diversity on VF columns."""
    processors, base_name = scenario_multi_source_vf_diversity(str(pipeline_data))
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=processors,
        root_key_order=["dataloaders", "features", "metrics"],
        output_dir=str(output_path),
        config_name=base_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    features_table = pq.read_table(output_path / f"{base_name}_features.parquet")
    assert "image_bytes_contrast" in features_table.column_names
    assert "image_bytes_blur" in features_table.column_names

    metrics_table = pq.read_table(output_path / f"{base_name}_metrics.parquet")
    df_metrics = metrics_table.to_pandas()
    diversity_cols = [c for c in df_metrics.columns if c.endswith("_simpson")]
    assert len(diversity_cols) > 0

    for col in diversity_cols:
        values = df_metrics[col].dropna()
        if len(values) > 0:
            assert values.between(0, 1).all(), f"{col} values outside [0, 1]"


@pytest.mark.timeout(300)
def test_scenario_feature_selection_assist(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Completeness + representativeness on the same columns."""
    processors, base_name = scenario_feature_selection_assist(str(pipeline_data))
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=processors,
        root_key_order=["dataloaders", "metrics"],
        output_dir=str(output_path),
        config_name=base_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    table = pq.read_table(output_path / f"{base_name}_metrics.parquet")
    df = table.to_pandas()

    assert any(c.startswith("completeness_") for c in df.columns)
    assert any("_chi-square_" in c for c in df.columns)
