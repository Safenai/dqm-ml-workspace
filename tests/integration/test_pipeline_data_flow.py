"""Integration tests verifying cross-stage column wiring in the pipeline.

Tests data flow between features, metrics, and gap processors, including
mixed raw + feature column inputs in a single ``columns.input`` list.
"""

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from tests.utils.pipeline_configs import (
    _make_completeness,
    _make_diversity,
    _make_domain_gap,
    _make_domain_gap_split,
    _make_features_embeddings,
    _make_representativeness,
    _make_visual_features,
    build_pipeline_config,
    make_loader,
)

from dqm_ml_job.cli import execute


@pytest.mark.timeout(600)
def test_feature_to_gap_split(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Embeddings -> domain_gap(split by source)."""
    config_name = "flow_feature_to_gap_split"
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=[
            _make_features_embeddings(input_col="image_bytes"),
            _make_domain_gap_split(input_col="image_bytes_embedding", metric="mmd_linear"),
        ],
        root_key_order=["dataloaders", "features", "gap"],
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
    execute(["-p", str(config_path)])

    gap_table = pq.read_table(output_path / f"{config_name}_gap.parquet")
    df = gap_table.to_pandas()

    assert "mmd_linear" in df.columns, "mmd_linear column missing"
    assert "selection_source" in df.columns, "selection_source missing"
    assert "selection_target" in df.columns, "selection_target missing"
    assert df["mmd_linear"].notna().any(), "mmd_linear is all null"
    assert df["selection_source"].nunique() > 1, "Expected multiple selection sources"
    assert df["mmd_linear"].iloc[0] > 0, "Expected positive gap between different sources"


@pytest.mark.timeout(600)
def test_feature_to_gap_mixed_input(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Visual features -> diversity on raw + feature columns in columns.input."""
    config_name = "flow_feature_to_metric_mixed_input"
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=[
            _make_visual_features(input_col="image_bytes"),
            _make_diversity(
                columns=["class_name", "image_bytes_contrast", "image_bytes_blur"],
            ),
        ],
        root_key_order=["dataloaders", "features", "metrics"],
        output_dir=str(output_path),
        config_name=config_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    metrics_table = pq.read_table(output_path / f"{config_name}_metrics.parquet")
    assert metrics_table.num_rows > 0, "Metrics output is empty"

    df = metrics_table.to_pandas()
    metric_cols = [c for c in df.columns if "_simpson" in c or "_gini" in c]
    assert len(metric_cols) > 0, "No diversity metrics in output"

    for col in metric_cols:
        assert df[col].notna().any(), f"{col} is all null"


@pytest.mark.timeout(600)
def test_metric_on_feature_only(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Representativeness on feature-generated columns only."""
    config_name = "flow_metric_on_feature_only"
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=[
            _make_visual_features(input_col="image_bytes"),
            _make_representativeness(
                columns=["image_bytes_contrast", "image_bytes_luminosity"],
                distribution="uniform",
            ),
        ],
        root_key_order=["dataloaders", "features", "metrics"],
        output_dir=str(output_path),
        config_name=config_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    metrics_table = pq.read_table(output_path / f"{config_name}_metrics.parquet")
    df = metrics_table.to_pandas()
    metric_cols = [c for c in df.columns if "_chi-square_" in c]
    assert len(metric_cols) > 0, "No representativeness metrics in output"


@pytest.mark.timeout(600)
def test_metric_on_raw_only(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Completeness on raw loader columns only."""
    config_name = "flow_metric_on_raw_only"
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=[
            _make_completeness(columns=["blur_score", "contrast", "quality_score"]),
        ],
        root_key_order=["dataloaders", "metrics"],
        output_dir=str(output_path),
        config_name=config_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    metrics_table = pq.read_table(output_path / f"{config_name}_metrics.parquet")
    df = metrics_table.to_pandas()
    assert "completeness_overall" in df.columns, "completeness_overall missing"
    assert df["completeness_overall"].iloc[0] < 1.0, "Expected completeness < 1.0 due to nulls"


@pytest.mark.timeout(600)
def test_multi_metric(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Multiple metrics (completeness + diversity + representativeness) on same data."""
    config_name = "flow_multi_metric"
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=[
            _make_completeness(columns=["blur_score", "contrast", "quality_score"]),
            _make_diversity(columns=["class_name"]),
            _make_representativeness(columns=["brightness"], distribution="uniform"),
        ],
        root_key_order=["dataloaders", "features", "metrics"],
        output_dir=str(output_path),
        config_name=config_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    table = pq.read_table(output_path / f"{config_name}_metrics.parquet")
    assert table.num_rows > 0

    df = table.to_pandas()
    assert any(c.startswith("completeness_") for c in df.columns), "No completeness columns"
    assert any(c.endswith("_simpson") for c in df.columns), "No diversity columns"
    assert any("_chi-square_" in c for c in df.columns), "No representativeness columns"


@pytest.mark.timeout(600)
def test_multi_gap(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Multiple gap metrics (MMD-Linear + Wasserstein-1D) on same embeddings."""
    config_name = "flow_multi_gap"
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=[
            _make_features_embeddings(input_col="image_bytes"),
            _make_domain_gap(input_col="image_bytes_embedding", metric="mmd_linear"),
            _make_domain_gap(
                name="domain_gap_wasserstein",
                input_col="image_bytes_embedding",
                metric="wasserstein_1d",
            ),
        ],
        root_key_order=["dataloaders", "features", "gap"],
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
    execute(["-p", str(config_path)])

    table = pq.read_table(output_path / f"{config_name}_gap.parquet")
    df = table.to_pandas()

    assert "mmd_linear" in df.columns, "mmd_linear column missing"
    assert "wasserstein_1d" in df.columns, "wasserstein_1d column missing"
    assert df["mmd_linear"].notna().any(), "mmd_linear is all null"
    assert df["wasserstein_1d"].notna().any(), "wasserstein_1d is all null"


@pytest.mark.timeout(600)
def test_vf_to_diversity(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Visual features -> diversity on [class_name, image_bytes_luminosity]."""
    config_name = "flow_vf_to_diversity"
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=[
            _make_visual_features(input_col="image_bytes"),
            _make_diversity(columns=["class_name", "image_bytes_luminosity"]),
        ],
        root_key_order=["dataloaders", "features", "metrics"],
        output_dir=str(output_path),
        config_name=config_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    table = pq.read_table(output_path / f"{config_name}_metrics.parquet")
    df = table.to_pandas()
    metric_cols = [c for c in df.columns if "_simpson" in c or "_gini" in c]
    assert len(metric_cols) > 0, "No diversity metrics in output"


@pytest.mark.timeout(600)
def test_vf_to_representativeness(
    pipeline_data: Path,
    output_path: Path,
) -> None:
    """Visual features -> representativeness on [blur_score, image_bytes_luminosity] (mixed)."""
    config_name = "flow_vf_to_representativeness"
    config_path = build_pipeline_config(
        data_path=str(pipeline_data),
        processors=[
            _make_visual_features(input_col="image_bytes"),
            _make_representativeness(
                columns=["blur_score", "image_bytes_luminosity"],
                distribution="uniform",
            ),
        ],
        root_key_order=["dataloaders", "features", "metrics"],
        output_dir=str(output_path),
        config_name=config_name,
        loaders=[make_loader(data_path=str(pipeline_data), batch_size=100)],
    )
    execute(["-p", str(config_path)])

    table = pq.read_table(output_path / f"{config_name}_metrics.parquet")
    df = table.to_pandas()
    metric_cols = [c for c in df.columns if "_chi-square_" in c]
    assert len(metric_cols) > 0, "No representativeness metrics in output"
