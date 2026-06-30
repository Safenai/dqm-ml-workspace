"""Integration test: full end-to-end pipeline matching examples/config/full_story.yaml.

Exercises all 8 processors (visual features, ResNet-18 embeddings, completeness,
diversity, representativeness, FID, MMD-RBF, Wasserstein domain gap) on synthetic
image data with 3 sources x 2 classes = 6 selections.
"""

from pathlib import Path
from timeit import default_timer as timer

import pyarrow.parquet as pq
import pytest
from tests.utils.pipeline_configs import build_pipeline_config, make_loader, scenario_full_story

from dqm_ml_job.cli import execute

_VF_COLS = [
    "image_bytes_luminosity",
    "image_bytes_contrast",
    "image_bytes_blur",
    "image_bytes_entropy",
]
_VF_PVAL_COLS = [f"{c}_chi-square_p_value" for c in _VF_COLS]
_VF_GRTE_COLS = [f"{c}_grte_grte_value" for c in _VF_COLS]
_VF_KS_COLS = [f"{c}_kolmogorov-smirnov_statistic" for c in _VF_COLS]


@pytest.mark.timeout(600)
def test_full_story(
    full_story_data: Path,
    output_path: Path,
) -> None:
    """Full end-to-end pipeline: features, metrics, and gap with all processors."""
    processors, base_name = scenario_full_story(str(full_story_data))

    config_path = build_pipeline_config(
        data_path=str(full_story_data),
        processors=processors,
        root_key_order=["dataloaders", "features", "metrics", "gap"],
        output_dir=str(output_path),
        config_name=base_name,
        loaders=[
            make_loader(
                data_path=str(full_story_data),
                name="safari_animals",
                batch_size=500,
                filters=[{"column": "source", "values": ["safari"]}],
                split={"by": "class_name", "values": ["elephant", "zebra"]},
            ),
            make_loader(
                data_path=str(full_story_data),
                name="reserve_animals",
                batch_size=500,
                filters=[{"column": "source", "values": ["reserve"]}],
                split={"by": "class_name", "values": ["elephant", "zebra"]},
            ),
            make_loader(
                data_path=str(full_story_data),
                name="zoo_animals",
                batch_size=500,
                filters=[{"column": "source", "values": ["zoo"]}],
                split={"by": "class_name", "values": ["elephant", "zebra"]},
            ),
        ],
    )

    start = timer()
    execute(["-p", str(config_path)])
    elapsed = timer() - start
    print(f"Execution time ({base_name}): {elapsed:.2f}s")

    features_path = output_path / f"{base_name}_features.parquet"
    metrics_path = output_path / f"{base_name}_metrics.parquet"
    gap_path = output_path / f"{base_name}_gap.parquet"

    # ── Features ──────────────────────────────────────────────────────
    assert features_path.exists(), f"Missing features output: {features_path}"
    features_table = pq.read_table(features_path)
    df_features = features_table.to_pandas()

    for col in _VF_COLS:
        assert col in df_features.columns, f"Missing VF column: {col}"
    assert "image_bytes_embedding" in df_features.columns, "Missing embedding column"
    assert "source" in df_features.columns
    assert "class_name" in df_features.columns
    assert features_table.num_rows > 0, "Features output is empty"

    # ── Metrics ────────────────────────────────────────────────────────
    assert metrics_path.exists(), f"Missing metrics output: {metrics_path}"
    metrics_table = pq.read_table(metrics_path)
    df_metrics = metrics_table.to_pandas()

    # Completeness
    assert "completeness_overall" in df_metrics.columns
    overall = df_metrics["completeness_overall"].iloc[0]
    assert overall < 1.0, f"Expected completeness_overall < 1.0, got {overall}"
    assert overall > 0.8, f"Expected completeness_overall > 0.8, got {overall}"
    for col in _VF_COLS + ["quality_score"]:
        ccol = f"completeness_{col}"
        assert ccol in df_metrics.columns, f"Missing completeness column: {ccol}"

    # Diversity
    for metric in ["simpson", "gini", "shannon", "richness"]:
        dcol = f"class_name_{metric}"
        assert dcol in df_metrics.columns, f"Missing diversity column: {dcol}"
        values = df_metrics[dcol].dropna()
        if len(values) > 0:
            assert values.between(0, 1).all(), f"{dcol} values outside [0, 1]"

    # Representativeness — NaN check (critical)
    for pcol in _VF_PVAL_COLS:
        assert pcol in df_metrics.columns, f"Missing representativeness column: {pcol}"
        vals = df_metrics[pcol].dropna()
        assert len(vals) > 0, f"{pcol} has no non-null values"
        assert all(0.0 <= v <= 1.0 for v in vals), f"{pcol} values outside [0, 1]"

    for gcol in _VF_GRTE_COLS:
        assert gcol in df_metrics.columns, f"Missing GRTE column: {gcol}"
        vals = df_metrics[gcol].dropna()
        assert len(vals) > 0, f"{gcol} has no non-null values"
        assert all(0.0 <= v <= 1.0 for v in vals), f"{gcol} values outside [0, 1]"

    for kcol in _VF_KS_COLS:
        assert kcol in df_metrics.columns, f"Missing KS column: {kcol}"
        vals = df_metrics[kcol].dropna()
        assert len(vals) > 0, f"{kcol} has no non-null values"
        assert all(0.0 <= v <= 1.0 for v in vals), f"{kcol} values outside [0, 1]"

    # Shannon-entropy
    for col in _VF_COLS:
        sc = f"{col}_shannon-entropy_entropy"
        assert sc in df_metrics.columns, f"Missing shannon-entropy column: {sc}"

    # ── Gap ────────────────────────────────────────────────────────────
    assert gap_path.exists(), f"Missing gap output: {gap_path}"
    gap_table = pq.read_table(gap_path)
    df_gap = gap_table.to_pandas()

    assert "selection_source" in df_gap.columns
    assert "selection_target" in df_gap.columns

    for gcol in ["fid", "mmd_rbf", "wasserstein_1d"]:
        assert gcol in df_gap.columns, f"Missing gap column: {gcol}"
        vals = df_gap[gcol].dropna()
        assert len(vals) > 0, f"{gcol} has no non-null values"
        assert (vals > 0).all(), f"{gcol} should be positive for different sources"

    assert gap_table.num_rows > 0, "Gap output is empty"
    print("All assertions passed.")
