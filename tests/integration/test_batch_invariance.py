"""Test that dataloader ``batch_size`` does not materially affect metric results.

Each metric is run with three batch-size configurations (small / medium /
single-batch) and the output is compared across runs.  Tolerances are
per-metric — see ``_TOL``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.utils.pipeline_configs import ProcessorSpec, build_pipeline_config, make_loader

from dqm_ml_job.cli import execute

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZES_LARGE: dict[str, int] = {
    "small": 100,
    "medium": 10_000,
    "single": 1_000_000,
}
BATCH_SIZES_DOMAIN_GAP: dict[str, int] = {
    "small": 10,
    "medium": 100,
    "single": 1_000,
}

_TOL = {
    "exact": {"atol": 0.0, "rtol": 0.0},
    "float": {"atol": 1e-10, "rtol": 0.0},
    "relaxed": {"atol": 0.05, "rtol": 0.0},
    "gap": {"atol": 1e-3, "rtol": 0.0},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_and_load(
    data_path: str,
    processors: list[ProcessorSpec],
    root_key_order: list[str],
    batch_size: int,
    output_dir: Path,
    config_name: str,
) -> pa.Table:
    """Build a pipeline config, execute it, and return the output table.

    Determines the output file from the *last* interface in
    ``root_key_order`` (the one that holds the answer).
    """
    loaders = [
        make_loader(data_path=data_path, batch_size=batch_size),
    ]
    config_path = build_pipeline_config(
        data_path=data_path,
        processors=processors,
        root_key_order=root_key_order,
        output_dir=str(output_dir),
        config_name=config_name,
        loaders=loaders,
    )
    execute(["-p", str(config_path)])

    interface = root_key_order[-1]
    return pq.read_table(output_dir / f"{config_name}_{interface}.parquet")


def _run_all_batch_sizes(
    data_path: str,
    processors: list[ProcessorSpec],
    root_key_order: list[str],
    output_dir: Path,
    label: str,
    batch_sizes: dict[str, int],
) -> dict[str, pa.Table]:
    """Run the pipeline at each batch size and return a dict of tables."""
    results: dict[str, pa.Table] = {}
    for name, bs in batch_sizes.items():
        results[name] = _run_and_load(
            data_path,
            processors,
            root_key_order,
            bs,
            output_dir,
            f"{label}_{name}",
        )
    return results


def _compare_results(
    results: dict[str, pa.Table],
    atol: float,
    rtol: float = 0.0,
    *,
    include_cols: set[str] | None = None,
) -> None:
    """Assert that small- and medium-batch results match the single-batch baseline.

    Only floating-point columns are compared.  If ``include_cols`` is given
    only those columns are checked.
    """
    single = results["single"]
    for label in ("small", "medium"):
        df = results[label]
        assert df.num_rows == single.num_rows, f"{label}: row count mismatch"
        for col in single.column_names:
            if include_cols is not None and col not in include_cols:
                continue
            typ = single.column(col).type
            if not pa.types.is_floating(typ):
                continue
            a = df.column(col).to_numpy()
            b = single.column(col).to_numpy()
            if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                max_diff = np.nanmax(np.abs(a - b))
                pytest.fail(f"{label}/{col}: max_diff={max_diff:.2e} > atol={atol}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchInvariance:
    """Batch-size invariance for each metric group."""

    # -- Completeness -------------------------------------------------------

    def test_completeness(self, large_tabular_data: Path, output_path: Path) -> None:
        processors = [
            ProcessorSpec(
                name="completeness",
                type="completeness",
                interface="metrics",
                config={
                    "columns": {
                        "input": [
                            "column_1",
                            "column_3",
                            "column_6",
                            "column_9",
                        ],
                    },
                    "include_per_column": True,
                    "include_overall": True,
                },
            ),
        ]
        results = _run_all_batch_sizes(
            str(large_tabular_data),
            processors,
            ["dataloaders", "metrics"],
            output_path,
            "completeness",
            BATCH_SIZES_LARGE,
        )
        _compare_results(results, **_TOL["exact"])

    # -- Diversity ----------------------------------------------------------

    def test_diversity(self, large_tabular_data: Path, output_path: Path) -> None:
        processors = [
            ProcessorSpec(
                name="diversity",
                type="diversity",
                interface="metrics",
                config={
                    "columns": {"input": ["source", "class_name"]},
                },
            ),
        ]
        results = _run_all_batch_sizes(
            str(large_tabular_data),
            processors,
            ["dataloaders", "metrics"],
            output_path,
            "diversity",
            BATCH_SIZES_LARGE,
        )
        _compare_results(results, **_TOL["exact"])

    # -- Representativeness -------------------------------------------------

    def test_representativeness(self, large_tabular_data: Path, output_path: Path) -> None:
        vf_cols = [
            "vf_luminosity",
            "vf_contrast",
            "vf_blur",
            "vf_entropy",
        ]
        processors = [
            ProcessorSpec(
                name="representativeness",
                type="representativeness",
                interface="metrics",
                config={
                    "columns": {"input": vf_cols},
                    "metrics": [
                        "chi-square",
                        "grte",
                        "shannon-entropy",
                        "kolmogorov-smirnov",
                    ],
                    "distribution": "normal",
                    "histogram": {"bins": 10},
                    "mean_std_estimation": "user_provided",
                    "distribution_params": [
                        {"column": "vf_luminosity", "mean": 0.5, "std": 0.289},
                        {"column": "vf_contrast", "mean": 0.5, "std": 0.5},
                        {"column": "vf_blur", "mean": 0.0, "std": 0.3},
                        {"column": "vf_entropy", "mean": 4.0, "std": 2.31},
                    ],
                },
            ),
        ]
        results = _run_all_batch_sizes(
            str(large_tabular_data),
            processors,
            ["dataloaders", "metrics"],
            output_path,
            "representativeness",
            BATCH_SIZES_LARGE,
        )
        # KS is stochastic (random subsampling per batch) — exclude from comparison.
        # Bin edges are deterministic via user_provided strategy.
        non_ks_cols = {c for c in results["single"].column_names if "kolmogorov-smirnov" not in c}
        _compare_results(results, include_cols=non_ks_cols, **_TOL["relaxed"])

    # -- Domain Gap (7 single-layer metrics) --------------------------------

    def test_domain_gap(self, batch_invariance_data: Path, output_path: Path) -> None:
        """Batch invariance for FID, MMD-*, KL-MVN-Diag, Wasserstein-1D, PAD.

        All 7 gap processors share the same features-embeddings processor
        and the same split dataloader (safari vs reserve).
        """
        features_proc = ProcessorSpec(
            name="embeddings",
            type="features_embeddings",
            interface="features",
            config={
                "columns": {"input": ["image_bytes"]},
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
        )

        emb = "image_bytes_embedding"
        gap_metrics: list[ProcessorSpec] = [
            ProcessorSpec(
                name="fid",
                type="domain_gap",
                interface="gap",
                config={
                    "columns": {"input": [emb]},
                    "distance": {"metric": "fid", "epsilon": 1e-6},
                },
            ),
            ProcessorSpec(
                name="mmd_rbf",
                type="domain_gap",
                interface="gap",
                config={
                    "columns": {"input": [emb]},
                    "distance": {
                        "metric": "mmd_rbf",
                        "kernel_params": {"gamma": 1.0},
                    },
                },
            ),
            ProcessorSpec(
                name="mmd_linear",
                type="domain_gap",
                interface="gap",
                config={
                    "columns": {"input": [emb]},
                    "distance": {"metric": "mmd_linear"},
                },
            ),
            ProcessorSpec(
                name="mmd_poly",
                type="domain_gap",
                interface="gap",
                config={
                    "columns": {"input": [emb]},
                    "distance": {"metric": "mmd_poly"},
                },
            ),
            ProcessorSpec(
                name="klmvn_diag",
                type="domain_gap",
                interface="gap",
                config={
                    "columns": {"input": [emb]},
                    "distance": {"metric": "klmvn_diag"},
                },
            ),
            ProcessorSpec(
                name="wasserstein",
                type="domain_gap",
                interface="gap",
                config={
                    "columns": {"input": [emb]},
                    "distance": {"metric": "wasserstein_1d"},
                },
            ),
            ProcessorSpec(
                name="pad",
                type="domain_gap",
                interface="gap",
                config={
                    "columns": {"input": [emb]},
                    "distance": {"metric": "pad"},
                },
            ),
        ]

        loaders = [
            make_loader(
                data_path=str(batch_invariance_data),
                batch_size=0,  # placeholder, replaced per-run
                split={"by": "source", "values": ["safari", "reserve"]},
            ),
        ]

        results: dict[str, pa.Table] = {}
        for name, bs in BATCH_SIZES_DOMAIN_GAP.items():
            loaders[0]["batch_size"] = bs
            config_path = build_pipeline_config(
                data_path=str(batch_invariance_data),
                processors=[features_proc, *gap_metrics],
                root_key_order=["dataloaders", "features", "gap"],
                output_dir=str(output_path),
                config_name=f"domain_gap_{name}",
                loaders=loaders,
            )
            execute(["-p", str(config_path)])
            results[name] = pq.read_table(output_path / f"domain_gap_{name}_gap.parquet")

        _compare_results(
            results,
            include_cols=set(results["single"].column_names) - {"mmd_poly"},
            **_TOL["gap"],
        )
        # MMD-Poly involves catastrophic cancellation with large kernel values
        # and degree-3 exponentiation, so it needs a wider tolerance.
        _compare_results(results, include_cols={"mmd_poly"}, atol=0.1, rtol=0.0)

    # -- Domain Gap — CMD (multi-layer) -------------------------------------

    def test_domain_gap_cmd(self, batch_invariance_data: Path, output_path: Path) -> None:
        """CMD uses multi-layer embeddings — separate pipeline config."""
        layers = [
            "maxpool",
            "layer1.1.relu_1",
            "layer2.1.relu_1",
            "layer3.1.relu_1",
            "layer4.1.relu_1",
        ]
        emb_cols = [f"image_bytes_emb_{layer.replace('.', '_')}" for layer in layers]

        features_proc = ProcessorSpec(
            name="embeddings",
            type="features_embeddings",
            interface="features",
            config={
                "columns": {"input": ["image_bytes"]},
                "model": {
                    "arch": "resnet18",
                    "n_layer_feature": layers,
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
        )

        gap_proc = ProcessorSpec(
            name="cmd_gap",
            type="domain_gap",
            interface="gap",
            config={
                "columns": {"input": emb_cols},
                "distance": {
                    "metric": "cmd",
                    "feature_weights": [1.0] * len(layers),
                    "k": 5,
                },
            },
        )

        loaders = [
            make_loader(
                data_path=str(batch_invariance_data),
                batch_size=0,
                split={"by": "source", "values": ["safari", "reserve"]},
            ),
        ]

        results: dict[str, pa.Table] = {}
        for name, bs in BATCH_SIZES_DOMAIN_GAP.items():
            loaders[0]["batch_size"] = bs
            config_path = build_pipeline_config(
                data_path=str(batch_invariance_data),
                processors=[features_proc, gap_proc],
                root_key_order=["dataloaders", "features", "gap"],
                output_dir=str(output_path),
                config_name=f"domain_gap_cmd_{name}",
                loaders=loaders,
            )
            execute(["-p", str(config_path)])
            results[name] = pq.read_table(output_path / f"domain_gap_cmd_{name}_gap.parquet")

        _compare_results(results, **_TOL["gap"])
