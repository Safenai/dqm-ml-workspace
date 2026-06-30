"""Standalone full-pipeline debug script — no Jupyter, no caching."""

from pathlib import Path
import sys

from dqm_ml_images.visual_features import VisualFeaturesProcessor
from dqm_ml_pytorch.domain_gap import DomainGapProcessor
from dqm_ml_pytorch.image_embedding import ImageEmbeddingProcessor
import numpy as np
import pandas as pd
import pyarrow as pa

from dqm_ml_core import (
    CompletenessProcessor,
    DiversityProcessor,
    RepresentativenessProcessor,
)
from dqm_ml_job.dataloaders import ParquetDataLoader
from dqm_ml_job.job import DatasetJob
from dqm_ml_job.outputwriter import ParquetOutputWriter

# ── Monkey-patch RepresentativenessProcessor for debugging ──────────────────
_orig_compute_chi_square = RepresentativenessProcessor._compute_chi_square_metric
_orig_compute_expected = RepresentativenessProcessor._compute_expected_counts
_orig_init_edges = RepresentativenessProcessor._initialize_bin_edges


def _debug_initialize_bin_edges(self, sample_data, col):
    result = _orig_init_edges(self, sample_data, col)
    print(f"  [DEBUG] _initialize_bin_edges: col={col}, mean_std_estimation={self.mean_std_estimation}")
    print(f"    dist_params keys: {list(self.dist_params.keys())}")
    col_params = self.dist_params.get(col, {})
    print(f"    col_params for {col}: {col_params}")
    if col in self._bin_edges:
        edges = self._bin_edges[col]
        print(f"    edges length={len(edges)}: {[f'{e:.4f}' if np.isfinite(e) else str(e) for e in edges]}")
    if col in self._bin_params:
        params = self._bin_params[col]
        print(f"    bin_params: mean={params['mean']:.6f}, std={params['std']:.6f}")
    return result


def _debug_compute_expected(self, col, batch_metrics, total_count, edges):
    result = _orig_compute_expected(self, col, batch_metrics, total_count, edges)
    print(f"  [DEBUG] _compute_expected_counts: col={col}, total_count={total_count}")
    print(f"    exp_counts (first 5): {result[:5]}")
    print(f"    exp_counts sum: {result.sum()}, all > 0: {(result > 0).all()}, sum > 0: {(result > 0).sum()}")
    return result


def _debug_chi_square(self, obs_counts, exp_counts):
    print("  [DEBUG-CS] _compute_chi_square_metric called!")
    print(f"    obs_counts: {obs_counts}")
    print(f"    exp_counts: {exp_counts}")
    result = _orig_compute_chi_square(self, obs_counts, exp_counts)
    print(f"    result: {result}")
    return result


RepresentativenessProcessor._initialize_bin_edges = _debug_initialize_bin_edges
RepresentativenessProcessor._compute_expected_counts = _debug_compute_expected
RepresentativenessProcessor._compute_chi_square_metric = _debug_chi_square

# ── Config ──────────────────────────────────────────────────────────────────
data_file = "examples/data/samples_with_images.parquet"
if not Path(data_file).exists():
    print("Run `python script/generate_data.py` first")
    sys.exit(1)

# ── 1. Global feature stats (user_provided representativeness) ──────────────
print("=" * 72)
print("Computing global feature stats ...")
raw = pd.read_parquet(data_file)
table = pa.Table.from_pandas(raw[["image_bytes", "source"]])

vf_stats = VisualFeaturesProcessor(
    name="vf_stats",
    config={
        "columns": {"input": ["image_bytes"]},
        "features": ["luminosity", "contrast", "blur", "entropy"],
        "grayscale": True,
        "normalize": True,
        "histogram": {"bins": 256},
        "laplacian_kernel": "3x3",
    },
)

feat_cols = [
    "image_bytes_luminosity",
    "image_bytes_contrast",
    "image_bytes_blur",
    "image_bytes_entropy",
]
feat_values = {c: [] for c in feat_cols}
for batch in table.to_batches(max_chunksize=500):
    feats = vf_stats.compute_features(batch)
    for c in feat_values:
        feat_values[c].extend(feats[c].to_numpy())

dist_params = [
    {
        "column": c,
        "mean": float(np.mean(feat_values[c])),
        "std": float(np.std(feat_values[c], ddof=0)),
    }
    for c in feat_cols
]
print("Global feature stats (user_provided):")
for p in dist_params:
    print(f"  {p['column']:40s}  mean={p['mean']:.6f}  std={p['std']:.6f}")

# ── 2. DataLoaders ─────────────────────────────────────────────────────────
print("Creating dataloaders ...")
loader_safari = ParquetDataLoader(
    name="safari",
    config={
        "path": data_file,
        "batch_size": 500,
        "filters": [{"column": "source", "values": ["safari"]}],
        "split": {"by": "class_name", "values": ["elephant", "zebra"]},
    },
)

loader_reserve = ParquetDataLoader(
    name="reserve",
    config={
        "path": data_file,
        "batch_size": 500,
        "filters": [{"column": "source", "values": ["reserve"]}],
        "split": {"by": "class_name", "values": ["elephant", "zebra"]},
    },
)

loader_zoo = ParquetDataLoader(
    name="zoo",
    config={
        "path": data_file,
        "batch_size": 500,
        "filters": [{"column": "source", "values": ["zoo"]}],
        "split": {"by": "class_name", "values": ["elephant", "zebra"]},
    },
)

dataloaders = {
    "safari": loader_safari,
    "reserve": loader_reserve,
    "zoo": loader_zoo,
}

# ── 3. Processors ──────────────────────────────────────────────────────────
print("Creating processors ...")
vf = VisualFeaturesProcessor(
    name="visual_features",
    config={
        "columns": {"input": ["image_bytes"]},
        "features": ["luminosity", "contrast", "blur", "entropy"],
        "grayscale": True,
        "normalize": True,
        "histogram": {"bins": 256},
        "laplacian_kernel": "3x3",
    },
)

emb = ImageEmbeddingProcessor(
    name="embedding",
    config={
        "columns": {"input": ["image_bytes"]},
        "model": {"arch": "resnet18", "n_layer_feature": -2},
        "infer": {
            "batch_size": 32,
            "width": 64,
            "height": 64,
            "norm_mean": [0.485, 0.456, 0.406],
            "norm_std": [0.229, 0.224, 0.225],
        },
    },
)

comp = CompletenessProcessor(
    name="completeness",
    config={
        "columns": {
            "input": [
                "quality_score",
                "image_bytes_luminosity",
                "image_bytes_contrast",
                "image_bytes_blur",
                "image_bytes_entropy",
            ]
        },
        "include_per_column": True,
        "include_overall": True,
    },
)

div = DiversityProcessor(
    name="diversity",
    config={
        "columns": {"input": ["class_name"]},
        "metrics": ["simpson", "gini", "shannon", "richness"],
    },
)

rep = RepresentativenessProcessor(
    name="representativeness",
    config={
        "columns": {
            "input": [
                "image_bytes_luminosity",
                "image_bytes_contrast",
                "image_bytes_blur",
                "image_bytes_entropy",
            ]
        },
        "metrics": [
            "chi-square",
            "grte",
            "shannon-entropy",
            "kolmogorov-smirnov",
        ],
        "distribution": "normal",
        "mean_std_estimation": "user_provided",
        "distribution_params": dist_params,
        "histogram": {"bins": 5},
    },
)

fid = DomainGapProcessor(
    name="fid_gap",
    config={
        "columns": {"input": ["image_bytes_embedding"]},
        "distance": {"metric": "fid", "epsilon": 1e-6},
    },
)

mmd = DomainGapProcessor(
    name="mmd_rbf_gap",
    config={
        "columns": {"input": ["image_bytes_embedding"]},
        "distance": {
            "metric": "mmd_rbf",
            "kernel_params": {"gamma": 1.0},
        },
    },
)

wass = DomainGapProcessor(
    name="wasserstein_gap",
    config={
        "columns": {"input": ["image_bytes_embedding"]},
        "distance": {"metric": "wasserstein_1d"},
    },
)

processors = {
    "vf": vf,
    "emb": emb,
    "comp": comp,
    "div": div,
    "rep": rep,
    "fid": fid,
    "mmd": mmd,
    "wass": wass,
}

# ── 4. OutputWriters ───────────────────────────────────────────────────────
print("Creating output writers ...")
feat_writer = ParquetOutputWriter(
    name="features",
    config={
        "path_pattern": "examples/outputs/full_story_features.parquet",
        "columns": ["sample_id", "source", "class_name"],
    },
)

met_writer = ParquetOutputWriter(
    name="metrics",
    config={
        "path_pattern": "examples/outputs/full_story_metrics.parquet",
    },
)

gap_writer = ParquetOutputWriter(
    name="delta",
    config={
        "path_pattern": "examples/outputs/full_story_gap.parquet",
    },
)

# ── 5. DatasetJob ──────────────────────────────────────────────────────────
print("Assembling DatasetJob ...")
job = DatasetJob(
    dataloaders=dataloaders,
    metrics=processors,
    features_output=feat_writer,
    progress_bar=True,
    threads=4,
    compute_seed=42,
    compute_device="auto",
)

print("Running job ...")
metrics_dict, delta_table = job.run()

print("Writing outputs ...")
met_writer.write_metrics_dict(metrics_dict)

if delta_table is not None:
    delta_data = {col: delta_table.column(col) for col in delta_table.column_names}
    gap_writer.write_table("delta", delta_data)

# ── 6. Diagnostics — Representativeness chi-square ─────────────────────────
print("\n" + "=" * 72)
print("REPRESENTATIVENESS DIAGNOSTICS (FROM IN-MEMORY metrics_dict)")
print("=" * 72)

# Build table from in-memory metrics_dict (NOT from parquet — it's stale
# because write_metrics_dict uses accumulate mode when path has no {})
sel_names = sorted(metrics_dict.keys(), key=lambda s: (s.split("_")[0], s.split("_")[-1]))
metric_names = [k for k in metrics_dict[sel_names[0]] if not (k.startswith("__") and k.endswith("__"))]
rows = []
for sel in sel_names:
    row = {"selection": sel}
    for m in metric_names:
        row[m] = metrics_dict[sel][m]
    rows.append(row)
metrics = pd.DataFrame(rows)

# Show every chi-square column for every selection
chi2_cols = [c for c in metrics.columns if "chi-square" in c]
print("All chi-square columns:")
print(metrics[["selection", "count"] + chi2_cols].to_markdown(index=False))

# Build the p-value matrix
feat_names = ["luminosity", "contrast", "blur", "entropy"]
feat_labels = ["Luminosity", "Contrast", "Blur", "Entropy"]
sel_labels = [s.replace("animals_", "") for s in metrics["selection"]]

print("\np-value matrix:")
pval_rows = {}
nlog_rows = {}
interp_rows = {}
for feat, label in zip(feat_names, feat_labels, strict=True):
    pcol = f"image_bytes_{feat}_chi-square_p_value"
    icol = f"image_bytes_{feat}_chi-square_interpretation"
    raw = metrics[pcol].values
    interp = metrics[icol].values
    pval_rows[label] = raw
    nlog_rows[label] = -np.log10(raw.astype(float))
    interp_rows[feat] = interp

pval_df = pd.DataFrame(pval_rows, index=sel_labels).T
print(pval_df.to_markdown(floatfmt=".4g"))

print("\n-log10(p-value) matrix:")
nlog_df = pd.DataFrame(nlog_rows, index=sel_labels).T
print(nlog_df.to_markdown(floatfmt=".2f"))

print("\nInterpretation matrix:")
interp_df = pd.DataFrame(interp_rows, index=sel_labels).T
print(interp_df.to_markdown())

# Check for NaN or insufficient_bins
print("\n" + "=" * 72)
print("STATUS CHECK")
print("=" * 72)
has_nan = False
has_insufficient = False
for feat in feat_names:
    pcol = f"image_bytes_{feat}_chi-square_p_value"
    icol = f"image_bytes_{feat}_chi-square_interpretation"
    for _, row in metrics.iterrows():
        pval = row[pcol]
        interp = row[icol]
        sel = row["selection"]
        if pd.isna(pval):
            print(f"  ❌ {sel}/{feat}: p-value = NaN  (interpretation = {interp})")
            has_nan = True
        if interp == "insufficient_bins":
            print(f"  ❌ {sel}/{feat}: insufficient_bins")
            has_insufficient = True

if not has_nan and not has_insufficient:
    print("  ✅ All chi-square p-values are valid — no NaN, no insufficient_bins")
elif has_nan and not has_insufficient:
    print("  ⚠️  NaN(s) found but no insufficient_bins")
elif not has_nan and has_insufficient:
    print("  ⚠️  insufficient_bins found but no NaN")
else:
    print("  ❌ Both NaN and insufficient_bins found")

print("\nDone.")
