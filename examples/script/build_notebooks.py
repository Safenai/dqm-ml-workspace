"""Build script for the two scenario notebooks."""

import json

I = "    "  # one indent level

# ---- Constants for duplicated strings (SonarQube python:S1192) ----
CONFIG_MARKER = "# ---- Config ----"
RUN_CONFIG = "run(config)"
LUMINOSITY = '"image_bytes_luminosity",'
CONTRAST = '"image_bytes_contrast",'
BLUR = '"image_bytes_blur",'
ENTROPY = '"image_bytes_entropy",'
TIGHT_LAYOUT = "plt.tight_layout()"
PLT_SHOW = "plt.show()"
PRINT_DF = "print(df.to_markdown(index=False))"
COLORBAR = "plt.colorbar(im, shrink=0.8)"
CONFIG_OPEN = "config={"
COLUMNS_INPUT_IMAGE_BYTES = '"columns": {"input": ["image_bytes"]},'
PATH_DATA_FILE = '"path": data_file,'
BATCH_SIZE_500 = '"batch_size": 500,'
FILTERS_OPEN = '"filters": ['
SPLIT_OPEN = '"split": {'
BY_CLASS_NAME = '"by": "class_name",'
VALUES_ELEPHANT_ZEBRA = '"values": ["elephant", "zebra"],'
COLUMNS_INPUT_EMBEDDING = '"columns": {"input": ["image_bytes_embedding"]},'
VF_PARQUET = '"../outputs/story_visual_features.parquet"'


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text + "\n"]}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


def indent_join(lines_list, indent=I):
    """Join lines and add indent prefix to each."""
    return "\n".join(indent + l for l in lines_list)


def save(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Written {path}")


# ═══════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — Individual steps via run(config_dict)
# ═══════════════════════════════════════════════════════════════════

# cells_n1: step-by-step notebook — each metric runs independently with its own config
cells_n1 = []

cells_n1.append(
    md(
        "# Notebook \u2014 Individual Steps\n"
        "\n"
        "Runs each step from [overview.md](overview.md) individually using `dqm_ml_job.cli.run()` "
        "with inline Python dict configs. Each step: builds the config, calls `run()`, loads the "
        "output parquet, and adds a matplotlib visualization."
        "\n"
        "Prerequisites:"
        " - install dqm-ml in a virtual environment"
        "    with uv: `uv sync`"
        " - install notebooks dependencies"
        "    with uv: `uv pip install 'packages/dqm-ml[notebooks]'`"
        "\n"
        "Then select the virtual environment for jupyter"
        "\n"
        "## 0. Setup\n"
    )
)

cells_n1.append(
    code(
        "from dqm_ml_job.cli import run",
        "",
        "import yaml",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from pathlib import Path",
        "",
        "",
        "def _strip_examples(d):",
        I + '"""Recursively strip "examples/" prefix from path values.',
        I + "Notebook runs in examples/notebooks/, YAML configs assume repo root.",
        I + '"""',
        I + "if isinstance(d, dict):",
        I * 2 + "return {",
        I * 3 + "k: (_strip_examples(v) if k != 'path'",
        I * 3 + '     else "../" + v.removeprefix("examples/"))',
        I * 3 + "for k, v in d.items()",
        I * 2 + "}",
        I + "if isinstance(d, list):",
        I * 2 + "return [_strip_examples(i) for i in d]",
        I + "return d",
        "",
        "",
        "def _load_config(path):",
        I + '"""Load YAML config and adjust paths for notebook location."""',
        I + "cfg = yaml.safe_load(Path(path).read_text())",
        I + "return _strip_examples(cfg)",
        "",
        "",
        "# Verify data files exist",
        'data_dir = Path("../data")',
        'if not (data_dir / "samples_with_images.parquet").exists():',
        I + 'print("Run `python script/generate_data.py` first")',
    )
)

# ═══════════════════════════════════════════════════════════
# STEP 1: Visual Features
# ═══════════════════════════════════════════════════════════

cells_n1.append(
    md(
        "## 1. Visual Features\n"
        "\n"
        "Extracts per-image quality indicators (luminosity, contrast, blur, entropy) from synthetic 32\u00d732 images."
    )
)

cells_n1.append(
    code(
        CONFIG_MARKER,
        'config = _load_config("../config/scenario/visual_features.yaml")',
        RUN_CONFIG,
        "",
        'df = pd.read_parquet(' + VF_PARQUET + ')',
        'print("First 5 rows:")',
        "print(df.head(5).to_markdown(index=False))",
        'print("\\nAggregated by source:")',
        "feature_cols = [",
        I + LUMINOSITY,
        I + CONTRAST,
        I + BLUR,
        I + ENTROPY,
        "]",
        'agg = df.groupby("source")[feature_cols].mean()',
        "print(agg.to_markdown())",
    )
)

cells_n1.append(
    code(
        'df = pd.read_parquet(' + VF_PARQUET + ')',
        "feature_cols = [",
        I + LUMINOSITY,
        I + CONTRAST,
        I + BLUR,
        I + ENTROPY,
        "]",
        'agg = df.groupby("source")[feature_cols].mean()',
        "",
        "fig, ax = plt.subplots(figsize=(10, 4.5))",
        "x = np.arange(len(feature_cols))",
        "w = 0.25",
        'colors = ["#4C72B0", "#DD8452", "#55A868"]',
        'short_labels = ["Lum", "Contrast", "Blur", "Entropy"]',
        "for i, src in enumerate(agg.index):",
        I + "ax.bar(x + i * w, [agg.loc[src, c] for c in feature_cols], w,",
        I * 2 + "label=src, color=colors[i])",
        "ax.set_xticks(x + w)",
        "ax.set_xticklabels(short_labels)",
        'ax.set_ylabel("Mean value")',
        'ax.set_title("Visual Features by Acquisition Source")',
        "ax.legend()",
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

# ═══════════════════════════════════════════════════════
# STEP 2: Embeddings
# ═══════════════════════════════════════════════════════

cells_n1.append(md("## 2. Embeddings\n\nComputes 512-d ResNet-18 embedding vectors for every image."))

cells_n1.append(
    code(
        CONFIG_MARKER,
        'config = _load_config("../config/scenario/embeddings.yaml")',
        RUN_CONFIG,
        "",
        'df = pd.read_parquet("../outputs/story_embeddings.parquet")',
        "#print(df.head(5).to_markdown(index=False))",
    )
)

cells_n1.append(
    code(
        "from sklearn.decomposition import PCA",
        "",
        'df = pd.read_parquet("../outputs/story_embeddings.parquet")',
        "# Reduce 512-d ResNet-18 embeddings to 2 components for scatter-plot visualization",
        'emb = np.array(df["image_bytes_embedding"].to_list())',
        "",
        "pca = PCA(n_components=2, random_state=42)",
        "coords = pca.fit_transform(emb)",
        "vp = pca.explained_variance_ratio_",
        "",
        "fig, ax = plt.subplots(figsize=(8, 6))",
        'sources = df["source"].unique()',
        'colors = ["#4C72B0", "#DD8452", "#55A868"]',
        "for src, c in zip(sources, colors):",
        I + 'mask = df["source"] == src',
        I + "ax.scatter(coords[mask, 0], coords[mask, 1],",
        I * 2 + 'c=c, label=src, alpha=0.5, s=8, edgecolors="none")',
        I + 'ax.set_xlabel(f"PC1 ({vp[0]:.1%} var)")',
        I + 'ax.set_ylabel(f"PC2 ({vp[1]:.1%} var)")',
        I + 'ax.set_title("Embeddings \u2014 PCA Projection by Source")',
        I + "ax.legend(markerscale=4)",
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

# ═══════════════════════════════════════════════════════════
# STEP 3: Completeness
# ═══════════════════════════════════════════════════════════

cells_n1.append(
    md(
        "## 3. Completeness\n\nChecks every sample for missing values in the visual-feature columns (1200 rows, 4 features)."
    )
)

cells_n1.append(
    code(
        CONFIG_MARKER,
        'config = _load_config("../config/scenario/completeness.yaml")',
        RUN_CONFIG,
        "",
        'df = pd.read_parquet("../outputs/story_completeness.parquet")',
        PRINT_DF,
    )
)

cells_n1.append(
    code(
        'df = pd.read_parquet("../outputs/story_completeness.parquet")',
        "cols = [",
        I + '"completeness_image_bytes_luminosity",',
        I + '"completeness_image_bytes_contrast",',
        I + '"completeness_image_bytes_blur",',
        I + '"completeness_image_bytes_entropy",',
        "]",
        "vals = [df[col].iloc[0] for col in cols]",
        "",
        "fig, ax = plt.subplots(figsize=(8, 3.5))",
        'ax.barh(["image_bytes_luminosity", "image_bytes_contrast", "image_bytes_blur", "image_bytes_entropy"], vals,',
        I + 'color=["#4C72B0", "#DD8452", "#55A868", "#F1CE63"])',
        'ax.axvline(1.0, color="red", ls="--", lw=1, label="100 %")',
        "ax.set_xlim(0.8, 1.01)",
        'ax.set_xlabel("Completeness")',
        'ax.set_title("Per-Column Completeness")',
        "ax.legend()",
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

# ═══════════════════════════════════════════════════════
# STEP 4: Diversity
# ═══════════════════════════════════════════════════════

cells_n1.append(
    md(
        "## 4. Diversity\n"
        "\n"
        "Measures class balance and category spread using Simpson, Gini, Shannon, and Richness indices on the visual features dataset (output of step 3), grouped by class and source."
    )
)

cells_n1.append(
    code(
        CONFIG_MARKER,
        'config = _load_config("../config/scenario/diversity.yaml")',
        RUN_CONFIG,
        "",
        'df = pd.read_parquet("../outputs/story_diversity.parquet")',
        PRINT_DF,
    )
)

cells_n1.append(
    code(
        'raw = pd.read_parquet(' + VF_PARQUET + ')',
        "",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))",
        "",
        "# Left bar: class frequency across all sources",
        'class_counts = raw["class_name"].value_counts()',
        'ax1.bar(class_counts.index, class_counts.values, color="#4C72B0")',
        'ax1.set_title("Class Distribution")',
        'ax1.tick_params(axis="x", rotation=60)',
        'ax1.set_ylabel("Count")',
        "",
        "# Right pie: proportion of samples per source",
        'source_counts = raw["source"].value_counts()',
        'colors_pt = ["#4C72B0", "#55A868", "#DD8452"]',
        "ax2.pie(source_counts.values, labels=source_counts.index,",
        I + 'autopct="%1.0f%%", colors=colors_pt, startangle=90)',
        'ax2.set_title("Source Split")',
        "",
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

# ═══════════════════════════════════════════════════════
# STEP 5: Representativeness
# ═══════════════════════════════════════════════════════

cells_n1.append(
    md(
        "## 5. Representativeness\n"
        "\n"
        "Tests whether luminosity, contrast, blur, and entropy (from step 3) follow a normal distribution using chi-square, KS, entropy, and GRTE."
    )
)

cells_n1.append(
    code(
        CONFIG_MARKER,
        'config = _load_config("../config/scenario/representativeness.yaml")',
        RUN_CONFIG,
        "",
        'df = pd.read_parquet("../outputs/story_representativeness.parquet")',
        "",
        "# Compact summary table",
        "rows = []",
        'for col in ["image_bytes_luminosity", "image_bytes_contrast", "image_bytes_blur", "image_bytes_entropy"]:',
        I + "rows.append({",
        I * 2 + '"column": col,',
        I * 2 + '"chi-square p": df[f"{col}_chi-square_p_value"].iloc[0],',
        I * 2 + '"chi-square": df[f"{col}_chi-square_interpretation"].iloc[0],',
        I * 2 + '"KS p": df[f"{col}_kolmogorov-smirnov_p_value"].iloc[0],',
        I * 2 + '"KS": df[f"{col}_kolmogorov-smirnov_interpretation"].iloc[0],',
        I * 2 + '"GRTE": df[f"{col}_grte_grte_value"].iloc[0],',
        I * 2 + '"GRTE int.": df[f"{col}_grte_interpretation"].iloc[0],',
        I + "})",
        "summary = pd.DataFrame(rows)",
        "print(summary.to_markdown(index=False))",
    )
)

cells_n1.append(
    code(
        'raw = pd.read_parquet(' + VF_PARQUET + ')',
        'columns = ["image_bytes_luminosity", "image_bytes_contrast", "image_bytes_blur", "image_bytes_entropy"]',
        "short = {k: k.replace('image_bytes_', '') for k in columns}",
        "",
        "# Overlay normal-fit curves (red dashed) on per-feature histograms",
        "fig, axes = plt.subplots(2, 2, figsize=(10, 7))",
        "",
        "for ax, col in zip(axes.ravel(), columns):",
        I + "vals = raw[col].dropna()",
        I + 'ax.hist(vals, bins=50, density=True, alpha=0.6, color="#4C72B0")',
        I + "mu, std = vals.mean(), vals.std()",
        I + "x = np.linspace(vals.min(), vals.max(), 200)",
        I + "pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * std ** 2))",
        I + 'ax.plot(x, pdf, "r-", lw=2, label="Normal fit")',
        I + "ax.set_title(short[col].title())",
        I + 'ax.set_xlabel("Value")',
        I + 'ax.set_ylabel("Density")',
        I + "ax.legend(fontsize=8)",
        "",
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

# ═══════════════════════════════════════════════════════
# STEP 6: Domain Gap (Split)
# ═══════════════════════════════════════════════════════

cells_n1.append(
    md(
        "## 6. Domain Gap (Split)\n"
        "\n"
        "Splits the image dataset by `source` (zoo, safari, reserve) and computes pairwise FID between every pair of subsets."
    )
)

cells_n1.append(
    code(
        CONFIG_MARKER,
        'config = _load_config("../config/scenario/domain_gap_with_split.yaml")',
        RUN_CONFIG,
        "",
        'df = pd.read_parquet("../outputs/story_gap_split.parquet")',
        PRINT_DF,
    )
)

cells_n1.append(
    code(
        'df = pd.read_parquet("../outputs/story_gap_split.parquet")',
        "# Build symmetric FID matrix from pairwise results",
        'sel_names = sorted(set(df["selection_source"]) | set(df["selection_target"]))',
        "sources = list(sel_names)",
        'labels = [s.split("_", 1)[-1] for s in sel_names]',
        "n = len(sources)",
        "matrix = np.full((n, n), np.nan)",
        "for _, r in df.iterrows():",
        I + 'i = sources.index(r["selection_source"])',
        I + 'j = sources.index(r["selection_target"])',
        I + 'matrix[i, j] = r["fid"]',
        I + 'matrix[j, i] = r["fid"]',
        "np.fill_diagonal(matrix, 0)",
        "",
        "# Heatmap: darker = larger distribution shift between sources",
        "fig, ax = plt.subplots(figsize=(6, 5))",
        'im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")',
        "ax.set_xticks(range(n))",
        "ax.set_yticks(range(n))",
        "ax.set_xticklabels(labels)",
        "ax.set_yticklabels(labels)",
        "for i in range(n):",
        I + "for j in range(n):",
        I * 2 + "val = matrix[i, j]",
        I * 2 + 'color = "white" if val > 200 else "black"',
        I * 2 + 'ax.text(j, i, f"{val:.0f}", ha="center", va="center",',
        I * 3 + 'color=color, fontweight="bold")',
        'ax.set_title("Pairwise FID (Zoo / Safari / Reserve)")',
        COLORBAR,
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

# ═══════════════════════════════════════════════════════
# STEP 7: Domain Gap (Filter)
# ═══════════════════════════════════════════════════════

cells_n1.append(
    md(
        "## 7. Domain Gap (Filter)\n"
        "\n"
        "Filters to safari and reserve sources with two separate loaders and measures MMD-RBF between them."
    )
)

cells_n1.append(
    code(
        CONFIG_MARKER,
        'config = _load_config("../config/scenario/domain_gap_with_filter.yaml")',
        RUN_CONFIG,
        "",
        'df = pd.read_parquet("../outputs/story_gap_filter.parquet")',
        PRINT_DF,
    )
)


# ═══════════════════════════════════════════════════════
# STEP 8: Domain Gap (Split + Filter)
# ═══════════════════════════════════════════════════════

cells_n1.append(
    md(
        "## 8. Domain Gap (Split + Filter)\n"
        "\n"
        "Filters to zoo only, then splits by all class_name values (elephant, giraffe, lion, zebra). "
        "Wasserstein-1D measures distribution shift between every class pair, "
        "producing a 4\u00d74 heatmap of pairwise embedding distances within a single controlled environment."
    )
)

cells_n1.append(
    code(
        CONFIG_MARKER,
        'config = _load_config("../config/scenario/domain_gap_with_split_and_filter.yaml")',
        RUN_CONFIG,
        "",
        'df = pd.read_parquet("../outputs/story_gap_split_filter.parquet")',
        PRINT_DF,
    )
)

cells_n1.append(
    code(
        'df = pd.read_parquet("../outputs/story_gap_split_filter.parquet")',
        "# Build 4x4 Wasserstein-1D matrix, one row/col per animal class",
        'classes = ["elephant", "giraffe", "lion", "zebra"]',
        'labels = ["Elephant", "Giraffe", "Lion", "Zebra"]',
        "matrix = np.full((4, 4), np.nan)",
        "for _, r in df.iterrows():",
        I + 'src_cls = r["selection_source"].split("_")[-1]',
        I + 'tgt_cls = r["selection_target"].split("_")[-1]',
        I + "i = classes.index(src_cls)",
        I + "j = classes.index(tgt_cls)",
        I + 'matrix[i, j] = r["wasserstein_1d"]',
        I + 'matrix[j, i] = r["wasserstein_1d"]',
        "np.fill_diagonal(matrix, 0)",
        "",
        "fig, ax = plt.subplots(figsize=(7, 6))",
        'im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")',
        "ax.set_xticks(range(4))",
        "ax.set_yticks(range(4))",
        'ax.set_xticklabels(labels, rotation=45, ha="right")',
        "ax.set_yticklabels(labels)",
        "# Heatmap of embedding-distance between class pairs (same source)",
        "for i in range(4):",
        I + "for j in range(4):",
        I * 2 + "val = matrix[i, j]",
        I * 2 + 'color = "white" if val > 0.05 else "black"',
        I * 2 + 'ax.text(j, i, f"{val:.3f}", ha="center", va="center",',
        I * 3 + 'color=color, fontweight="bold")',
        'ax.set_title("Wasserstein-1D Distance (Zoo: All Classes)")',
        COLORBAR,
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

save("examples/notebooks/scenario_individual_steps.ipynb", cells_n1)


# ═══════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — Full Pipeline via DatasetJob
# ═══════════════════════════════════════════════════════════════════

# cells_n2: full-pipeline notebook — single DatasetJob with 8 processors, 3 loaders, 3 writers

cells_n2 = []

cells_n2.append(
    md(
        "# Notebook \u2014 Full Pipeline\n"
        "\n"
        "Builds the complete end-to-end pipeline from `full_story.yaml` using explicit `DatasetJob` "
        "object construction. This demonstrates how the Python object model works under the hood: "
        "DataLoaders, Processors, OutputWriters, and the Job orchestrator.\n"
        "\n"
        "The pipeline runs 3 filtered+split data loaders, 2 feature processors (visual + embedding), "
        "3 metric processors (completeness, diversity, representativeness), and 3 gap processors "
        "(FID, MMD-RBF, Wasserstein-1D) \u2014 all in a single `job.run()` call."
        "\n"
        "Prerequisites:"
        " - install dqm-ml in a virtual environment"
        "    with uv: `uv sync`"
        " - install notebooks dependencies"
        "    with uv: `uv pip install 'packages/dqm-ml[notebooks]'`"
        "\n"
        "Then select the virtual environment for jupyter"
        "\n"
        "### Setup\n"
    )
)

cells_n2.append(
    code(
        "from dqm_ml_job.job import DatasetJob",
        "from dqm_ml_job.dataloaders import ParquetDataLoader",
        "from dqm_ml_job.outputwriter import ParquetOutputWriter",
        "from dqm_ml_images.visual_features import VisualFeaturesProcessor",
        "from dqm_ml_pytorch.image_embedding import ImageEmbeddingProcessor",
        "from dqm_ml_pytorch.domain_gap import DomainGapProcessor",
        "from dqm_ml_core import (",
        I + "CompletenessProcessor,",
        I + "DiversityProcessor,",
        I + "RepresentativenessProcessor,",
        ")",
        "",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from pathlib import Path",
        "",
        "# Verify data files exist",
        'data_file = "../data/samples_with_images.parquet"',
        "if not Path(data_file).exists():",
        I + 'print("Run `python script/generate_data.py` first")',
    )
)

# ---- Global feature stats (for user_provided representativeness) ----"
cells_n2.append(
    code(
        "import pyarrow as pa",
        "",
        "# Compute global mean/std of visual features across ALL data",
        "# so representativeness uses a consistent reference distribution",
        "# instead of per-selection first-batch estimates.",
        'raw = pd.read_parquet("../data/samples_with_images.parquet")',
        'table = pa.Table.from_pandas(raw[["image_bytes", "source"]])',
        "",
        "vf_stats = VisualFeaturesProcessor(",
        I + 'name="vf_stats",',
        I + CONFIG_OPEN,
        I * 2 + COLUMNS_INPUT_IMAGE_BYTES,
        I * 2 + '"features": ["luminosity", "contrast", "blur", "entropy"],',
        I * 2 + '"grayscale": True,',
        I * 2 + '"normalize": True,',
        I * 2 + '"histogram": {"bins": 256},',
        I * 2 + '"laplacian_kernel": "3x3",',
        I + "},",
        ")",
        "",
        'feat_cols = ["image_bytes_luminosity", "image_bytes_contrast",',
        I * 2 + '"image_bytes_blur", "image_bytes_entropy"]',
        "feat_values = {c: [] for c in feat_cols}",
        "for batch in table.to_batches(max_chunksize=500):",
        I + "feats = vf_stats.compute_features(batch)",
        I + "for c in feat_values:",
        I * 2 + "feat_values[c].extend(feats[c].to_numpy())",
        "",
        "dist_params = [",
        I + '{"column": c, "mean": float(np.mean(feat_values[c])),',
        I * 2 + '"std": float(np.std(feat_values[c], ddof=0))}',
        I + "for c in feat_cols",
        "]",
        'print("Global feature stats for user_provided representativeness:")',
        "for p in dist_params:",
        I + 'print(f\'  {p["column"]:40s}  mean={p["mean"]:.4f}  std={p["std"]:.4f}\')',
    )
)

# ---- DataLoaders ----
cells_n2.append(
    md(
        "### Build DataLoaders\n"
        "\n"
        "Three `ParquetDataLoader` instances, each filtering by a different source and splitting by class."
    )
)

cells_n2.append(
    code(
        "loader_safari = ParquetDataLoader(",
        I + 'name="safari",',
        I + CONFIG_OPEN,
        I * 2 + PATH_DATA_FILE,
        I * 2 + BATCH_SIZE_500,
        I * 2 + FILTERS_OPEN,
        I * 3 + '{"column": "source", "values": ["safari"]},',
        I * 2 + "],",
        I * 2 + SPLIT_OPEN,
        I * 3 + BY_CLASS_NAME,
        I * 3 + VALUES_ELEPHANT_ZEBRA,
        I * 2 + "},",
        I + "},",
        ")",
        "",
        "loader_reserve = ParquetDataLoader(",
        I + 'name="reserve",',
        I + CONFIG_OPEN,
        I * 2 + PATH_DATA_FILE,
        I * 2 + BATCH_SIZE_500,
        I * 2 + FILTERS_OPEN,
        I * 3 + '{"column": "source", "values": ["reserve"]},',
        I * 2 + "],",
        I * 2 + SPLIT_OPEN,
        I * 3 + BY_CLASS_NAME,
        I * 3 + VALUES_ELEPHANT_ZEBRA,
        I * 2 + "},",
        I + "},",
        ")",
        "",
        "loader_zoo = ParquetDataLoader(",
        I + 'name="zoo",',
        I + CONFIG_OPEN,
        I * 2 + PATH_DATA_FILE,
        I * 2 + BATCH_SIZE_500,
        I * 2 + FILTERS_OPEN,
        I * 3 + '{"column": "source", "values": ["zoo"]},',
        I * 2 + "],",
        I * 2 + SPLIT_OPEN,
        I * 3 + BY_CLASS_NAME,
        I * 3 + VALUES_ELEPHANT_ZEBRA,
        I * 2 + "},",
        I + "},",
        ")",
        "",
        "dataloaders = {",
        I + '"safari": loader_safari,',
        I + '"reserve": loader_reserve,',
        I + '"zoo": loader_zoo,',
        "}",
        'print(f"Created {len(dataloaders)} dataloaders")',
    )
)

# ---- Processors ----
cells_n2.append(
    md(
        "### Build Processors\n"
        "\n"
        "Eight processors: visual features \u2192 embeddings (feature producers), then completeness, "
        "diversity, representativeness (metrics), and finally three domain-gap variants (FID, MMD-RBF, Wasserstein)."
    )
)

cells_n2.append(
    code(
        "# Feature processors",
        "vf = VisualFeaturesProcessor(",
        I + 'name="visual_features",',
        I + CONFIG_OPEN,
        I * 2 + COLUMNS_INPUT_IMAGE_BYTES,
        I * 2 + '"features": ["luminosity", "contrast", "blur", "entropy"],',
        I * 2 + '"grayscale": True,',
        I * 2 + '"normalize": True,',
        I * 2 + '"histogram": {"bins": 256},',
        I * 2 + '"laplacian_kernel": "3x3",',
        I + "},",
        ")",
        "",
        "emb = ImageEmbeddingProcessor(",
        I + 'name="embedding",',
        I + CONFIG_OPEN,
        I * 2 + COLUMNS_INPUT_IMAGE_BYTES,
        I * 2 + '"model": {"arch": "resnet18", "n_layer_feature": -2},',
        I * 2 + '"infer": {',
        I * 3 + '"batch_size": 32,',
        I * 3 + '"width": 64,',
        I * 3 + '"height": 64,',
        I * 3 + '"norm_mean": [0.485, 0.456, 0.406],',
        I * 3 + '"norm_std": [0.229, 0.224, 0.225],',
        I * 2 + "},",
        I + "},",
        ")",
        "",
        "# Metric processors",
        "comp = CompletenessProcessor(",
        I + 'name="completeness",',
        I + CONFIG_OPEN,
        I * 2 + '"columns": {',
        I * 3 + '"input": [',
        I * 4 + '"quality_score",',
        I * 3 + "],",
        I * 2 + "},",
        I * 2 + '"include_per_column": True,',
        I * 2 + '"include_overall": True,',
        I + "},",
        ")",
        "",
        "div = DiversityProcessor(",
        I + 'name="diversity",',
        I + CONFIG_OPEN,
        I * 2 + '"columns": {"input": ["class_name"]},',
        I * 2 + '"metrics": ["simpson", "gini", "shannon", "richness"],',
        I + "},",
        ")",
        "",
        "rep = RepresentativenessProcessor(",
        I + 'name="representativeness",',
        I + CONFIG_OPEN,
        I * 2 + '"columns": {',
        I * 3 + '"input": [',
        I * 4 + LUMINOSITY,
        I * 4 + CONTRAST,
        I * 4 + BLUR,
        I * 4 + ENTROPY,
        I * 3 + "],",
        I * 2 + "},",
        I * 2 + '"metrics": [',
        I * 3 + '"chi-square", "grte", "shannon-entropy", "kolmogorov-smirnov",',
        I * 2 + "],",
        I * 2 + '"distribution": "normal",',
        I * 2 + '"mean_std_estimation": "user_provided",',
        I * 2 + '"distribution_params": dist_params,',
        I * 2 + '"histogram": {"bins": 5},',
        I + "},",
        ")",
        "",
        "# Gap processors",
        "fid = DomainGapProcessor(",
        I + 'name="fid_gap",',
        I + CONFIG_OPEN,
        I * 2 + COLUMNS_INPUT_EMBEDDING,
        I * 2 + '"distance": {"metric": "fid", "epsilon": 1e-6},',
        I + "},",
        ")",
        "",
        "mmd = DomainGapProcessor(",
        I + 'name="mmd_rbf_gap",',
        I + CONFIG_OPEN,
        I * 2 + COLUMNS_INPUT_EMBEDDING,
        I * 2 + '"distance": {',
        I * 3 + '"metric": "mmd_rbf",',
        I * 3 + '"kernel_params": {"gamma": 1.0},',
        I * 2 + "},",
        I + "},",
        ")",
        "",
        "wass = DomainGapProcessor(",
        I + 'name="wasserstein_gap",',
        I + CONFIG_OPEN,
        I * 2 + COLUMNS_INPUT_EMBEDDING,
        I * 2 + '"distance": {"metric": "wasserstein_1d"},',
        I + "},",
        ")",
        "",
        "feature_processors = {\"vf\": vf, \"emb\": emb}",
        "metric_processors = {\"comp\": comp, \"div\": div, \"rep\": rep}",
        "gap_processors = {\"fid\": fid, \"mmd\": mmd, \"wass\": wass}",
        'print(f"Created {sum(len(d) for d in [feature_processors, metric_processors, gap_processors])} processors")',
    )
)

# ---- OutputWriters ----
cells_n2.append(
    md(
        "### Build Output Writers\n"
        "\n"
        "Three parquet writers for features, per-selection metrics, and pairwise gap results."
    )
)

cells_n2.append(
    code(
        "feat_writer = ParquetOutputWriter(",
        I + 'name="features",',
        I + CONFIG_OPEN,
        I * 2 + '"path_pattern": "../outputs/full_story_features.parquet",',
        I * 2 + '"columns": ["sample_id", "source", "class_name"],',
        I + "},",
        ")",
        "",
        "met_writer = ParquetOutputWriter(",
        I + 'name="metrics",',
        I + CONFIG_OPEN,
        I * 2 + '"path_pattern": "../outputs/full_story_metrics.parquet",',
        I + "},",
        ")",
        "",
        "gap_writer = ParquetOutputWriter(",
        I + 'name="delta",',
        I + CONFIG_OPEN,
        I * 2 + '"path_pattern": "../outputs/full_story_gap.parquet",',
        I + "},",
        ")",
        "",
        'print("Output writers ready")',
    )
)

# ---- Assemble and run ----
cells_n2.append(
    md(
        "### Assemble DatasetJob and Execute\n"
        "\n"
        "The `DatasetJob` ties everything together. It discovers data selections from the loaders "
        "(6 total: 3 sources \u00d7 2 classes), streams through batches, computes features and metrics, "
        "and produces the delta (gap) table."
    )
)

cells_n2.append(
    code(
        "job = DatasetJob(",
        I + "dataloaders=dataloaders,",
        I + "features_processors=feature_processors,",
        I + "metrics_processors=metric_processors,",
        I + "gap_processors=gap_processors,",
        I + "features_output=feat_writer,",
        I + "progress_bar=True,",
        I + "threads=4,",
        I + "compute_seed=42,",
        I + 'compute_device="auto",',
        ")",
        "",
        "metrics_dict, delta_table = job.run()",
        "",
        "# Write metrics and gap outputs",
        "# Flush after each write — ParquetOutputWriter uses accumulate mode",
        "# when path_pattern lacks '{}', buffering data without flushing.",
        "met_writer.write_metrics_dict(metrics_dict)",
        "met_writer.flush()",
        "",
        "if delta_table is not None:",
        I + "delta_data = {",
        I * 2 + "col: delta_table.column(col)",
        I * 2 + "for col in delta_table.column_names",
        I + "}",
        I + 'gap_writer.write_table("delta", delta_data)',
        I + "gap_writer.flush()",
        "",
        'print(f"Processed {len(metrics_dict)} selections")',
        'print(f"Delta table: {delta_table.num_rows if delta_table else 0} rows")',
    )
)

# ---- Display results ----
cells_n2.append(
    md(
        "### Results\n"
        "\n"
        "Three output files \u2014 features (605 rows), metrics (6 rows), gap (45 rows = 15 pairs \u00d7 3 metrics)."
    )
)

cells_n2.append(
    code(
        "# Load the three output files produced by DatasetJob",
        'features = pd.read_parquet("../outputs/full_story_features.parquet")',
        'metrics = pd.read_parquet("../outputs/full_story_metrics.parquet")',
        'gap = pd.read_parquet("../outputs/full_story_gap.parquet")',
        "",
        '#print("### Features \u2014 aggregated by source")',
        "feat_cols = [",
        I + LUMINOSITY,
        I + CONTRAST,
        I + BLUR,
        I + ENTROPY,
        "]",
        '#print(features.groupby("source")[feat_cols].describe().to_markdown())',
        "",
        '#print("\\n### Per-Selection Metrics")',
        "sel_cols = [",
        I + '"selection", "count",',
        I + '"completeness_overall",',
        I + '"completeness_quality_score",',
        I + '"class_name_richness",',
        I + '"class_name_shannon",',
        I + '"image_bytes_luminosity_chi-square_interpretation",',
        I + '"image_bytes_contrast_chi-square_interpretation",',
        I + '"image_bytes_blur_chi-square_interpretation",',
        I + '"image_bytes_entropy_chi-square_interpretation",',
        "]",
        "#print(metrics[sel_cols].to_markdown(index=False))",
        "",
        '#print("\\n### Gap \u2014 pivoted (one row per pair)")',
        "metric_cols = [",
        I + '"selection_source", "selection_target",',
        I + '"fid", "mmd_rbf", "wasserstein_1d",',
        "]",
        'gap_clean = gap[metric_cols].dropna(how="all", subset=["fid", "mmd_rbf", "wasserstein_1d"])',
        "# Reshape: one row per pair with all 3 metrics",
        'pivoted = gap_clean.groupby(["selection_source", "selection_target"]).first().reset_index()',
        'pivoted = pivoted.sort_values("fid", ascending=False, na_position="last")',
        "#print(pivoted.to_markdown(index=False))",
        "",
        '#print("\\n### Representativeness \\u2014 Chi-Square p-values")',
        'feat_names = ["luminosity", "contrast", "blur", "entropy"]',
        'feat_labels = ["Luminosity", "Contrast", "Blur", "Entropy"]',
        'sel_labels = [s.replace("animals_", "") for s in metrics["selection"]]',
        "",
        "pval_rows = {}",
        "nlog_rows = {}",
        "for feat, label in zip(feat_names, feat_labels):",
        I + 'raw = metrics[f"image_bytes_{feat}_chi-square_p_value"].values',
        I + "nlog = -np.log10(raw.astype(float))",
        I + "pval_rows[label] = raw",
        I + "nlog_rows[label] = nlog",
        "",
        "pval_df = pd.DataFrame(pval_rows, index=sel_labels).T",
        "nlog_df = pd.DataFrame(nlog_rows, index=sel_labels).T",
        "",
        '#print("\\nRaw p-values:")',
        '#print(pval_df.to_markdown(floatfmt=".4g"))',
        '#print("\\n\\u2212log\\u2081\\u2080(p-value) (heatmap values):")',
        '#print(nlog_df.to_markdown(floatfmt=".2f"))',
        "#print()",
    )
)

# ---- Viz A: Representativeness Heatmap ----
cells_n2.append(
    md(
        "### Visualization A: Representativeness Heatmap\n"
        "\n"
        "Chi-square normality test p-values (as \u2212log\u2081\u2080) for each visual feature across all 6 selections.\n"
        "Higher values mean stronger deviation from a normal distribution. Safari selections clearly stand out with non-normal contrast, blur, and entropy."
    )
)

cells_n2.append(
    code(
        'feat_names = ["luminosity", "contrast", "blur", "entropy"]',
        'feat_labels = ["Luminosity", "Contrast", "Blur", "Entropy"]',
        "",
        "# −log₁₀ transform: larger values → stronger deviation from normal",
        "pval_data = []",
        "for feat in feat_names:",
        I + 'col = f"image_bytes_{feat}_chi-square_p_value"',
        I + "vals = -np.log10(metrics[col].values.astype(float))",
        I + "pval_data.append(vals)",
        "pval_matrix = np.array(pval_data)",
        "",
        "fig, ax = plt.subplots(figsize=(9, 5))",
        'im = ax.imshow(pval_matrix, cmap="YlOrRd", aspect="equal", vmin=0)',
        "",
        'selection_labels = [s.replace("animals_", "") for s in metrics["selection"]]',
        "ax.set_xticks(range(6))",
        "ax.set_yticks(range(4))",
        'ax.set_xticklabels(selection_labels, rotation=45, ha="right", fontsize=8)',
        "ax.set_yticklabels(feat_labels, fontsize=9)",
        "",
        "for i in range(4):",
        I + "for j in range(6):",
        I * 2 + "val = pval_matrix[i, j]",
        I * 2 + "if np.isnan(val):",
        I * 3 + 'txt = "N/A"',
        I * 3 + 'color = "gray"',
        I * 2 + "else:",
        I * 3 + 'txt = f"{val:.2f}"',
        I * 3 + 'color = "white" if val > 90 else "black"',
        I * 2 + 'ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)',
        "",
        'ax.set_title("Normality Test (chi-square) \\u2014 \\u2212log\\u2081\\u2080(p-value)")',
        COLORBAR,
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

# ---- Viz B: FID heatmap ----
cells_n2.append(
    md(
        "### Visualization B: FID Heatmap\n"
        "\n"
        "All 6 selections compared pairwise \u2014 reveals which (source, class) pairs are most different."
    )
)

cells_n2.append(
    code(
        "# Build 6x6 FID matrix",
        'gap_clean = gap.dropna(subset=["fid"])',
        'selections = sorted(set(gap_clean["selection_source"].unique()) | set(gap_clean["selection_target"].unique()))',
        "n = len(selections)",
        "fid_mat = np.full((n, n), np.nan)",
        'name_map = {s: s.replace("animals_", "") for s in selections}',
        "",
        "for _, r in gap_clean.iterrows():",
        I + 'i = selections.index(r["selection_source"])',
        I + 'j = selections.index(r["selection_target"])',
        I + 'fid_mat[i, j] = r["fid"]',
        I + 'fid_mat[j, i] = r["fid"]',
        "np.fill_diagonal(fid_mat, 0)",
        "",
        "fig, ax = plt.subplots(figsize=(8, 7))",
        'im = ax.imshow(fid_mat, cmap="YlOrRd", aspect="equal")',
        "labels_short = [name_map[s] for s in selections]",
        "ax.set_xticks(range(n))",
        "ax.set_yticks(range(n))",
        'ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)',
        "ax.set_yticklabels(labels_short, fontsize=8)",
        "for i in range(n):",
        I + "for j in range(n):",
        I * 2 + "val = fid_mat[i, j]",
        I * 2 + "if not np.isnan(val):",
        I * 3 + 'color = "white" if val > 135 else "black"',
        I * 3 + 'ax.text(j, i, f"{val:.0f}", ha="center", va="center",',
        I * 4 + "color=color, fontsize=7)",
        'ax.set_title("Pairwise FID \u2014 All 6 Selections")',
        COLORBAR,
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

# ---- Viz C: Three Metrics Compared ----
cells_n2.append(
    md(
        "### Visualization C: Three Metrics Compared\n"
        "\n"
        "Min-max normalised FID, MMD-RBF, and Wasserstein-1D for the top 8 pairs, sorted by FID."
    )
)

cells_n2.append(
    code(
        "# Pivot and normalise for comparison",
        "pvt = pivoted.copy()",
        'norm_cols = ["fid", "mmd_rbf", "wasserstein_1d"]',
        "for c in norm_cols:",
        I + "lo, hi = pvt[c].min(), pvt[c].max()",
        I + 'pvt[f"{c}_norm"] = (pvt[c] - lo) / (hi - lo) if hi > lo else 0',
        "",
        "top8 = pvt.head(8)",
        'top8["pair"] = [',
        I + 'row["selection_source"].replace("animals_", "")',
        I + '+ " \u2192 "',
        I + '+ row["selection_target"].replace("animals_", "")',
        I + "for _, row in top8.iterrows()",
        "]",
        "",
        "fig, ax = plt.subplots(figsize=(10, 5))",
        "x = np.arange(len(top8))",
        "w = 0.25",
        'ax.bar(x - w, top8["fid_norm"], w, label="FID", color="#4C72B0")',
        'ax.bar(x, top8["mmd_rbf_norm"], w, label="MMD-RBF", color="#DD8452")',
        'ax.bar(x + w, top8["wasserstein_1d_norm"], w, label="Wasserstein", color="#55A868")',
        "ax.set_xticks(x)",
        'ax.set_xticklabels(top8["pair"], rotation=45, ha="right", fontsize=8)',
        'ax.set_ylabel("Normalised score (0\u20131)")',
        'ax.set_title("Top 8 Domain Gaps \u2014 Three Metrics (Min-Max Normalised)")',
        "ax.legend(fontsize=9)",
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

# ---- Viz D: Embedding Space ----
cells_n2.append(
    md(
        "### Visualization D: Embedding Space\n"
        "\n"
        "PCA projection of all 605 image embeddings, colored by acquisition source with class as marker style."
    )
)

cells_n2.append(
    code(
        "from sklearn.decomposition import PCA",
        "",
        'emb = np.array(features["image_bytes_embedding"].to_list())',
        "# Color = source (zoo/safari/reserve), marker = class (elephant/lion/giraffe/zebra)",
        "pca = PCA(n_components=2, random_state=42)",
        "coords = pca.fit_transform(emb)",
        "vp = pca.explained_variance_ratio_",
        "",
        "fig, ax = plt.subplots(figsize=(9, 7))",
        'source_colors = {"zoo": "#4C72B0", "safari": "#DD8452", "reserve": "#55A868"}',
        'class_markers = {"elephant": "o", "lion": "s", "giraffe": "^", "zebra": "D"}',
        "",
        'for src in ["zoo", "safari", "reserve"]:',
        I + 'for cls in ["elephant", "lion", "giraffe", "zebra"]:',
        I * 2 + 'mask = (features["source"] == src) & (features["class_name"] == cls)',
        I * 2 + "if mask.any():",
        I * 3 + "ax.scatter(coords[mask, 0], coords[mask, 1],",
        I * 4 + "c=source_colors[src], marker=class_markers[cls],",
        I * 4 + 'alpha=0.5, s=15, edgecolors="none",',
        I * 4 + 'label=f"{src}_{cls}", )',
        "",
        I + 'ax.set_xlabel(f"PC1 ({vp[0]:.1%} var)")',
        I + 'ax.set_ylabel(f"PC2 ({vp[1]:.1%} var)")',
        I + 'ax.set_title("Embedding Space \u2014 PCA (color = source, shape = class)")',
        I + "ax.legend(markerscale=2, fontsize=6, ncol=2)",
        TIGHT_LAYOUT,
        PLT_SHOW,
    )
)

# ---- Summary ----
cells_n2.append(
    md(
        "### Summary\n"
        "\n"
        "The full pipeline produced three output files with feature vectors, per-selection metrics, "
        "and pairwise domain-gap scores. Key findings:\n"
        "\n"
        "* **FID** (36\u2013636 dynamic range) provides the clearest separation between domains\n"
        "* **MMD-RBF** with default \u03b3=1.0 shows minimal variation (0.018\u20130.023) across all pairs\n"
        "* **Wasserstein-1D** follows FID\u2019s ranking with moderate separation\n"
        "* Same-source, different-species pairs have the lowest FID (36\u201346)\n"
        "* Cross-source pairs of the same species rank higher (122\u2013624)\n"
        "* Zoo\u2194safari pairs show the largest shift (FID 533\u2013636)\n"
        "* Safari selections have non-normal contrast, blur, and entropy "
        "(chi-square p < 0.05); reserve and zoo are predominantly normal "
        "or have insufficient histogram bins\n"
        "\n"
        "For this dataset, FID alone suffices to detect and rank domain gaps."
    )
)

save("examples/notebooks/scenario_full_pipeline.ipynb", cells_n2)
