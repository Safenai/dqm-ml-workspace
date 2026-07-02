# Metrics Guide

DQM-ML provides three **Interfaces** of processors to assess different aspects of data quality. This guide helps you choose the right interface and metric for your needs.

> **See also:** [Concepts](formal_concepts.md) for definitions of **Metric**, **Domain Gap**, **Batch Metric**, and related terminology used throughout this page.

## Quick Decision Guide

Not sure which metric you need? Use this guide:

| If you need to... | Use this Interface | Use this metric | Complexity |
|-------------------|--------------------|-----------------|------------|
| **Find missing values** | **Metrics** | [Completeness](metrics/completeness.md) | Low (CPU only) |
| **Check if data matches a distribution** | **Metrics** | [Representativeness](metrics/representativeness.md) | Low (CPU only) |
| **Measure category diversity** | **Metrics** | [Diversity](metrics/diversity.md) | Low (CPU only) |
| **Compare train/test distributions** | **Gap** | [Domain Gap](metrics/domain_gap.md) | High (requires PyTorch) |
| **Check image quality** | **Features** | [Visual Features](metrics/visual_features.md) | Medium (CPU only) |
| **Generate image embeddings** | **Features** | [Embedding Features](metrics/features_embeddings.md) | High (requires PyTorch) |

### Complexity Guide

- **Low (CPU only)**: Metrics (Completeness, Representativeness, Diversity) — runs on any machine
- **Medium**: Features (Visual Features) — requires opencv, but no GPU needed
- **High (GPU recommended)**: Features (Embedding Features), Gap (Domain Gap) — use PyTorch, faster with GPU

### The Math behind the metrics

Each metric is based on established statistical methods:

**Metrics Interface:**
- **Completeness**: Ratio of non-null values
- **Representativeness**: χ² (Chi-Square), KS (Kolmogorov-Smirnov), Shannon Entropy, GRTE
- **Diversity**: Simpson, Gini-Simpson, Shannon Entropy, Richness

**Gap Interface:**
- **Domain Gap**: MMD (Linear, RBF, Poly), FID, Wasserstein, KLMVN, PAD, CMD

**Features Interface:**
- **Visual Features**: Laplacian variance, histogram entropy
- **Embedding Features**: Pre-trained ResNet embeddings

## Available Metrics by Interface

### Metrics Interface (`dqm_ml.metrics`)

These are the most commonly used metrics for tabular data quality:

- **[Completeness](metrics/completeness.md)** - Checks for missing/null values in your data
- **[Representativeness](metrics/representativeness.md)** - Validates that data follows an expected distribution (Normal, Uniform)
- **[Diversity](metrics/diversity.md)** - Measures category diversity via Simpson, Gini-Simpson, Shannon, and Richness indices

Package: `dqm-ml-core` | Entry Point: `dqm_ml.metrics`

### Features Interface (`dqm_ml.features`)

For extracting feature columns from data:

- **[Visual Features](metrics/visual_features.md)** - Extracts image quality indicators like brightness, contrast, sharpness, and entropy
- **[Embedding Features](metrics/features_embeddings.md)** - Generates vector embeddings from images (e.g., ResNet-50) for downstream analysis

Packages: `dqm-ml-images`, `dqm-ml-pytorch` | Entry Point: `dqm_ml.features`

### Gap Interface (`dqm_ml.gap`)

For pairwise comparison between two Data Selections:

- **[Domain Gap](metrics/domain_gap.md)** - Measures statistical distance between two datasets (useful for comparing dataset distributions)

Package: `dqm-ml-pytorch` | Entry Point: `dqm_ml.gap`

## How Metrics are Configured

Each metric is configured under its corresponding interface in your YAML config:

- `features:` — Visual Features, Embedding Features
- `metrics:` — Completeness, Representativeness, Diversity
- `gap:` — Domain Gap

See the [Configuration Guide](configuration/overview.md) for details.

Each metric page has:
- Configuration parameters
- Example YAML config
- Output format
