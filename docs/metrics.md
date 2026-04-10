# Metrics Guide

DQM-ML provides several types of metrics to assess different aspects of data quality. This guide helps you choose the right metric for your needs.

## Quick Decision Guide

Not sure which metric you need? Use this guide:

| If you need to... | Use this metric | Complexity |
|-------------------|-----------------|------------|
| **Find missing values** | [Completeness](metrics/completeness.md) | Low (CPU only) |
| **Check if data matches a distribution** | [Representativeness](metrics/representativeness.md) | Low (CPU only) |
| **Compare train/test distributions** | [Domain Gap](metrics/domain_gap.md) | High (requires PyTorch) |
| **Check image quality** | [Visual Features](metrics/visual_features.md) | Medium (CPU only) |

### Complexity Guide

- **Low (CPU only)**: Completeness, Representativeness — runs on any machine
- **Medium**: Visual Features — requires opencv, but no GPU needed
- **High (GPU recommended)**: Domain Gap — uses PyTorch, faster with GPU

### The Math behind the metrics

Each metric is based on established statistical methods:

- **Completeness**: Ratio of non-null values
- **Representativeness**: χ² (Chi-Square), KS (Kolmogorov-Smirnov), Shannon Entropy, GRTE
- **Domain Gap**: MMD (Maximum Mean Discrepancy), FID (Fréchet Inception Distance), Wasserstein
- **Visual Features**: Laplacian variance, histogram entropy

## Available Metrics by Package

### Core Metrics (`dqm-ml-core`)

These are the most commonly used metrics for tabular data quality:

- **[Completeness](metrics/completeness.md)** - Checks for missing/null values in your data
- **[Representativeness](metrics/representativeness.md)** - Validates that data follows an expected distribution (Normal, Uniform)

### Visual Metrics (`dqm-ml-images`)

For analyzing image datasets:

- **[Visual Features](metrics/visual_features.md)** - Extracts image quality indicators like brightness, contrast, sharpness, and entropy

### Advanced Metrics (`dqm-ml-pytorch`)

For comparing datasets using deep learning embeddings:

- **[Domain Gap](metrics/domain_gap.md)** - Measures statistical distance between two datasets (useful for detecting data drift)

## How Metrics are Configured

Each metric is configured in the `metrics_processor` section of your YAML config. See the [Configuration Guide](configuration.md) for details.

Each metric page has:
- Configuration parameters
- Example YAML config
- Output format
