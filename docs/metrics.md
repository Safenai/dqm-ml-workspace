# Metrics Guide

DQM-ML provides several types of metrics to assess different aspects of data quality. This guide helps you choose the right metric for your needs.

## Choosing a Metric

Not sure which metric you need? Here's a quick guide:

| What you want to check | Use this metric |
|------------------------|-----------------|
| **Missing values** | [Completeness](metrics/completeness.md) |
| **Data distribution matches expected pattern** | [Representativeness](metrics/representativeness.md) |
| **Train/test data drift** | [Domain Gap](metrics/domain_gap.md) |
| **Image brightness, blur, quality** | [Visual Features](metrics/visual_features.md) |

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

## Metric Details

### Completeness

Measures what percentage of your data is present (non-null). Great for finding missing values:

```yaml
metrics_processor:
  completeness:
    type: completeness
    input_columns: [column_a, column_b]
    include_per_column: true
    include_overall: true
```

### Representativeness

Checks if your data follows a known distribution (Normal or Uniform). Useful for:

- Validating synthetic data
- Checking for data drift
- Ensuring balanced datasets

Includes multiple statistical tests:

- **Chi-Square (χ²)** — Goodness-of-fit test for categorical/binned data
- **Kolmogorov-Smirnov (KS)** — Non-parametric test for continuous distributions
- **Shannon Entropy** — Measures information diversity in your data
- **GRTE** (Granular Relative Theoretical Entropy) — Developed in the Confiance.ai program

### Domain Gap

Measures how different two datasets are from each other. Use it to:

- Compare training and test distributions
- Detect data shift over time
- Validate data augmentation

Available metrics:

- **Wasserstein** — Earth mover's distance for distribution comparison
- **MMD** (Maximum Mean Discrepancy) — Kernel-based distribution distance
- **FID** (Fréchet Inception Distance) — Deep learning-based image distance
- **KLMVN** (Kullback-Leibler Multivariate Normal) — KL divergence for Gaussian distributions
- **H-Divergence** — Hypothesis-based divergence measure

### Visual Features

Extracts image quality metrics like:

- **Luminosity** — Brightness level
- **Contrast** — Difference between light and dark
- **Blur** — Sharpness/clarity
- **Entropy** — Information diversity

## How Metrics are Configured

Each metric is configured in the `metrics_processor` section of your YAML config. See the [Configuration Guide](configuration.md) for details.

Each metric page has:
- Configuration parameters
- Example YAML config
- Output format
