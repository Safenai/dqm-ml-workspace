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

## How Metrics are Configured

Each metric is configured in the `metrics_processor` section of your YAML config. See the [Configuration Guide](configuration.md) for details.

Each metric page has:
- Configuration parameters
- Example YAML config
- Output format
