# DQM-ML CLI Wrapper

Main CLI entry point for DQM-ML. Consolidates all modular packages into a single command-line interface.

## Installation

```bash
# Basic installation (core only)
pip install dqm-ml

# Installation with optional components
pip install "dqm-ml[all]"      # Everything
pip install "dqm-ml[job]"      # core + job
pip install "dqm-ml[pytorch]" # core + pytorch
pip install "dqm-ml[images]"  # core + images
pip install "dqm-ml[notebooks]" # Jupyter support
```

## Quick Start

### Process a Dataset

Run a data quality pipeline from a configuration file:

```bash
dqm-ml process -p config.yaml
```

### List Available Plugins

Show all registered metrics and data loaders:

```bash
dqm-ml list
```

### Check Version

```bash
dqm-ml version
```

## Commands

| Command | Description |
|---------|-------------|
| **process** | Execute a data quality pipeline from a YAML config |
| **list** | Show all available plugins (metrics, loaders) |
| **version** | Display version information |

## Configuration

DQM-ML uses YAML configuration files to define:

- Data sources (dataloaders)
- Metrics to compute (metrics: interface)
- Output settings (outputs)

### Completeness Example

```yaml
metrics:
  processors:
    - name: completeness
      type: completeness
      columns:
        input: [col_a, col_b]

dataloaders:
  loaders:
    - name: train
      type: parquet
      path: data/train.parquet
```

### Representativeness Example

```yaml
metrics:
  processors:
    - name: representativeness
      type: representativeness
      columns:
        input: [feature_x, feature_y]
      distribution: "normal"
      metrics: ["chi-square", "kolmogorov-smirnov"]

dataloaders:
  loaders:
    - name: train
      type: parquet
      path: data/train.parquet
```

### Domain Gap Example

```yaml
gap:
  processors:
    - name: domain_gap
      type: domain_gap
      columns:
        input: ["embedding"]
      distance:
        metric: "mmd_linear"

dataloaders:
  loaders:
    - name: source
      type: parquet
      path: data/source.parquet
    - name: target
      type: parquet
      path: data/target.parquet
```

### Visual Features Example

```yaml
features:
  processors:
    - name: visual
      type: image_features
      columns:
        input: ["image_data"]
      grayscale: true

dataloaders:
  loaders:
    - name: images
      type: parquet
      path: data/images.parquet
```

### Multiple Metrics Example

```yaml
metrics:
  processors:
    - name: completeness
      type: completeness
      columns:
        input: [col_a, col_b]
    - name: representativeness
      type: representativeness
      columns:
        input: [feature_x]
      distribution: "normal"

dataloaders:
  loaders:
    - name: train
      type: parquet
      path: data/train.parquet
```

## See Also

- [Formal and Core Concepts](https://safenai.github.io/dqm-ml-workspace/docs/formal_concepts.md) for definitions of **Processor**, **Metric**, **Feature**, and related terminology.
- [Documentation](https://safenai.github.io/dqm-ml-workspace/)
- [Metrics Guide](https://safenai.github.io/dqm-ml-workspace/docs/metrics/)
- [Configuration Guide](https://safenai.github.io/dqm-ml-workspace/docs/configuration/)