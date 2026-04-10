# DQM-ML V2 CLI Wrapper

Main CLI entry point for DQM-ML V2. Consolidates all modular packages into a single command-line interface.

## Installation

```bash
# Basic installation (core only)
pip install dqm-ml-v2

# Installation with optional components
pip install "dqm-ml-v2[all]"      # Everything
pip install "dqm-ml-v2[pipeline]" # core + job
pip install "dqm-ml-v2[pytorch]"  # core + pytorch
pip install "dqm-ml-v2[images]"   # core + images
pip install "dqm-ml-v2[notebooks]" # Jupyter support
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
- Metrics to compute (metrics_processor)
- Output settings (outputs)

### Completeness Example

```yaml
dataloaders:
  train:
    type: parquet
    path: data/train.parquet

metrics_processor:
  completeness:
    type: completeness
    input_columns: [col_a, col_b]
```

### Representativeness Example

```yaml
dataloaders:
  train:
    type: parquet
    path: data/train.parquet

metrics_processor:
  representativeness:
    type: representativeness
    input_columns: [feature_x, feature_y]
    distribution: "normal"
    metrics: ["chi-square", "kolmogorov-smirnov"]
```

### Domain Gap Example

```yaml
dataloaders:
  source:
    type: parquet
    path: data/source.parquet
  target:
    type: parquet
    path: data/target.parquet

metrics_processor:
  domain_gap:
    type: domain_gap
    INPUT:
      embedding_col: "features"
    DELTA:
      metric: "mmd_linear"
```

### Visual Features Example

```yaml
dataloaders:
  images:
    type: parquet
    path: data/images.parquet

metrics_processor:
  visual:
    type: visual_metric
    input_columns: ["image_data"]
    grayscale: true
```

### Multiple Metrics Example

```yaml
dataloaders:
  train:
    type: parquet
    path: data/train.parquet

metrics_processor:
  completeness:
    type: completeness
    input_columns: [col_a, col_b]
  
  representativeness:
    type: representativeness
    input_columns: [feature_x]
    distribution: "normal"
```

## See Also

- [Documentation](https://safenai.github.io/dqm-ml-workspace/)
- [Metrics Guide](https://safenai.github.io/dqm-ml-workspace/docs/metrics/)
- [Configuration Guide](https://safenai.github.io/dqm-ml-workspace/docs/configuration/)