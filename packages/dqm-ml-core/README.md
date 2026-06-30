# DQM-ML Core

Core package for DQM-ML V2 providing the foundational API and standard metrics for data quality assessment.

## Installation

```bash
pip install dqm-ml-core
```

> **Note:** `dqm-ml-core` provides **Metric** **Processors** only — no CLI or job orchestration. Use directly via Python or with `dqm-ml-job` for YAML config execution.

## Quick Start

### Completeness Example

```python
from dqm_ml_core import CompletenessProcessor

processor = CompletenessProcessor(
    name="my_check",
    config={"columns": {"input": ["col_a", "col_b"]}, "include_per_column": true, "include_overall": true}
)
result = processor.compute({})
print(f"Completeness: {result['overall_completeness']}")
```

### Representativeness Example

```python
from dqm_ml_core import RepresentativenessProcessor
import numpy as np

# Create sample data (e.g., 1000 samples from normal distribution)
data = np.random.randn(1000)

processor = RepresentativenessProcessor(
    name="dist_check",
    config={
        "columns": {"input": ["feature"]},
        "distribution": "normal",
        "metrics": ["chi-square", "kolmogorov-smirnov"]
    }
)

result = processor.compute({})
print(f"Chi-Square p-value: {result['feature_chi-square_pvalue']}")
print(f"KS p-value: {result['feature_kolmogorov-smirnov_pvalue']}")
```

### With dqm-ml-job

For running from a YAML config, install together with `dqm-ml-job`:

```bash
pip install dqm-ml-job dqm-ml-core
```

Then use this config:

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

## Core Concepts

### DatametricProcessor

The base class for all **Metrics** and **Feature** extractors. It supports a streaming architecture by splitting computation into two phases:

1. **Batch Level**: `compute_batch_metric()` updates intermediate **Batch Metric** statistics for a single chunk of data.
2. **Dataset Level**: `compute()` aggregates these statistics into final **Metric** scores.

## Included Metrics

| Metric | Description |
|--------|-------------|
| **Completeness** | Analyzes null/missing values in your dataset |
| **Representativeness** | Statistical distribution analysis (Chi-Square, KS, Shannon Entropy, GRTE) |
| **Diversity** | Measures category distribution spread (Simpson, Gini-Simpson, Shannon, Richness) |

## For Developers

To create a new metric:

1. Subclass `dqm_ml_core.api.data_processor.DatametricProcessor`.
2. Define `needed_columns()`, `generated_features()`, and `generated_metrics()`.
3. Implement the streaming logic in `compute_batch_metric()` and `compute()`.

## Dependencies

DQM-ML is modular. For core metrics:

```bash
# Minimal: use as library only
pip install dqm-ml-core

# For YAML config execution
pip install dqm-ml-job dqm-ml-core

# Full stack with all metrics
pip install dqm-ml-job dqm-ml-core dqm-ml-images dqm-ml-pytorch
```

## See Also

- [Formal and Core Concepts](https://safenai.github.io/dqm-ml-workspace/docs/formal_concepts.md) for definitions of **Sample**, **Metric**, **Data Selection**, **Batch**, and related terminology.
- [Metrics Documentation](https://safenai.github.io/dqm-ml-workspace/docs/metrics/)
- [API Reference](https://safenai.github.io/dqm-ml-workspace/reference/)