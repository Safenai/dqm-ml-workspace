# DQM-ML Core

Core package for DQM-ML V2 providing the foundational API and standard metrics for data quality assessment.

## Installation

```bash
pip install dqm-ml-core
```

> **Note:** `dqm-ml-core` provides metric processors only — no CLI or job orchestration. Use directly via Python or with `dqm-ml-job` for YAML config execution.

## Quick Start

### Completeness Example

```python
from dqm_ml_core import CompletenessProcessor

processor = CompletenessProcessor(
    name="my_check",
    config={"input_columns": ["col_a", "col_b"]}
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
        "input_columns": ["feature"],
        "distribution": "normal",
        "metrics": ["chi-square", "kolmogorov-smirnov"],
        "distribution_params": {"mean": 0.0, "std": 1.0}
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

## Key Concepts

### DatametricProcessor

The base class for all metrics and feature extractors. It supports a streaming architecture by splitting computation into two phases:

1. **Batch Level**: `compute_batch_metric()` updates intermediate statistics for a single chunk of data.
2. **Dataset Level**: `compute()` aggregates these statistics into final scores.

## Included Metrics

| Metric | Description |
|--------|-------------|
| **Completeness** | Analyzes null/missing values in your dataset |
| **Representativeness** | Statistical distribution analysis (Chi-Square, KS, Shannon Entropy, GRTE) |

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

- [Metrics Documentation](https://safenai.github.io/dqm-ml-workspace/docs/metrics/)
- [API Reference](https://safenai.github.io/dqm-ml-workspace/reference/)