# DQM-ML Core

Core package for DQM-ML V2 providing the foundational API and standard metrics for data quality assessment.

## Installation

```bash
pip install dqm-ml-core
```

> **Note:** `dqm-ml-core` provides **Metrics Processors** only — no CLI or job orchestration. Use directly via Python or with `dqm-ml-job` for YAML config execution.

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

### Three Processor Interfaces

DQM-ML V2 defines three distinct processor interfaces, each with its own base class:

| Interface | Base Class | Purpose |
|-----------|------------|---------|
| **Metrics** | `MetricsProcessor` | Compute aggregated metric scores from data (Completeness, Representativeness, Diversity) |
| **Features** | `FeaturesProcessor` | Extract feature columns from data (Visual Features, Embeddings) |
| **Gap** | `GapProcessor` | Compute pairwise distances between selections (Domain Gap) |

All three inherit from a common `Processor` base class (`dqm_ml_core.api.processor:16`) which provides:
- `__init__`, `_check_failure_rate`, `_check_image_fail_fast`, `needed_columns()`, `reset()`

#### MetricsProcessor

Extends `Processor`. Implement:
- `generated_metrics()` → `list[str]` — output metric names
- `extract_columns(batch, prev_features)` → `dict[str, pa.Array]` — select columns (optional, default in base)
- `compute_batch_metric(features)` → `dict[str, pa.Array]` — batch statistics
- `compute(batch_metrics)` → `dict[str, Any]` — final scores

#### FeaturesProcessor

Extends `Processor`. Implement:
- `generated_features()` → `list[str]` — output feature column names
- `compute_features(batch, prev_features)` → `dict[str, pa.Array]` — new feature columns
- `needed_columns()` → `list[str]` — input columns needed (optional, default: `input_columns`)

#### GapProcessor

Extends `Processor`. Implement:
- `extract_features(batch, prev_features)` → `dict[str, pa.Array]` — retrieve embeddings
- `compute_batch_metric(features)` → `dict[str, pa.Array]` — batch statistics
- `compute(batch_metrics)` → `dict[str, Any]` — final scores
- `compute_delta(source, target)` → `dict[str, Any]` — pairwise distances

## Included Metrics

| Metric | Description |
|--------|-------------|
| **Completeness** | Analyzes null/missing values in your dataset |
| **Representativeness** | Statistical distribution analysis (Chi-Square, KS, Shannon Entropy, GRTE) |
| **Diversity** | Measures category distribution spread (Simpson, Gini-Simpson, Shannon, Richness) |

## For Developers

To create a new **Metrics Processor**:

1. Subclass `dqm_ml_core.api.metrics_processor.MetricsProcessor`.
2. Implement `generated_metrics()`, `extract_columns()` (optional), `compute_batch_metric()`, and `compute()`.
3. Register in `[project.entry-points."dqm_ml.metrics"]` in `pyproject.toml`.

To create a **Features Processor** or **Gap Processor**, use the respective base classes in `dqm_ml_core.api.features_processor` and `dqm_ml_core.api.gap_processor`.

Reference implementations:
- `CompletenessProcessor` — simple streaming metric
- `RepresentativenessProcessor` — statistical tests
- `DiversityProcessor` — value-count accumulation

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