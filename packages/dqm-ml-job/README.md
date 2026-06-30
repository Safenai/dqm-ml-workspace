# DQM-ML Job

Orchestration engine for DQM-ML V2. Handles data loading, processing, and output writing.

## Installation

```bash
pip install dqm-ml-job
```

> **Note:** `dqm-ml-job` handles data loading and orchestration. To compute **Metrics**, you also need at least one of: `dqm-ml-core`, `dqm-ml-images`, or `dqm-ml-pytorch` (see Dependencies below).

## Quick Start

### Using Python

```python
from dqm_ml_job.cli import execute

# Execute a data quality job from a YAML config
execute(["-p", "config.yaml"])
```

### Using Python Module

```bash
python -m dqm_ml_job.cli -p config.yaml
```

Example `config.yaml`:

```yaml
metrics:
  processors:
    - name: completeness
      type: completeness
      columns:
        input: [col_a, col_b]

dataloaders:
  loaders:
    - name: my_data
      type: parquet
      path: data/train.parquet
```

## Dependencies

DQM-ML is modular — `dqm-ml-job` provides the orchestration, but you need additional packages to compute actual metrics:

```bash
# For Completeness and Representativeness
pip install dqm-ml-job dqm-ml-core

# For Visual Features
pip install dqm-ml-job dqm-ml-images

# For Domain Gap
pip install dqm-ml-job dqm-ml-pytorch

# All metrics
pip install dqm-ml-job dqm-ml-core dqm-ml-images dqm-ml-pytorch
```

## Key Components

### DatasetPipeline

The main orchestrator that:

- Loads the configuration
- Discovers plugins via entry points
- Executes the streaming loop
- Manages memory and I/O efficiency

### Protocols

| Protocol | Description |
|----------|-------------|
| **DataLoader** | Factory for creating **Data Selections** (e.g., Parquet, CSV loaders) |
| **DataSelection** | Represents a specific **Data Selection** and provides an iterator over **Batches** |
| **OutputWriter** | Persists computed **Features** or **Metrics** to disk |

## Built-in Loaders

| Loader | Description |
|--------|-------------|
| **parquet** | Optimized loading using PyArrow |
| **csv** | Flexible loading using Pandas |

## See Also

- [Formal and Core Concepts](https://safenai.github.io/dqm-ml-workspace/docs/formal_concepts.md) for definitions of **Data Selection**, **Processor**, **Batch**, and related terminology.
- [Configuration Guide](https://safenai.github.io/dqm-ml-workspace/docs/configuration/)
- [Architecture Documentation](https://safenai.github.io/dqm-ml-workspace/docs/dqm-ml-v2/)