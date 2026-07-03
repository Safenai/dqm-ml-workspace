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
features:
  processors:
    - name: image_quality
      type: image_features
      columns:
        input: ["image_data"]
      features: [luminosity, contrast, blur, entropy]
      grayscale: true

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

| Interface | Package | Entry Point Group |
|-----------|---------|-------------------|
| **Features** | `dqm-ml-images` | `dqm_ml.features` |
| **Features (Embeddings)** | `dqm-ml-pytorch` | `dqm_ml.features` |
| **Metrics** | `dqm-ml-core` | `dqm_ml.metrics` |
| **Gap** | `dqm-ml-pytorch` | `dqm_ml.gap` |

```bash
# For Metrics (Completeness, Representativeness, Diversity)
pip install dqm-ml-job dqm-ml-core

# For Visual Features
pip install dqm-ml-job dqm-ml-images

# For Image Embeddings + Domain Gap
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

## Adding a Custom DataLoader

A DataLoader discovers available selections from a data source. Implement the `DataLoader` protocol and a companion `DataSelection` class.

### Protocol Overview

`DataLoader` (`dqm_ml_job/dataloaders/proto.py`):

```python
class DataLoader(Protocol):
    def get_selections(self) -> list[DataSelection]: ...
```

`DataSelection` (`dqm_ml_job/dataloaders/proto.py`):

```python
class DataSelection(Protocol):
    name: str

    def bootstrap(self, columns_list: list[str]) -> None: ...
    def get_nb_batches(self) -> int: ...
    def __iter__(self) -> Any: ...
```

### Example: JSON Lines Loader

Create `dqm_ml_job/dataloaders/jsonl.py`:

```python
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
from dqm_ml_job.dataloaders.proto import DataSelection

logger = logging.getLogger(__name__)


class JsonLinesDataSelection(DataSelection):
    def __init__(self, name: str, path: str, batch_size: int = 10_000):
        self.name = name
        self.path = path
        self.batch_size = batch_size
        self._lines: list[dict[str, Any]] | None = None

    def bootstrap(self, columns_list: list[str]) -> None:
        with open(self.path) as f:
            self._lines = [json.loads(line) for line in f]

    def get_nb_batches(self) -> int:
        n = len(self._lines) if self._lines else 0
        return (n // self.batch_size) + (1 if n % self.batch_size else 0)

    def __iter__(self) -> Any:
        if self._lines is None:
            return
        for i in range(0, len(self._lines), self.batch_size):
            yield pa.RecordBatch.from_pylist(self._lines[i:i + self.batch_size])


class JsonLinesDataLoader:
    type: str = "jsonl"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        config = config or {}
        self.name = name
        self.path = config["path"]
        self.batch_size = config.get("batch_size", 10_000)

    def get_selections(self) -> list[DataSelection]:
        return [JsonLinesDataSelection(self.name, self.path, self.batch_size)]
```

### Registration

Add to the registry in `dqm_ml_job/dataloaders/__init__.py`:

```python
from dqm_ml_job.dataloaders.jsonl import JsonLinesDataLoader

dqml_dataloaders_registry["jsonl"] = JsonLinesDataLoader
```

For external packages, register via entry points in `pyproject.toml`:

```toml
[project.entry-points."dqm_ml.dataloaders"]
jsonl = "my_package.dataloaders:JsonLinesDataLoader"
```

## Adding a Custom OutputWriter

An OutputWriter persists computed features or metrics to a storage backend.

### Protocol Overview

`OutputWriter` (`dqm_ml_job/outputwriter/__init__.py`):

```python
class OutputWriter(Protocol):
    columns: list[str]
    name: str
    add_dataloader_column: bool
    dataloader_column_name: str

    def write_metrics_dict(self, metrics_dict: dict[str, dict[str, Any]]) -> None: ...
    def write_table(self, name: str, table: Any, part_index: int | None = None) -> None: ...
```

### Example: CSV Output Writer

Create `dqm_ml_job/outputwriter/csv.py`:

```python
import csv
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
from dqm_ml_core.models.global_ import StorageConfig
from dqm_ml_core.models.outputs import ParquetOutputConfig

logger = logging.getLogger(__name__)


class CsvOutputWriter:
    def __init__(self, name: str, config: dict[str, Any] | None = None):
        cfg = ParquetOutputConfig.model_validate(config or {})  # reuse path_pattern, columns
        self.name = name
        self.path_pattern = cfg.path_pattern
        self.columns = list(cfg.columns)
        self.add_dataloader_column = cfg.add_dataloader_column
        self.dataloader_column_name = cfg.dataloader_column_name

    def write_metrics_dict(self, metrics_dict: dict[str, dict[str, Any]]) -> None:
        pass  # simplified — write all rows per selection

    def write_table(self, name: str, table: Any, part_index: int | None = None) -> None:
        if isinstance(table, dict):
            table = pa.table(table)
        if isinstance(table, pa.Table):
            df = table.to_pandas()
            path = Path(self.path_pattern.format(name=name))
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            logger.info(f"Wrote CSV to {path}")
```

### Registration

Add to `dqm_ml_job/outputwriter/__init__.py`:

```python
from dqm_ml_job.outputwriter.csv import CsvOutputWriter

dqml_outputs_registry["csv"] = CsvOutputWriter
```

For external packages, use entry points:

```toml
[project.entry-points."dqm_ml.outputwriter"]
csv = "my_package.outputwriters:CsvOutputWriter"
```

## Built-in Loaders

| Loader | Description |
|--------|-------------|
| **parquet** | Optimized loading using PyArrow |
| **csv** | Flexible loading using Pandas |

## See Also

- [Formal and Core Concepts](https://safenai.github.io/dqm-ml-workspace/docs/formal_concepts.md) for definitions of **Data Selection**, **Processor**, **Batch**, and related terminology.
- [Configuration Guide](https://safenai.github.io/dqm-ml-workspace/docs/configuration/)
- [Architecture Documentation](https://safenai.github.io/dqm-ml-workspace/docs/dqm-ml-v2/)