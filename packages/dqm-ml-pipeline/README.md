# DQM-ML Pipeline

This package provides the orchestration engine for DQM-ML V2. It handles the lifecycle of data processing, from loading to metric computation and output writing.

## Key Components

### `DatasetPipeline`

The main orchestrator that:

* Loads the configuration.
* Discovers plugins via entry points.
* Executes the streaming loop.
* Manages memory and I/O efficiency.

### Protocols

* `DataLoader`: A factory for creating data selections (e.g., Parquet, CSV loaders).
* `DataSelection`: Represents a specific subset of data and provides an iterator over batches.
* `OutputWriter`: Persists computed features or metrics to disk.

## Built-in Loaders

* `parquet`: Optimized loading using PyArrow.
* `csv`: Flexibile loading using Pandas.
