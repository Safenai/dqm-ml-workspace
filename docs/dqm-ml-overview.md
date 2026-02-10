# DQM-ML V2 Project Overview

DQM-ML V2 is a modular Python framework designed to compute data quality metrics for Machine Learning datasets at scale. It evolves from the original `dqm-ml` library with a focus on a unified API, streaming processing for large datasets, and a decoupled package architecture.

## Architecture

The project is managed as a `uv` workspace with several specialized packages located in the `packages/` directory:

* `dqm-ml-core`: Defines the base API (`DatametricProcessor`) and provides core metrics like Completeness and Representativeness.
* `dqm-ml-job`: Orchestrates the data processing flow. It handles data loading via `DataSelection` and `DataLoader` protocols and executes metric processors in a streaming fashion.
* `dqm-ml-images`: Provides metrics and feature extraction specifically for image data (luminosity, contrast, blur, entropy).
* `dqm-ml-pytorch`: Implements advanced metrics requiring PyTorch, such as Domain Gap (FID, KLMVN, MMD).
* `dqm-ml-v2`: A wrapper package providing the main CLI (`dqm-ml-v2`) and handling optional dependencies.

## Key Technologies

* `uv`: Fast Python package manager and workspace orchestrator.
* `pyarrow`: Primary data format for efficient batch processing and I/O.
* `nox`: Task runner for testing, linting, and documentation building.
* `mkdocs-material`: Documentation framework.

## Building and Running

### Setup

```bash
# Synchronize workspace and install dependencies
uv sync
```

### Execution

The main entry point is the `dqm-ml-v2` CLI:

```bash
# List available metrics and loaders
uv run dqm-ml-v2 list

# Execute a pipeline from a configuration file
uv run dqm-ml-v2 process -p config.yaml
```

### Testing and Quality

```bash
# Run all tests, linting, and type checking
uv run nox

# Run a specific session
uv run nox -s test_dev
uv run nox -s lint
uv run nox -s type_check
```

## Development Conventions

### Metric Implementation

All new metrics should inherit from `dqm_ml_core.api.data_processor.DatametricProcessor` and implement:

* `compute_features()`: for computing sample features used by metrics
* `compute_batch_metric()`: For intermediate statistics.
* `compute()`: For final dataset-level aggregation.
* `compute_delta()`: (Optional) For comparing two datasets.

### Data Loading

Data loaders follow a two-tier abstraction:

1. `DataLoader`: Factory that discovers available `DataSelection`s.
2. `DataSelection`: Handles actual batch iteration for a specific subset of data.

### Coding Style

* `linting`: Managed by `ruff`. Line length is set to 120.
* `typing`: Strict type checking with `mypy`.
* `formatting`: Handled by `ruff format`.
* `flake8`: Strictly adhere to rules, specifically:
