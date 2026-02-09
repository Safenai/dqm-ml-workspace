# Contributing to DQM-ML

DQM-ML is designed to be easily extensible. You can contribute by adding your own metrics, fixing bugs, or improving documentation.

## Development Environment Setup

We rely on `uv` for fast development and workspace management.

### 1. Prerequisites

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install git-lfs for large test files
sudo apt-get install git-lfs
git lfs pull

# Initialize submodules (legacy dqm-ml)
git submodule update --init --recursive
```

### 2. Workspace Initialization

Synchronize the workspace and install all dependencies:

```bash
uv sync
```

### 3. Quality Standards

We enforce strict quality checks via `nox`. Please ensure all checks pass before submitting a PR.

* `uv run nox -s lint`: Check for style issues.
* `uv run nox -s fmt`: Automatically format code.
* `uv run nox -s type_check`: Static type analysis with MyPy.
* `uv run nox -s test`: Run all tests with coverage.

## Steps to Add a Metric

### 1. Inherit from `DatametricProcessor`

Create a new class in your package that inherits from `dqm_ml_core.api.data_processor.DatametricProcessor`.

```python
from dqm_ml_core.api.data_processor import DatametricProcessor
import pyarrow as pa

class MyNewMetric(DatametricProcessor):
    def compute_features(self, batch, prev_features=None):
        # Optional: compute per-sample features
        return {}

    def compute_batch_metric(self, features):
        # Mandatory: Compute intermediate statistics for the batch
        return {}

    def compute(self, batch_metrics=None):
        # Mandatory: Compute final dataset-level metric
        return {}
```

### 2. Register via Entry Points

Add your metric to your package's `pyproject.toml` to make it discoverable by the registry.

```toml
[project.entry-points."dqm_ml.metrics"]
my_new_metric = "my_package:MyNewMetric"
```

### 3. Testing

Add a unit test for your metric and a configuration example in the `tests/` directory.

## Best Practices

* **Streaming Friendly**: Ensure your `compute_batch_metric` only computes what is necessary for the final aggregation to keep memory usage low.
* **Type Safety**: Use PyArrow for data handling to ensure compatibility with the rest of the pipeline.
* **Documentation**: Provide a docstring for your class explaining what the metric measures and its configuration parameters.
