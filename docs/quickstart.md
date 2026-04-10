# Quick Start

Get up and running with DQM-ML in minutes.

## Installation

```bash
# Install the core package with basic metrics
pip install dqm-ml-core

# Install all packages (core + images + PyTorch + job orchestration)
pip install dqm-ml-core dqm-ml-job dqm-ml-images dqm-ml-pytorch

# Or install the CLI wrapper
pip install dqm-ml-v2
```

## Using the CLI

Run a data quality check from the command line:

```bash
dqm-ml process -p examples/config/completeness.yaml
```

That's it! The CLI reads a simple YAML configuration file and outputs your metrics.

## Using the Python API

Want more control? Use the Python API directly:

```python
import pandas as pd
from dqm_ml_core import CompletenessProcessor

# Create a sample dataset
df = pd.DataFrame({
    "name": ["Alice", "Bob", None, "Diana"],
    "age": [25, 30, 35, None],
    "score": [0.9, 0.8, 0.7, 0.6]
})

# Create and run the completeness processor
processor = CompletenessProcessor(
    name="my_completeness",
    config={"input_columns": ["name", "age", "score"]}
)

# Get the results
result = processor.compute({})
print(f"Overall completeness: {result['overall_completeness']}")
```

## Using the MetricRunner

For quick exploration in a notebook or script:

```python
import pandas as pd
from dqm_ml_core import CompletenessProcessor, MetricRunner

df = pd.DataFrame({"a": [1, 2, None, 4], "b": [5, None, 7, 8]})
runner = MetricRunner()

results = runner.run(df, [CompletenessProcessor(config={"input_columns": ["a", "b"]})])
print(results)
```

> **Tip:** For interactive exploration, check out our [Jupyter notebook example](https://github.com/Safenai/dqm-ml-workspace/tree/main/examples/multiple_metrics_tests_v2.ipynb).

## Next Steps

- Learn about [available metrics](metrics.md)
- Understand [configuration](configuration.md) options
- Explore [package-specific documentation](dqm-ml-overview.md#packages)