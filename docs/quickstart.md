# Quick Start

Get up and running with DQM-ML in minutes.

## Installation

```bash
# Install the CLI wrapper with core metrics
pip install dqm-ml

# Or install all packages (core + images + PyTorch + job orchestration)
pip install dqm-ml-core dqm-ml-job dqm-ml-images dqm-ml-pytorch

# Install with extras
pip install "dqm-ml[all]"         # Everything
pip install "dqm-ml[job]"         # CLI + pipeline
pip install "dqm-ml[notebooks]"   # Jupyter support
```

## Quick Usage

### CLI

Run a data quality pipeline from a configuration file:

```bash
dqm-ml process -p config.yaml
```

That's it! The CLI reads a simple YAML configuration file and outputs your metrics.

### CLI Example with Config File

Here's a complete example matching the Python API below:

**1. Create a data file (`data.csv`):**
```csv
name,age,score
Alice,25,0.9
Bob,30,0.8,35,0.7
Diana,,0.6
```

**2. Create a config file (`config.yaml`):**
```yaml
config:
  dataloaders:
    my_data:
      type: csv
      path: ./data.csv

  metrics_processor:
    completeness:
      type: completeness
      input_columns: [name, age, score]

  outputs:
    metrics:
      type: parquet
      path_pattern: output_metrics.parquet
      columns: []
```

**3. Run the pipeline:**
```bash
dqm-ml process -p config.yaml
```

> **Note:** Example files are available in `tests/fixtures/getting_started/` in the repository.

### Python API

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

### MetricRunner (Interactive)

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

## Troubleshooting

**"Command not found" after pip install?**

```bash
# Ensure you're in the right environment
pip show dqm-ml

# Or use python -m module
python -m dqm_ml --help
```

**Missing dependencies?**

```bash
# Install with extras
pip install "dqm-ml[all]"
```