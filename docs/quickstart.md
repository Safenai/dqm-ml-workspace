# Quick Start

Get up and running with DQM-ML in minutes.

> **See also:** [Concepts](formal_concepts.md) for definitions of **Sample**, **Metric**, **Processor**, and related terminology used throughout this page.

> **What's new?** See the [Release Notes](./RELEASE.md) for the latest features and improvements.

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

Here's a complete example below:

**1. Create a data file (`data.csv`):**
```csv
name,age,score
Alice,25,0.9
Bob,30,0.8,35,0.7
Diana,,0.6
```

**2. Create a config file (`config.yaml`):**
```yaml
metrics:
  outputs:
    path: output_metrics.parquet
  processors:
    - name: completeness
      type: completeness
      columns:
        input: [name, age, score]

dataloaders:
  loaders:
    - name: my_data
      type: csv
      path: data.csv
```

**3. Run the pipeline:**
```bash
dqm-ml process -p config.yaml
```

**4. Read the results:**
```bash
python -c "import pandas as pd; print(pd.read_parquet('output_metrics.parquet').to_string())"
```

> **Note:** Example files are available in [`examples/getting_started/`](../examples/getting_started/) in the repository.

### Python API

Want more control? Use the Python API directly:

```python
import pandas as pd
import pyarrow as pa
from dqm_ml_core import CompletenessProcessor

# Create a sample dataset
df = pd.DataFrame({
    "name": ["Alice", "Bob", None, "Diana"],
    "age": [25, 30, 35, None],
    "score": [0.9, 0.8, 0.7, 0.6]
})

# Create the completeness processor
processor = CompletenessProcessor(
    name="my_completeness",
    config={"columns": {"input": ["name", "age", "score"]}}
)

# Run the full pipeline: features -> batch metrics -> aggregated results
batch = pa.RecordBatch.from_pandas(df)
features = processor.compute_features(batch)
batch_metrics = processor.compute_batch_metric(features)
result = processor.compute(batch_metrics)
print(f"Overall completeness: {result['completeness_overall']}")
```

### ProcessorRunner (Interactive)

For quick exploration in a notebook or script:

```python
import pandas as pd
from dqm_ml_core import CompletenessProcessor, ProcessorRunner

df = pd.DataFrame({"a": [1, 2, None, 4], "b": [5, None, 7, 8]})
runner = ProcessorRunner()

results = runner.run(df, [CompletenessProcessor(config={"columns": {"input": ["a", "b"]}})])
print(results)
```

> **Tip:** For interactive exploration, check out our [Jupyter notebook example](../examples/notebooks/multiple_metrics_tests_v2.ipynb).

## Next Steps

- Learn about [available metrics](metrics.md)
- Understand [configuration](configuration/overview.md) options
- Read [CLI Reference](cli.md) for command details
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