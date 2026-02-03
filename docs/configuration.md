# Configuration Guide

DQM-ML V2 pipelines are configured using YAML files. A configuration file defines where the data comes from, which metrics to compute, and where to save the results.

## YAML Structure

A standard configuration file consists of three main sections:

### 1. `dataloaders`

Defines the source of your data. Multiple loaders can be defined.

* `type`: The loader plugin to use (`parquet`, `csv`).
* `path`: Path to the data file or directory.
* `selections`: (Optional) Specific subsets of data to load.

```yaml
dataloaders:
  my_dataset:
    type: parquet
    path: "data/raw_data.parquet"
```

### 2. `metrics_processor`

Defines the metrics or feature extractors to run on the loaded data. Each processor has a unique name and a specific `type`.

```yaml
metrics_processor:
  # Completeness check
  null_check:
    type: completeness
    input_columns: ["col_a", "col_b"]

  # Visual feature extraction
  quality_check:
    type: visual_metrique
    input_columns: ["image_bytes"]
```

### 3. `outputs` (Optional)

Defines where to save generated features (if any).

* `type`: The output writer plugin (`parquet`).
* `path`: Destination file path.
* `columns`: Which columns (original + generated) to include in the output.

```yaml
outputs:
  features_save:
    type: parquet
    path: "outputs/data_with_features.parquet"
    columns: ["sample_id", "m_luminosity", "m_blur_level"]
```

## Complete Example

```yaml
dataloaders:
  train_data:
    type: parquet
    path: "data/train.parquet"

metrics_processor:
  completeness:
    type: completeness
    include_per_column: true

  visual:
    type: visual_metrique
    input_columns: ["img"]

outputs:
  save_results:
    type: parquet
    path: "output/enriched_data.parquet"
    columns: ["sample_id", "m_luminosity"]
```

## Running with Configuration

Use the `process` command with the `-p` (or `--path-config`) flag:

```bash
uv run dqm-ml process -p my_config.yaml
```