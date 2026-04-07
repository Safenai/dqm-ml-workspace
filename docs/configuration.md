# Configuration Guide

DQM-ML pipelines are configured using **YAML** files. This guide explains how to write configuration files that tell DQM-ML where to find data, which metrics to compute, and where to save results.

Don't know YAML? No problem! It's just a way to write structured data. Think of it like a structured to-do list:

```yaml
# This is a comment (ignored by DQM-ML)
dataloaders:           # "Here's where I define data sources"
  my_data:             # "Name this data source"
    type: parquet      # "Use the parquet reader"
    path: "data.csv"   # "Look for the file here"
```

## YAML Basics (Quick Refresher)

| YAML | Meaning |
|------|---------|
| `key: value` | Assign a value |
| `- item` | List item |
| `# comment` | Comment (ignored) |
| `"string"` | Explicit string (use quotes if value has special chars) |

## Configuration Structure

A DQM-ML config has three main sections:

### 1. `dataloaders` - Where's the data?

Defines where your data comes from. You can define multiple data sources.

```yaml
dataloaders:
  # Give your data a memorable name
  my_training_data:
    # Use "parquet" for .parquet files, "csv" for .csv files
    type: parquet
    # Path to the file (relative paths work too!)
    path: "data/train.parquet"
    # Optional: batch size for processing (default: 10000)
    batch_size: 5000
```

### 2. `metrics_processor` - What to compute

Defines which metrics or feature extractors to run on your data.

```yaml
metrics_processor:
  # Give each metric a unique name
  null_check:
    # The type of metric (from the registry)
    type: completeness
    # Which columns to analyze
    input_columns: ["age", "income", "email"]
    # Include per-column scores?
    include_per_column: true
    # Include overall score?
    include_overall: true

  image_quality:
    type: visual_metric
    input_columns: ["image_bytes"]
```

### 3. `outputs` - Where to save results

Defines where to save generated features (optional - if omitted, only metrics are saved).

```yaml
outputs:
  feature_output:
    type: parquet
    # Where to save
    path: "output/enriched_data.parquet"
    # Which columns to include (original + generated)
    columns: ["sample_id", "m_luminosity", "m_blur_level"]
```

## Complete Example

Here's a full configuration file that does something useful:

```yaml
dataloaders:
  train_data:
    type: parquet
    path: "data/train.parquet"
    batch_size: 10000

metrics_processor:
  # Check for missing values
  completeness:
    type: completeness
    input_columns: ["age", "income", "zip_code"]
    include_per_column: true
    include_overall: true

  # Extract image features
  visual:
    type: visual_metric
    input_columns: ["image_data"]
    grayscale: true

outputs:
  save_features:
    type: parquet
    path: "output/train_features.parquet"
    columns: ["sample_id", "m_luminosity", "m_contrast"]
```

## Running Your Configuration

Once you've written your config file, run it with:

```bash
uv run dqm-ml process -p my_config.yaml
```

That's it! DQM-ML will:
1. Load your data in batches
2. Run the metrics you specified
3. Save results to the output location

## Common Patterns

### Running Multiple Metrics

```yaml
metrics_processor:
  completeness_check:
    type: completeness
    input_columns: ["col_a", "col_b"]
  
  representativeness_check:
    type: representativeness
    input_columns: ["feature_x", "feature_y"]
    distribution: "normal"
```

### Processing Multiple Datasets

```yaml
dataloaders:
  train_data:
    type: parquet
    path: "data/train.parquet"
  
  test_data:
    type: parquet
    path: "data/test.parquet"

metrics_processor:
  train_completeness:
    type: completeness
    input_columns: ["col_a"]
  
  test_completeness:
    type: completeness
    input_columns: ["col_a"]
```
