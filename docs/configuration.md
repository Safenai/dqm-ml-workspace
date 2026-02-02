# Pipeline Configuration

DQM-ML uses a YAML-based configuration to define the execution pipeline. A typical configuration consists of three main sections: `dataloaders`, `metrics_processor`, and `outputs`.

## General Structure

```yaml
pipeline_config:
  dataloaders:
    # Definition of data sources
  metrics_processor:
    # Definition of metrics to compute
  outputs:
    # Definition of where to store results
```

## Dataloaders

Defines how to read the input data.

### Parquet
```yaml
dataloaders:
  my_loader:
    type: parquet
    path: "data/my_dataset.parquet"
    # Optional: filters, batch_size, etc.
```

### CSV
```yaml
dataloaders:
  my_loader:
    type: csv
    path: "data/my_dataset.csv"
```

## Metrics Processor

Defines which metrics to apply to which columns.

```yaml
metrics_processor:
  completeness_check:
    type: completeness
    input_columns: ["age", "income"]
```

## Outputs

Defines where and how to write the computed metrics and features.

```yaml
outputs:
  metrics:
    type: parquet
    path: "output/metrics.parquet"
  features:
    type: parquet
    path: "output/features.parquet"
  delta_metrics:
    type: parquet
    path: "output/features.parquet"    
```
