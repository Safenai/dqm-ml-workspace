# Data Loaders

Define where your data comes from and how to create data selections.

## Dataloaders Section

The `dataloaders` section in your config defines data sources:

```yaml
dataloaders:
  my_data:
    type: parquet
    path: data/train.parquet
```

## Data Selections

A data selection defines a subset of your data to analyze. DQM-ML can create multiple selections from a single dataloader configuration.

### Single Selection (Full Dataset)

By default, a dataloader creates one selection for the entire dataset:

```yaml
dataloaders:
  train_data:
    type: parquet
    path: data/train.parquet
```

- Creates 1 selection: `train_data`

### Filtered Selection

Use `filter` to select rows matching specific values:

```yaml
dataloaders:
  birds:
    type: parquet
    path: data/images.parquet
    filter:
      class: bird
```

- Creates 1 selection: `birds` (only rows where class = bird)

### Split Selection

Use `split_by` to create multiple selections based on column values:

```yaml
dataloaders:
  coco_classes:
    type: parquet
    path: data/images.parquet
    split_by: class
    split_values: [dog, cat, bird, elephant]
```

- Creates 4 selections: `coco_classes_dog`, `coco_classes_cat`, `coco_classes_bird`, `coco_classes_elephant`

If `split_values` is omitted, all unique values in the column are used.

### Selection Names

Selection names identify data in metrics output:

| Configuration | Selection Name(s) |
|--------------|------------------|
| No split | `<dataloader_name>` |
| With split | `<dataloader_name>_<value>` |

## Supported Data Types

| Type | Description |
|------|-------------|
| `parquet` | Apache Parquet files |
| `csv` | Comma-separated values |
| `proto` | Protocol Buffer streams |

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `type` | Data source type | Required |
| `path` | File path | Required |
| `batch_size` | Rows per batch | 10000 |
| `memory_limit` | Max memory per batch | 1GB |
| `threads` | Parallel threads | 1 |
| `filter` | Row filter (dict) | None |
| `split_by` | Column to split by | None |
| `split_values` | Values to create selections | All unique |

## Examples

### Filtered Data

```yaml
dataloaders:
  train_2023:
    type: parquet
    path: data/train.parquet
    filter:
      year: 2023
```

### Split by Column

```yaml
dataloaders:
  by_category:
    type: parquet
    path: data/products.parquet
    split_by: category
```

### Multiple Dataloaders

```yaml
dataloaders:
  train_data:
    type: parquet
    path: data/train.parquet
  
  test_data:
    type: parquet
    path: data/test.parquet
```

## Related Pages

- [Configuration](configuration.md) - Main configuration guide
- [Metrics](metrics.md) - Available metrics
- [CLI Reference](cli.md) - Command-line usage