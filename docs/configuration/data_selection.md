## 3. Data Selection — Where Data Comes From

The `dataloaders` section defines one or more data sources. Each data source produces one or more **Data Selections** — named subsets that flow into each interface's processors.

### 3.1 Data Loaders

```yaml
dataloaders:
  # Global config for all loaders (optional)
  storage:
    ...
  errors:
    ...

  # List of data sources
  loaders:
    - name: my_training_data
      type: parquet                # or csv
      path: "data/train.parquet"
      batch_size: 5000             # rows per batch (default: 10000)
      id_column: id                # optional row identifier column
      storage:                     # optional — overrides global storage for this loader
        type: local
```

Supported data types:

| Type | Description |
|------|-------------|
| `parquet` | Apache Parquet files — recommended for large datasets |
| `csv` | Comma-separated values — suitable for small datasets |

#### Choosing Your Data Format

| Format | Best For | Limitations |
|--------|----------|-------------|
| **Parquet** | Large datasets, columnar data, analytics | Not human-readable |
| **CSV** | Small datasets, interoperability | Slower, larger files |

**Recommendation**:

- **< 1 GB**: CSV is fine
- **> 1 GB**: Use Parquet for speed and memory efficiency
- **Production pipelines**: Parquet with compression

#### Batch Size Guide

| Data Size | Recommended Batch Size | Why |
|-----------|------------------------|-----|
| **< 100 MB** | 10,000 | Default works well |
| **100 MB – 1 GB** | 10,000 – 50,000 | Balance memory/performance |
| **1 – 10 GB** | 50,000 – 100,000 | Larger batches more efficient |
| **> 10 GB** | 100,000+ | Adjust based on available RAM |

> **Tip:** Start with the default (10,000) and increase if you have plenty of RAM. Smaller batches mean more overhead.

### 3.2 Data Selections

A dataloader creates selections based on three strategies:

**Single Selection (Full Dataset)** — one selection containing all rows:

```yaml
loaders:
  - name: train_data
    type: parquet
    path: "data/train.parquet"
```

→ Creates 1 selection: `train_data`

**Filtered Selection** — narrow rows by a metadata qualifier:

```yaml
loaders:
  - name: wildlife
    type: parquet
    path: "data/images.parquet"
    filters:
      - column: animal           # boolean column — keep only rows where animal=True
        values: [True]
```

→ Creates 1 selection: `wildlife`

> Filters are best for metadata qualifiers like `animal`, `domain` (interior/exterior), or `year`. They narrow the row set before any split is applied.

**Split Selection** — partition rows into named groups for comparison:

```yaml
loaders:
  - name: by_class
    type: parquet
    path: "data/images.parquet"
    split:
      by: class                   # column to split on
      values: [cat, dog, bird]    # create one selection per value (wildcards supported)
      exclude: [horse]            # exclude specific values (wildcards supported)
```

→ Creates 3 selections: `by_class_cat`, `by_class_dog`, `by_class_bird`

> If `values` is omitted, all unique values in the column are used as individual selections.

**Combine filters + split** — narrow then partition:

```yaml
loaders:
  - name: wildlife
    type: parquet
    path: "data/images.parquet"
    filters:
      - column: animal
        values: [True]
    split:
      by: class
      values: [cat, dog]
```

→ Creates 2 selections (`wildlife_cat`, `wildlife_dog`), both containing only `animal=True` rows. This pattern is useful for gap analysis between comparable subsets.

**Selection naming**:

| Configuration | Selection Name(s) |
|--------------|------------------|
| No split | `<loader_name>` |
| With split | `<loader_name>_<value>` |

### 3.3 Sample Path Configuration

Use `sample_path` to provide a base directory for resolving relative file paths in specific columns (e.g., image paths). Processors use this information to resolve and load resources at runtime.

```yaml
loaders:
  - name: source_dataset
    type: parquet
    path: "data/images.parquet"
    sample_path:
      - column: image_path
        prefix: /data/images
      - column: thumbnail
        prefix: /data/thumbnails
```

| Option | Description |
|--------|-------------|
| `column` | Column name containing relative file paths |
| `prefix` | Base directory for resolving relative paths (optional) |

**Path resolution**: When `prefix` is provided, column values (relative paths like `images/001.jpg`) remain unchanged in the batch. Processors read the prefix at runtime and resolve the absolute path when loading resources. The original column values are preserved in output files.

### 3.4 Transform

The `transform` parameter casts a column from its original type to another type. This is useful for enabling metrics on text columns or reducing memory usage.

```yaml
loaders:
  - name: data
    type: parquet
    path: "data/dataset.parquet"
    transform:
      - column: age
        to_type: float32          # cast int → float32
        in_place: false           # keep age, add age_float32
      - column: score
        to_type: float32          # cast float64 → float32 (memory saving)
        in_place: true            # overwrite score in-place
      - column: label
        to_type: categorical      # cast str → categorical
        in_place: true
      - column: is_valid
        to_type: int32            # cast bool → 0/1
        in_place: false
```

When `in_place` is `false` (default), the original column is kept and a new column is added with the name `<original_column>_<to_type>`. When `in_place` is `true`, the original column is overwritten.

Supported type conversions:

| To `int32` | `int64` | `float32` | `float64` | `bool` | `str` | `categorical` |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| int64 | int32 | int32/int64 | int32/int64/float32 | int32/int64 | any | str |
| float32 | float64 | float64 | — | — | — | — |
| bool | bool | bool | bool | — | — | — |

Common use cases:

- `float64 → float32`: Reduce memory usage by 50%; most ML models consume float32
- `int → float64`: Enable normalization and scaling before metrics computation
- `bool → int32`: Convert booleans to 0/1 for numeric metrics
- `str → categorical`: Enable completeness/representativeness/diversity metrics on text columns

### 3.5 Error Handling

Configure how the dataloader handles corrupt or malformed data:

```yaml
dataloaders:
  errors:
    default: silent_fail            # or fail_fast
    max_failure_rate: 0.05          # switch to fail_fast if exceeded
    images:
      on_decode_failure: silent_fail
      on_transform_error: silent_fail
      on_unsupported_format: fail_fast
    tabular:
      on_missing_column: fail_fast
      on_file_not_found: fail_fast
```

| Value | Behavior |
|-------|----------|
| `silent_fail` | Log a warning, emit sentinel (`NaN` / zero-vector / `None`), continue. Increments failure counter. |
| `fail_fast` | Log the error, halt the processor, raise a user-facing exception with sample index, file path, and error details. |

When `silent_fail` is active and the actual failure rate exceeds `max_failure_rate`, DQM-ML switches to `fail_fast` mode automatically.

> Interface-level and processor-level error configs override the dataloader-level settings.
