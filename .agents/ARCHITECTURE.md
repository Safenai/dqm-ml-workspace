# Architecture

## Pipeline Data Flow

```
pa.RecordBatch ──(per batch)──→ compute_features → dict[str, pa.Array]
                                           ↓
                                compute_batch_metric → dict[str, pa.Array] (accumulated)
                                           ↓
                                compute → dict[str, Any] (Python scalars)
                                           ↓
                                compute_delta(source, target) → dict[str, Any]
```

`accumulate_flush`: features buffered to 512 MB then written to Parquet.

## Configuration Structure

```yaml
config:
  dataloaders:
    selection_name:
      type: parquet          # entry point name
      path: ...
      batch_size: 10000      # default
      ...
  metrics_processor:
    instance_name:
      type: completeness     # entry point name
      input_columns: [...]   # simple config
      ...                    # or nested config
  outputs:
    metrics:
      type: parquet
      path_pattern: ...
    features:                # optional
      type: parquet
    delta_metrics:           # optional
      type: parquet
```

- `type:` field dispatches to registered entry point.
- Config root key must be `config:` (not `pipeline_config:`).
- `s3_filesystem:` can be `true` (reads env vars) or dict (`access_key`, `secret_key`, `endpoint_override`, `region`).

## Adding a Metric

Extend `DatametricProcessor` (`dqm_ml_core.api.data_processor:16`). Implement three methods:

| Method | Input | Output |
|---|---|---|
| `compute_features` | `pa.RecordBatch` | `dict[str, pa.Array]` — one array per sample |
| `compute_batch_metric` | `dict[str, pa.Array]` | `dict[str, pa.Array]` — scalar per batch |
| `compute` | `dict[str, pa.Array]` (concatenated across batches) | `dict[str, Any]` — final dataset scores |

- Config: accept `input_columns: list[str]` for simple metrics (ref: `completeness.py:35`). Use nested grouped sections for complex ones (ref: `image_embedding.py:47`).
- Register in `[project.entry-points."dqm_ml.metrics"]` → `type = "package.module:ClassName"`.
- ImageEmbedding produces `pa.FixedSizeListArray(pa.float32())` in column `"embedding"`. DomainGap consumes it by checking `isinstance(arr.type, pa.FixedSizeListArray)`.
- DiversityProcessor accumulates `pa.compute.value_counts()` per batch and merges into a `Counter` in `compute()` (`diversity.py:96`).
- `compute_delta` flag is deprecated — do not use.

## Adding a Dataloader

Implement two protocols (`dqm_ml_job.dataloaders.proto:10`):

- `DataSelection`: `bootstrap(columns)`, `get_nb_batches()`, `__iter__()` → yields `pa.RecordBatch`
- `DataLoader`: `get_selections()` → `list[DataSelection]`

Canonical ref: `ParquetDataLoader` at `parquet.py:126`. Supports `batch_size`, `threads`, `filters_dict`, `split_by`, `columns`, `s3_filesystem`.

Register in `[project.entry-points."dqm_ml.dataloaders"]`.

CSV anti-pattern: `PandasDataLoader` (`pandas.py:76`) loads the entire file into memory as a single batch — no streaming, no filtering, no S3. Only use for ad-hoc exploration.

## Adding an Output Writer

Implement `OutputWriter` protocol (`dqm_ml_job.outputwriter.__init__:17`):
- `columns: list[str]`, `name: str`
- `write_metrics_dict(data: dict[str, pa.Array]) → None`
- `write_table(table: pa.Table) → None`

Canonical ref: `ParquetOutputWriter` at `outputwriter/parquet.py:20`. Supports `path_pattern` with `{}` for partitioned writes, accumulate mode when `{}` absent.

Register in `[project.entry-points."dqm_ml.outputwriter"]`.

## Anti-Patterns (see also [CONFIGURATION_ISSUES.md](../CONFIGURATION_ISSUES.md))

- Mixing flat (`input_columns`) and nested (`DATA.image_column`) config styles.
- Case mismatch: code reads `MODEL`/`INFER` but YAML uses `model`/`infer`.
- Stale keys: `compute_delta` (parsed, never used), `output_metrics` (YAML key, code has `# TODO` and never reads it).
- Duplicated `s3_filesystem` in every component — use `dqm_ml_job.utils.s3.get_s3_filesystem()`.
- `output_columns` / `output_metrics` / `output_features` are synonyms — use `output_metrics`.

## File:line Index

| Pattern | File |
|---|---|
| DatametricProcessor base class | `packages/dqm-ml-core/src/dqm_ml_core/api/data_processor.py:16` |
| Simple metric (Completeness) | `packages/dqm-ml-core/src/dqm_ml_core/metrics/completeness.py:35` |
| Value-count streaming (Diversity) | `packages/dqm-ml-core/src/dqm_ml_core/metrics/diversity.py:96` |
| Complex metric (ImageEmbedding) | `packages/dqm-ml-pytorch/src/dqm_ml_pytorch/image_embedding.py:47` |
| ParquetDataLoader | `packages/dqm-ml-job/src/dqm_ml_job/dataloaders/parquet.py:126` |
| PandasDataLoader (CSV bridge) | `packages/dqm-ml-job/src/dqm_ml_job/dataloaders/pandas.py:76` |
| ParquetOutputWriter | `packages/dqm-ml-job/src/dqm_ml_job/outputwriter/parquet.py:20` |
| S3 utility | `packages/dqm-ml-job/src/dqm_ml_job/utils/s3.py:8` |
| Integration test structure | `tests/integration/test_completeness.py:15` |
| Entry points (dataloaders) | `packages/dqm-ml-job/pyproject.toml:32` |
| Entry points (output writers) | `packages/dqm-ml-job/pyproject.toml:37` |
