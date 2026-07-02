# Architecture

## Pipeline Data Flow

```
pa.RecordBatch ──(per batch)──→ extract_features → dict[str, pa.Array]
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
  features:
    processors:
      instance_name:
        type: image_features     # entry point name
        columns:
          input: [...]
        ...
  metrics:
    processors:
      instance_name:
        type: completeness       # entry point name
        columns:
          input: [...]
        ...
  gap:
    processors:
      instance_name:
        type: domain_gap         # entry point name
        columns:
          input: [...]
        distance:
          metric: mmd_linear
        ...
  outputs:
    features:
      type: parquet
      path_pattern: "output/features_{}.parquet"
    metrics:
      type: parquet
      path_pattern: "output/metrics_{}.parquet"
    gap:
      type: parquet
      path_pattern: "output/gap_{}-{}.parquet"
```

- `type:` field dispatches to registered entry point.
- Config root key must be `config:` (not `pipeline_config:`).
- `s3_filesystem:` can be `true` (reads env vars) or dict (`access_key`, `secret_key`, `endpoint_override`, `region`).
- Three processor interfaces: `features`, `metrics`, `gap` — each with own `processors` list.

## Adding a Features Processor

Extend `FeaturesProcessor` (`dqm_ml_core.api.features_processor:16`). Implement:

| Method | Input | Output | Required? |
|---|---|---|---|
| `generated_features()` | — | `list[str]` — output column names | Yes |
| `compute_features(batch, prev_features)` | `pa.RecordBatch`, `dict[str, pa.Array]` | `dict[str, pa.Array]` — new feature columns | Yes |
| `needed_columns()` | — | `list[str]` — input columns needed | No (default: `input_columns`) |

- Config: `columns.input` (list[str] or None), `columns.exclude`, `columns.rename/prefix/suffix`
- Register in `[project.entry-points."dqm_ml.features"]` → `type = "package.module:ClassName"`
- Output columns resolved via `_resolve_output_name(base_name)` using rename/prefix/suffix
- Ref: `VisualFeaturesProcessor` (`dqm_ml_images.visual_features`), `ImageEmbeddingProcessor` (`dqm_ml_pytorch.image_embedding`)

## Adding a Metrics Processor

Extend `MetricsProcessor` (`dqm_ml_core.api.metrics_processor:16`). Implement:

| Method | Input | Output | Required? |
|---|---|---|---|
| `generated_metrics()` | — | `list[str]` — output metric names | Yes |
| `extract_columns(batch, prev_features)` | `pa.RecordBatch`, `dict[str, pa.Array]` | `dict[str, pa.Array]` — selected columns | No (default in base) |
| `compute_batch_metric(features)` | `dict[str, pa.Array]` | `dict[str, pa.Array]` — batch stats | Yes |
| `compute(batch_metrics)` | `dict[str, pa.Array]` (concat across batches) | `dict[str, Any]` — final scores | Yes |

- Config: `columns.input`, `columns.exclude`, plus metric-specific params
- Register in `[project.entry-points."dqm_ml.metrics"]` → `type = "package.module:ClassName"`
- `extract_columns` handles `on_missing_column` (fail_fast/silent_fail) and wildcard resolution
- Diversity: accumulates `pa.compute.value_counts()` per batch, merges `Counter` in `compute()`
- Ref: `CompletenessProcessor`, `RepresentativenessProcessor`, `DiversityProcessor` (`dqm_ml_core.metrics`)

## Adding a Gap Processor

Extend `GapProcessor` (`dqm_ml_core.api.gap_processor:16`). Implement:

| Method | Input | Output | Required? |
|---|---|---|---|
| `extract_features(batch, prev_features)` | `pa.RecordBatch`, `dict[str, pa.Array]` | `dict[str, pa.Array]` — retrieve embeddings | Yes |
| `compute_batch_metric(features)` | `dict[str, pa.Array]` | `dict[str, pa.Array]` — batch stats | Yes |
| `compute(batch_metrics)` | `dict[str, pa.Array]` (concat) | `dict[str, Any]` — final scores | Yes |
| `compute_delta(source, target)` | `dict`, `dict` | `dict[str, Any]` — pairwise distances | Yes |

- Config: `columns.input` (resolved against `batch.schema.names + prev_features.keys()`), `columns.exclude`
- Register in `[project.entry-points."dqm_ml.gap"]` → `type = "package.module:ClassName"`
- `extract_features` searches both batch columns and previously generated features
- Embeddings: expects `pa.FixedSizeListArray(pa.float32())` in input column
- Ref: `DomainGapProcessor` (`dqm_ml_pytorch.domain_gap`)

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
| Processor base class | `packages/dqm-ml-core/src/dqm_ml_core/api/processor.py:16` |
| FeaturesProcessor base | `packages/dqm-ml-core/src/dqm_ml_core/api/features_processor.py:16` |
| MetricsProcessor base | `packages/dqm-ml-core/src/dqm_ml_core/api/metrics_processor.py:16` |
| GapProcessor base | `packages/dqm-ml-core/src/dqm_ml_core/api/gap_processor.py:16` |
| Features: VisualFeaturesProcessor | `packages/dqm-ml-images/src/dqm_ml_images/visual_features.py` |
| Features: ImageEmbeddingProcessor | `packages/dqm-ml-pytorch/src/dqm_ml_pytorch/image_embedding.py` |
| Metrics: CompletenessProcessor | `packages/dqm-ml-core/src/dqm_ml_core/metrics/completeness.py` |
| Metrics: RepresentativenessProcessor | `packages/dqm-ml-core/src/dqm_ml_core/metrics/representativeness.py` |
| Metrics: DiversityProcessor | `packages/dqm-ml-core/src/dqm_ml_core/metrics/diversity.py` |
| Gap: DomainGapProcessor | `packages/dqm-ml-pytorch/src/dqm_ml_pytorch/domain_gap.py` |
| ParquetDataLoader | `packages/dqm-ml-job/src/dqm_ml_job/dataloaders/parquet.py:126` |
| PandasDataLoader (CSV bridge) | `packages/dqm-ml-job/src/dqm_ml_job/dataloaders/pandas.py:76` |
| ParquetOutputWriter | `packages/dqm-ml-job/src/dqm_ml_job/outputwriter/parquet.py:20` |
| S3 utility | `packages/dqm-ml-job/src/dqm_ml_job/utils/s3.py:8` |
| Integration test structure | `tests/integration/test_completeness.py:15` |
| Entry points (dataloaders) | `packages/dqm-ml-job/pyproject.toml:32` |
| Entry points (output writers) | `packages/dqm-ml-job/pyproject.toml:37` |
| Entry points (metrics) | `packages/dqm-ml-core/pyproject.toml` |
| Entry points (features) | `packages/dqm-ml-images/pyproject.toml`, `packages/dqm-ml-pytorch/pyproject.toml` |
| Entry points (gap) | `packages/dqm-ml-pytorch/pyproject.toml` |
