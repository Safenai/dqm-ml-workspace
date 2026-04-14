# Metrics Computation

How metrics are computed in DQM-ML based on data selections.

## Metrics Computation Modes

Metrics are computed in two modes depending on the number of selections:

### Per-Selection Metrics

When there's a single selection, metrics compute statistics on that data:

```yaml
metrics_processor:
  completeness:
    type: completeness
```

**Output** (one row per selection):

| selection | completeness_overall |
|----------|---------------------|
| train_data | 0.95 |

### Delta Metrics (Pairwise Comparisons)

When there are 2+ selections, delta metrics compute pairwise comparisons between all selection pairs:

```yaml
# With split_by creating 4 selections
dataloaders:
  coco_classes:
    type: parquet
    split_by: class
    split_values: [dog, cat, bird, elephant]

metrics_processor:
  domain_gap:
    type: domain_gap
```

For N selections, DQM-ML computes N×(N-1)/2 unique pairs:

| N Selections | Pairs Computed |
|------------|--------------|
| 2 | 1 |
| 4 | 6 |
| 10 | 45 |

**Output** (one row per pair):

| mmd_linear | selection_source | selection_target |
|------------|-----------------|----------------|
| 53.5 | coco_classes_dog | coco_classes_cat |
| 60.8 | coco_classes_dog | coco_classes_horse |
| 141.0 | coco_classes_cat | coco_classes_horse |
| ... | ... | ... |

## Which Metrics Support Delta?

Only some metrics support pairwise comparisons:

| Metric | Per-Selection | Delta (Pairwise) |
|--------|---------------|------------------|
| Completeness | ✓ | |
| Representativeness | ✓ | |
| Domain Gap | | ✓ |
| Visual Features | ✓ | |

## Metrics Configuration

### Completeness

```yaml
metrics_processor:
  completeness:
    type: completeness
    input_columns: [name, age, score]
    include_per_column: true
    include_overall: true
```

### Representativeness

```yaml
metrics_processor:
  representativeness:
    type: representativeness
    input_columns: [feature_x, feature_y]
    distribution: normal  # or uniform
```

### Domain Gap

```yaml
metrics_processor:
  domain_gap:
    type: domain_gap
    INPUT:
      embedding_col: embedding
    DELTA:
      metric: mmd_linear  # or fid, wasserstein_1d, klmvn_diag
```

### Visual Features

```yaml
metrics_processor:
  visual_features:
    type: visual_features
    DATA:
      image_column: image_path
      mode: path
```

## Output Files

### Per-Selection Output

Filename pattern: `metrics_<processor_key>-.parquet`

Example columns:

| Column | Type | Description |
|--------|------|--------------|
| selection | string | Selection name |
| completeness_overall | float | Overall completeness |
| completeness_column_X | float | Per-column completeness |

### Delta Output

Filename pattern: `metrics_<processor_key>_delta-.parquet`

Example columns:

| Column | Type | Description |
|--------|------|--------------|
| mmd_linear | float | Domain gap score |
| fid | float | FID score |
| selection_source | string | First selection name |
| selection_target | string | Second selection name |

## Related Pages

- [Metrics Overview](metrics.md) - All available metrics
- [Completeness](metrics/completeness.md) - Completeness metric details
- [Domain Gap](metrics/domain_gap.md) - Domain gap metric details
- [Representativeness](metrics/representativeness.md) - Representativeness metric details
- [Visual Features](metrics/visual_features.md) - Visual features details
- [Configuration](configuration.md) - Full configuration guide
- [Data Loaders](dataloaders.md) - Data source configuration