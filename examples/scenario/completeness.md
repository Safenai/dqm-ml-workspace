# Completeness

[📄 Config](../config/scenario/completeness.yaml) · [📖 Docs](../../docs/metrics/completeness.md)

Checks that every sample has non-null values for the visual-feature columns computed by the Visual Features pipeline. Nulls here indicate an image-processing failure (corrupt bytes, load error, etc.). The scenario uses 1200 rows × 4 visual-feature columns — a small dataset, so false alarms are a risk.

## Data Loader

```yaml
# Reads the visual-features output (1200 rows, 4 feature columns).
dataloaders:
  loaders:
    - name: source_dataset
      type: parquet
      path: examples/outputs/story_visual_features.parquet
      batch_size: 50000
```

## Processor

```yaml
# Scan every row for null values in the visual-feature columns.
metrics:
  outputs:
    path: examples/outputs/story_completeness.parquet
  processors:
    - name: completeness
      type: completeness
      columns:
        input: [sample_id, image_bytes_luminosity, image_bytes_contrast, image_bytes_blur, image_bytes_entropy]
      include_per_column: true
      include_overall: true
```

## Prerequisite

```bash
dqm-ml process -p examples/config/scenario/visual_features.yaml
```

## Run

```bash
dqm-ml process -p examples/config/scenario/completeness_scenario.yaml
```

## Output

Per-column and overall completeness scores are written to `examples/outputs/story_completeness.parquet`:

```python
import pandas as pd

df = pd.read_parquet("examples/outputs/story_completeness.parquet")
print(df.to_markdown(index=False))
```

Each row corresponds to one dataset selection. `completeness_<column>` gives the fraction of non-null values per column; `completeness_overall` is the aggregate across all checked columns.

| selection      |   completeness_sample_id |   completeness_image_bytes_luminosity |   completeness_image_bytes_contrast |   completeness_image_bytes_blur |   completeness_image_bytes_entropy |   completeness_overall |
|:---------------|-------------------------:|--------------------------------------:|------------------------------------:|--------------------------------:|-----------------------------------:|-----------------------:|
| source_dataset |                        1 |                                     1 |                                   1 |                               1 |                                  1 |                      1 |

`sample_id` is 100 % complete — expected for a row identifier. All four visual-feature columns are also 100 % complete, meaning the image-processing pipeline successfully extracted features from every sample in this synthetic dataset. The overall completeness of 100 % confirms there are no data-quality issues to address. In a real-world scenario, any column below 100 % would point to a processing failure that should be investigated and either fixed or imputed before training.
