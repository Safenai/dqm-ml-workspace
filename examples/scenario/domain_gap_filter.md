# Domain Gap (Filter)

[📄 Config](../config/domain_gap_with_filter.yaml) · [📖 Docs](../../docs/metrics/domain_gap.md)

Two dataloaders read the same source file but filter on different `source` values (safari vs reserve). MMD-RBF measures the distribution shift between these two subsets in a focused pairwise comparison.

## Data Loaders with Filters

```yaml
# Two loaders, same file, different filter values. Each creates a
# selection containing only images from that acquisition source.
dataloaders:
  loaders:
    - name: safari
      type: parquet
      path: examples/data/samples_with_images.parquet
      batch_size: 500
      filters:
        - column: source
          values: [safari]

    - name: reserve
      type: parquet
      path: examples/data/samples_with_images.parquet
      batch_size: 500
      filters:
        - column: source
          values: [reserve]
```

## Feature Extraction

```yaml
# Both safari and reserve are embedded through ResNet-18.
features:
  outputs:
    path: examples/outputs/story_domain_gap_filter_features.parquet
    include:
      - sample_id
      - source
  processors:
    - name: embedding
      type: features_embeddings
      columns:
        input: [image_bytes]
      model:
        arch: resnet18
        n_layer_feature: -2
      infer:
        batch_size: 32
        width: 64
        height: 64
        norm_mean: [0.485, 0.456, 0.406]
        norm_std: [0.229, 0.224, 0.225]
```

## Gap Computation

```yaml
# MMD with an RBF kernel compares the two filtered subsets directly.
# The gamma parameter controls kernel width — larger values make the
# test more sensitive to local distribution differences.
gap:
  outputs:
    path: examples/outputs/story_gap_filter.parquet
  processors:
    - name: mmd_rbf_gap
      type: domain_gap
      columns:
        input: [image_bytes_embedding]
      distance:
        metric: mmd_rbf
        kernel_params:
          gamma: 1.0
```

## Run

```bash
dqm-ml process -p examples/config/scenario/domain_gap_with_filter.yaml
```

## Output

Two files are written to `examples/outputs/`:

- `domain_gap_filter_features.parquet` — per-image embeddings with source labels
- `gap_filter.parquet` — MMD-RBF score between safari and reserve

```python
import pandas as pd

df = pd.read_parquet("examples/outputs/story_gap_filter.parquet")
print(df.to_markdown(index=False))
```

The gap output has a single row (MMD-RBF score between safari and reserve):

|   mmd_rbf | selection_source   | selection_target   | source_target                  |
|----------:|:-------------------|:-------------------|:-------------------------------|
| 0.0049737 | safari     | reserve    | safari_reserve |

The MMD-RBF score (0.005) is close to zero, suggesting the mean embeddings of safari and reserve images are nearly identical under the RBF kernel. Yet the split-based FID above measured a gap of 119 between the same two sources. This discrepancy is informative: FID captures differences in both the mean and the covariance of the embedding distributions, while MMD-RBF with a fixed kernel width is primarily sensitive to mean shifts. When two domains differ in their spread or correlation structure — not just their average embedding — FID will flag a gap that MMD-RBF may not detect. Choosing between these metrics depends on which aspect of distribution shift is relevant to your model's robustness.
