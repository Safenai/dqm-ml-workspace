# Domain Gap (Split)

[📄 Config](../config/domain_gap_with_split.yaml) · [📖 Docs](../../docs/metrics/domain_gap.md)

Splits the dataset by `source` (zoo, safari, reserve) and computes the FID between every pair. A high FID between zoo and safari suggests a significant distribution shift — the model may need to be trained on both environments to generalise.

## Data Loader with Split

```yaml
# The `split` directive creates three virtual subsets — one per source
# value. Each subset flows through the pipeline independently.
dataloaders:
  loaders:
    - name: animals
      type: parquet
      path: examples/data/samples_with_images.parquet
      batch_size: 500
      split:
        by: source               # column to split on
        values: [zoo, safari, reserve]
```

## Feature Extraction

```yaml
# The same ResNet-18 embedding processor runs on every subset,
# writing all split groups to a single feature file.
features:
  outputs:
    path: examples/outputs/story_domain_gap_split_features.parquet
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
# FID (Fréchet Inception Distance) is computed pairwise between the
# three subsets: zoo↔safari, zoo↔reserve, safari↔reserve.
gap:
  outputs:
    path: examples/outputs/story_gap_split.parquet
  processors:
    - name: fid_gap
      type: domain_gap
      columns:
        input: [image_bytes_embedding]
      distance:
        metric: fid
        epsilon: 1e-6            # numerical stabiliser for covariance inversion
```

## Run

```bash
dqm-ml process -p examples/config/scenario/domain_gap_with_split.yaml
```

## Output

Two files are written to `examples/outputs/`:

- `domain_gap_split_features.parquet` — per-image embeddings with source labels
- `gap_split.parquet` — pairwise FID scores between zoo, safari, and reserve

```python
import pandas as pd

df = pd.read_parquet("examples/outputs/story_gap_split.parquet")
print(df.to_markdown(index=False))
```

The gap output has one row per pair of split groups — the FID score between every pair of acquisition sources:

|     fid | selection_source   | selection_target   | source_target                  |
|--------:|:-------------------|:-------------------|:-------------------------------|
| 563.486 | zoo        | safari     | zoo_safari     |
| 394.195 | zoo        | reserve    | zoo_reserve    |
| 119.208 | safari     | reserve    | safari_reserve |

The three pairwise gaps are not equal: zoo↔safari (563) and zoo↔reserve (394) are substantially larger than safari↔reserve (119). This means the zoo environment is the most distinctive — its enclosure-based images differ sharply from both safari and reserve images. Safari and reserve are relatively more similar to each other (FID ≈ 119 still represents a notable gap, but well below the zoo comparisons). In practice, this ranking helps prioritise domain-adaptation effort: zoo requires the most attention, while the smaller safari↔reserve gap suggests those two sources might be pooled with less risk of degrading model performance.
