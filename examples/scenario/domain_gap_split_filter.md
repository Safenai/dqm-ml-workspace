# Domain Gap (Split + Filter)

[📄 Config](../config/scenario/domain_gap_with_split_and_filter.yaml) · [📖 Docs](../../docs/metrics/domain_gap.md)

Filters to the zoo source only, then splits by `class_name` without restricting values — all four animal classes (elephant, giraffe, lion, zebra) are carried through. Wasserstein-1D compares every class pair, producing 6 pairwise distances that answer: within the zoo environment, how separable are the four species in embedding space?

## Data Loader with Filter and Split

```yaml
# Filter to zoo only (one controlled environment), then split by all
# class_name values. The pipeline creates four groups:
#   zoo/elephant, zoo/giraffe, zoo/lion, zoo/zebra
dataloaders:
  loaders:
    - name: zoo
      type: parquet
      path: examples/data/samples_with_images.parquet
      batch_size: 500
      filters:                  # keep only zoo-source images
        - column: source
          values: [zoo]
      split:                    # then split by class (no values = all)
        by: class_name
```

## Feature Extraction

```yaml
features:
  outputs:
    path: examples/outputs/story_domain_gap_split_filter_features.parquet
    include:
      - sample_id
      - source
      - class_name
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
# Wasserstein-1D (Earth Mover's Distance) compares every pair of groups.
# This isolates per-class distribution shift within a single source —
# e.g., how far zoo elephants are from zoo giraffes in embedding space.
gap:
  outputs:
    path: examples/outputs/story_gap_split_filter.parquet
  processors:
    - name: wasserstein_gap
      type: domain_gap
      columns:
        input: [image_bytes_embedding]
      distance:
        metric: wasserstein_1d
```

## Run

```bash
dqm-ml process -p examples/config/scenario/domain_gap_with_split_and_filter.yaml
```

## Output

Two files are written to `examples/outputs/`:

- `domain_gap_split_filter_features.parquet` — per-image embeddings with source/class labels
- `gap_split_filter.parquet` — Wasserstein score between every class pair (6 pairs)

```python
import pandas as pd

df = pd.read_parquet("examples/outputs/story_gap_split_filter.parquet")
print(df.to_markdown(index=False))
```

The gap output has 6 rows — one for each unordered class pair within the zoo environment:

|   wasserstein_1d | selection_source | selection_target | source_target                     |
|-----------------:|:-----------------|:-----------------|:----------------------------------|
|         0.075    | zoo_elephant     | zoo_giraffe      | zoo_elephant_zoo_giraffe          |
|         0.123    | zoo_elephant     | zoo_lion         | zoo_elephant_zoo_lion             |
|         0.092    | zoo_elephant     | zoo_zebra        | zoo_elephant_zoo_zebra            |
|         0.087    | zoo_giraffe      | zoo_lion         | zoo_giraffe_zoo_lion              |
|         0.065    | zoo_giraffe      | zoo_zebra        | zoo_giraffe_zoo_zebra             |
|         0.118    | zoo_lion         | zoo_zebra        | zoo_lion_zoo_zebra                |

These 6 distances can be visualised as a 4 × 4 heatmap, where each cell shows the Wasserstein-1D distance between a pair of classes (the diagonal is zero). The heatmap makes it easy to spot which classes overlap in embedding space (low values) and which are well separated (high values). In this scenario, `zoo_elephant → zoo_lion` (0.123) and `zoo_lion → zoo_zebra` (0.118) show the strongest separation, while `zoo_giraffe → zoo_zebra` (0.065) and `zoo_elephant → zoo_giraffe` (0.075) are the most overlapping pairs — consistent with large herbivores sharing similar enclosure backgrounds. A Wasserstein score near 0 would indicate complete distribution overlap; values approaching 0.2 or higher would suggest near-perfect separation. None of the pairs exceed 0.15 in this dataset, confirming that the embedding space preserves some class-specific signal but the four species are not strongly separable from appearance alone within a single controlled environment.
