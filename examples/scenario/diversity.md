# Diversity

[📄 Config](../config/diversity.yaml) · [📖 Docs](../../docs/metrics/diversity.md)

Measures how evenly samples are distributed across `class_name` and `source` using four complementary indices (Simpson, Gini-Simpson, Shannon Entropy, Richness). Useful for detecting class imbalance or source-representation issues before training.

## Data Loader

```yaml
# Reads the visual-features output from step 3.
# The 1200-row dataset has 4 classes × 3 sources.
dataloaders:
  loaders:
    - name: source_dataset
      type: parquet
      path: examples/outputs/story_visual_features.parquet
      batch_size: 10000
```

## Processor

```yaml
# Diversity runs on two categorical columns: animal class (4 types)
# and acquisition source (zoo, safari, reserve). Each metric index
# captures a different facet of balance:
#   - simpson / gini    dominance and evenness
#   - shannon            uncertainty / entropy
#   - richness           raw category count
metrics:
  outputs:
    path: examples/outputs/story_diversity.parquet
  processors:
    - name: diversity
      type: diversity
      columns:
        input: [class_name, source]
      metrics: [simpson, gini, shannon, richness]
```

## Run

```bash
dqm-ml process -p examples/config/scenario/diversity.yaml
```

## Output

Diversity indices (Simpson, Gini-Simpson, Shannon, Richness) are written to `examples/outputs/story_diversity.parquet`:

```python
import pandas as pd

df = pd.read_parquet("examples/outputs/story_diversity.parquet")
print(df.to_markdown(index=False))
```

Each row corresponds to one dataset selection. Metric columns are prefixed with the categorical column name (`class_name_*`, `source_*`). The four indices are: **Richness** (unique category count), **Shannon** (entropy), **Gini** (evenness-corrected), and **Simpson** (dominance).

| selection      |   class_name_richness |   class_name_shannon |   class_name_gini |   class_name_simpson |   source_richness |   source_shannon |   source_gini |   source_simpson |
|:---------------|----------------------:|---------------------:|------------------:|---------------------:|------------------:|-----------------:|--------------:|-----------------:|
| source_dataset |                     4 |              1.38592 |          0.749695 |             0.749695 |                 3 |          1.09698 |       0.66578 |         0.66578 |

The 4 animal classes are nearly perfectly balanced: Shannon = 1.386 (essentially ln(4) ≈ 1.386) and Simpson = 0.75 (the theoretical maximum for 4 equal classes is 0.75). No class dominates — the dataset needs no re-weighting or stratified sampling for class imbalance. The three acquisition sources are also well-balanced: Shannon = 1.097 (close to ln(3) ≈ 1.099) and Simpson = 0.666 (close to the maximum of 0.667). The reserve has slightly more samples (433 vs ~380 for zoo and safari), but the skew is negligible. If downstream experiments show source-specific bias, this small imbalance can be corrected with a simple per-source weight, but is unlikely to degrade model performance on its own.
