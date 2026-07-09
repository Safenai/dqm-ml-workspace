# Diversity Metric

The Diversity **Metric** evaluates the variety and distribution of categorical values in your dataset. It provides four complementary indices that measure how many distinct categories exist and how evenly they are distributed.

> **See also:** [Concepts](../formal_concepts.md) for definitions of **Metric**, **Sample**, and related terminology.

## What It Measures

Diversity quantifies the spread and balance of category occurrences. Use it to:

- **Detect class imbalance** — Identify columns dominated by a single value
- **Assess dataset coverage** — Measure how many distinct categories your data covers
- **Compare splits** — Verify that train/test splits have similar category distributions
- **Compare acquisition batches** — Detect category distribution shifts across different collection sessions or data sources to ensure stable dataset assembly

### Available Indices

| Index | Range | What It Measures |
|-------|-------|------------------|
| **Simpson** | [0, 1] | Probability two random samples belong to different categories (unbiased estimator). High = diverse. |
| **Gini-Simpson** | [0, 1] | Gini impurity — probability of incorrect classification. High = diverse. |
| **Shannon Entropy** | [0, ∞) | Information content / uncertainty. Grows with category count and evenness. |
| **Richness** | [1, N] | Simple count of unique categories. |

### Use Cases

- **Classification datasets** — Ensure all expected classes are present in training data
- **Data augmentation** — Verify augmented data doesn't collapse categories
- **Data pipeline integrity** — Detect when a data pipeline drops categories during dataset construction
- **Label quality** — Spot anomalous categories that might indicate labeling errors

## Processor Information

* **Class**: `DiversityProcessor`
* **Package**: `dqm-ml-core`
* **Type Name**: `diversity`

## Configuration Parameters

* `columns.input`: List of columns to analyze (required). Columns are cast to string for type-safe aggregation.
* `metrics`: List of diversity indices to compute (optional). Defaults to all four: `simpson`, `gini`, `shannon`, `richness`.

## Example YAML Configuration

### Full (all four indices):

```yaml
metrics:
  processors:
    - name: category_diversity
      type: diversity
      columns:
        input: ["class_label", "color", "region"]
      metrics: ["simpson", "gini", "shannon", "richness"]
```

### Subset (Simpson + Richness only):

```yaml
metrics:
  processors:
    - name: simple_diversity
      type: diversity
      columns:
        input: ["class_label"]
      metrics: ["simpson", "richness"]
```

## Output

The processor returns a dictionary with one key per column and metric:

* `<column>_simpson`: Simpson index score (0.0 to 1.0)
* `<column>_gini`: Gini-Simpson index score (0.0 to 1.0)
* `<column>_shannon`: Shannon entropy score (0.0 to ∞)
* `<column>_richness`: Number of unique categories (integer)
