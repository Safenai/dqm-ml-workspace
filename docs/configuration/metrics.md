### 2.5 Metrics Processors

Metrics processors take a **Sample Selection** and compute a **Metric** aggregated over all samples.

Metrics are computed in two modes depending on the number of **Sample Selections**:

- **Per-Selection**: when there is a single selection, metrics compute statistics on that data alone
- **Delta (Pairwise)**: when there are 2+ selections, some metrics compute pairwise comparisons between all selection pairs

| Metric | Per-Selection | Delta (Pairwise) |
|--------|---------------|------------------|
| Completeness | ✓ | |
| Representativeness | ✓ | |
| Diversity | ✓ | |

#### Completeness

Detects missing/null values in scalar columns.

```yaml
metrics:
  processors:
    - name: null_check
      type: completeness
      columns:
        input: ["age", "income", "email"]
      include_per_column: true
      include_overall: true
      include_metadata: false
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_per_column` | `true` | Emit per-column completeness scores |
| `include_overall` | `true` | Emit overall completeness score |
| `include_metadata` | `false` | Include null-count metadata |

> See [Completeness](../metrics/completeness.md) for interpretation thresholds and use cases.

> **💡 Runnable example:** See [../../examples/scenario/completeness.md](../../examples/scenario/completeness.md) for this metric in a complete pipeline.

#### Representativeness

Measures how well a column's distribution matches a target distribution.

```yaml
metrics:
  processors:
    - name: repr_check
      type: representativeness
      columns:
        input: ["feature_x", "feature_y"]
      metrics: [chi-square, grte, kolmogorov-smirnov, shannon-entropy]
      alpha: 0.05                       # significance level for chi-square and KS
      epsilon: 1e-9                     # division-by-zero prevention
      distribution: normal              # target distribution: normal or uniform
      mean_std_estimation: from_first_batch  # how mean/std are estimated: from_first_batch, per_batch, user_provided, from_all_data
      expected_counts_method: cdf       # expected bin counts: cdf (exact) or monte_carlo (stochastic)
      distribution_params:              # per-column params for user_provided strategy
        - column: feature_x
          mean: 0.0
          std: 1.0
        - column: feature_y
          mean: 0.0
          std: 1.0
      interpretation:
        follows_distribution: "fits target"
        does_not_follow_distribution: "diverges from target"
        high_diversity: "varied"
        low_diversity: "uniform"
        high_representativeness: "representative"
        low_representativeness: "under-represented"
      histogram:
        bins: 10
      shannon:
        threshold: 2.0                  # high/low diversity threshold
      grte:
        threshold: 0.5                  # high/low representativeness threshold
        scaling_factor: -2.0
      ks:
        sample_size: 500                # max samples per batch
        min_sample_size: 50             # min samples per batch
        sample_divisor: 20              # fraction divisor
```

> See [Representativeness](../metrics/representativeness.md) for GRTE formula, calibration, and use cases.

> **💡 Runnable example:** See [../../examples/scenario/representativeness.md](../../examples/scenario/representativeness.md) for this metric in a complete pipeline.

#### Diversity

Measures category distribution spread across categorical columns.

```yaml
metrics:
  processors:
    - name: diversity_check
      type: diversity
      columns:
        input: ["class_label", "category"]
      metrics: [simpson, gini, shannon, richness]
```

> See [Diversity](../metrics/diversity.md) for metric interpretation and use cases.

> **💡 Runnable example:** See [../../examples/scenario/diversity.md](../../examples/scenario/diversity.md) for this metric in a complete pipeline.
