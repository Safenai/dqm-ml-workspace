# Representativeness Metric

The Representativeness metric compares the distribution of your data to a target reference distribution (such as Normal or Uniform). It ensures that the dataset conforms to a given specification or requirement.

## What It Measures

Representativeness checks if your data follows a known distribution. Use it to:

- **Validate synthetic data** — Ensure generated data matches expected patterns
- **Check for data drift** — Detect changes in data distribution over time
- **Ensure balanced datasets** — Verify class distributions in classification tasks

### Available Statistical Tests

| Test | What It Does | Best For |
|------|-------------|----------|
| **Chi-Square (χ²)** | Goodness-of-fit test | Categorical/binned data |
| **Kolmogorov-Smirnov (KS)** | Non-parametric test | Continuous distributions |
| **Shannon Entropy** | Measures information diversity | General diversity |
| **GRTE** | Granular Relative Theoretical Entropy | Complex distributions (Confiance.ai) |

### Use Cases

- Validate that training data matches expected distributions
- Detect distribution shifts in production data
- Check if data augmentation preserved original distribution
- Verify synthetic data quality

## Processor Information

* **Class**: `RepresentativenessProcessor`
* **Package**: `dqm-ml-core`
* **Type Name**: `representativeness`

## Configuration Parameters

* `input_columns`: Numeric columns to analyze.
* `distribution`: Reference distribution name (`normal`, `uniform`).
* `metrics`: List of statistical tests to run (`chi-square`, `kolmogorov-smirnov`, `shannon-entropy`, `grte`).
* `bins`: Number of bins for histogram-based tests (default: 10).
* `distribution_params`: (Optional) Parameters for the reference distribution (e.g., `mean`, `std` for normal).

## Example YAML Configuration

```yaml
metrics_processor:
  dist_check:
    type: representativeness
    input_columns: ["feature_x"]
    distribution: "normal"
    metrics: ["chi-square", "kolmogorov-smirnov"]
    bins: 30
    distribution_params:
      mean: 0.0
      std: 1.0
```

## Output

The processor returns a dictionary with scores for each requested metric and column:

* `<column_name>_<metric_name>_score`: The statistical test score.
* `<column_name>_<metric_name>_pvalue`: The p-value associated with the test.
