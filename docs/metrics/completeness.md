# Completeness Metric

The Completeness **Metric** evaluates the presence of non-null values in your dataset. It is defined by the ratio of present (non-null) values to total expected values for each **Sample**.

> **See also:** [Concepts](../formal_concepts.md) for definitions of **Metric**, **Sample**, and related terminology.

## What It Measures

Completeness answers: *"How much of my data is actually there?"*

A score of 1.0 means no missing values; 0.0 means all values are missing.

### Use Cases

**1. Data quality gates before training**

Set a minimum completeness threshold per column as a quality gate. If critical columns fall below the threshold, reject the dataset or source before training.

| Completeness | Meaning | Action |
|---|---|---|
| 1.0 | No missing values | Ready for training |
| 0.95–0.99 | Minimal missing | Acceptable; ensure downstream model handles nulls or impute |
| 0.80–0.95 | Moderate missing | Investigate root cause (acquisition gap, ETL bug); impute |
| < 0.80 | Heavy missing | Drop column or source unless the missing pattern is informative |

**2. Feature selection**

Rank columns by completeness to decide which to keep, impute, or drop:

- Score = 1.0: keep as-is
- Score > 0.90: impute if the model requires dense input
- Score ≤ 0.90: consider dropping unless the column is high-value (e.g. a target label)

**3. Imputation planning**

The completeness score tells you the maximum information you can recover per column:

- High completeness (≥ 0.95): simple imputation (mean / median) is usually safe.
- Moderate completeness (0.80–0.95): more sophisticated imputation (KNN, MICE, model-based) may be worth the effort.
- Low completeness (< 0.80): imputation introduces more noise than signal — consider dropping.

*Tip:* Compare completeness patterns across your train/validation splits. If one split has significantly more missing values, the split is biased and should be re-done.

**4. Data source comparison**

When assembling a dataset from multiple sources, compute completeness per source. A source with systematically lower completeness may have acquisition issues (faulty sensor, network dropout, parsing errors). Use this to decide whether to include, repair, or exclude that source.

## Processor Information

- **Class**: `CompletenessProcessor`
- **Package**: `dqm-ml-core`
- **Type Name**: `completeness`

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns.input` | list[string] | all columns | Columns to analyze |
| `include_per_column` | bool | `true` | Output a score for each analyzed column |
| `include_overall` | bool | `true` | Output the average score across all columns |
| `include_metadata` | bool | `false` | Include detailed metadata in output |

There are no numerical parameters to tune — the metric simply counts non-null values. These boolean flags only control which outputs are generated.

## Example YAML Configuration

### Minimal — check all columns

```yaml
metrics:
  processors:
    - name: completeness_check
      type: completeness
```

### Targeted — check specific columns with per-column output

```yaml
metrics:
  processors:
    - name: completeness_check
      type: completeness
      columns:
        input: ["age", "income", "zip_code"]
      include_per_column: true
      include_overall: true
```

### Quality gate — overall score only, no per-column detail

```yaml
metrics:
  processors:
    - name: qc_gate
      type: completeness
      columns:
        input: ["*"]
      include_per_column: false
      include_overall: true
```

## Output

| Column | Type | Description |
|--------|------|-------------|
| `completeness_overall` | float | Average completeness score across all analyzed columns (0.0–1.0) |
| `completeness_<column>` | float | Completeness score for a specific column (if `include_per_column` is true) |
| `_metadata` | dict | JSON with `columns_analyzed`, `total_samples_per_column`, `per_column_scores`, `overall_score` (if `include_metadata` is true) |
