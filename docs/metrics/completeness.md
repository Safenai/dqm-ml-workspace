# Completeness Metric

The Completeness metric evaluates the presence of non-null values in your dataset. It is defined by the degree to which subject data associated with an entity has values for all expected attributes.

## What It Measures

Completeness measures what percentage of your data is present (non-null). It helps you find:

- **Missing values** in your training data
- **Columns with high null rates** that might need attention
- **Data collection gaps** that could affect model quality

A completeness score of 1.0 means no missing values; 0.0 means all values are missing.

### Use Cases

- Validate data after ETL pipelines
- Identify columns that need imputation
- Check data quality before model training
- Monitor data freshness in production

## Processor Information

* **Class**: `CompletenessProcessor`
* **Package**: `dqm-ml-core`
* **Type Name**: `completeness`

## Configuration Parameters

* `input_columns`: List of columns to analyze. If omitted, all columns are analyzed.
* `include_per_column`: Boolean, whether to output a score for each analyzed column.
* `include_overall`: Boolean, whether to output an average score across all columns.

## Example YAML Configuration

```yaml
metrics_processor:
  completeness_check:
    type: completeness
    input_columns: ["age", "income", "zip_code"]
    include_per_column: true
    include_overall: true
```

## Output

The processor returns a dictionary with the following keys:

* `completeness_overall`: The average completeness score (0.0 to 1.0).
* `completeness_<column_name>`: The completeness score for a specific column (if `include_per_column` is true).
