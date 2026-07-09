## 2. Processors — How to Configure Computation

Each interface contains a list of processors. A processor is a named unit of work with a registered type and its own parameters.

Each processor type has its own configuration parameters, documented on dedicated pages:

- [Features Processors](features.md) — image features and embedding processors
- [Metrics Processors](metrics.md) — completeness, representativeness, and diversity processors
- [Gap Processors](gap.md) — domain gap processors

### 2.1 Common Processor Structure

```yaml
<interface>:
  processors:
    - name: <processor_name>      # unique name within this interface
      type: <processor_type>      # registered type (e.g., completeness, image_features)
      columns:                    # which columns to process
        input: ["col_a", "col_b"]
      # ... processor-specific parameters ...
```

Every processor has:

- **`name`** — human-readable identifier; appears in output filenames and result columns
- **`type`** — selects the processor implementation from the plugin registry
- **`columns`** — which data columns to operate on (see below)
- **Processor-specific parameters** — documented per processor below

### 2.2 Columns

The `columns` key tells a processor where to find the data it needs:

```yaml
columns:
  input: ["col_1", "col_2"]     # columns to process (wildcards supported: "feature_*")
  exclude: ["meta_*"]           # columns to exclude from processing (wildcards supported)
  rename:                       # rename computed output columns
    - from: <metric_name>
      to: <renamed_metric>
  prefix: <prefix>              # prefix added to computed column names
  suffix: <suffix>              # suffix added to computed column names
```

Computed column naming: `<prefix><renamed_metric><suffix>`
