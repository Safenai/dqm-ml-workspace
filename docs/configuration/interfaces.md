## 1. Interfaces — What You Can Compute

DQM-ML provides 3 interfaces, each corresponding to a distinct function in the data quality methodology:

| Interface | Purpose | Output |
|-----------|---------|--------|
| `features` | Compute a **Feature** on each **Sample** (e.g., image luminosity, embeddings) | One row per sample, enriched with new columns |
| `metrics` | Compute a **Metric** on a **Sample Selection** (e.g., completeness, diversity) | One row per selection (or per-column scores) |
| `gap` | Compute a **Domain Gap** between 2 **Sample Selections** (e.g., MMD, FID) | One row per pair of selections |

### Interface Configuration

Each interface groups optional shared configuration with a list of processors:

```yaml
<interface>:      # features, metrics, or gap
  storage:        # optional — storage backend (overrides global)
    ...
  compute:        # optional — device, threads (overrides global)
    ...
  errors:         # optional — error policy (overrides global)
    ...
  outputs:        # optional — where to save results (see Outputs below)
    path: "output/..."
  processors:     # list of processor configurations
    - name: ...
      type: ...
```

> Interface-level `storage`, `compute`, and `errors` override the corresponding [global configuration](global.md) for all processors within that interface.

### Outputs

Each interface optionally defines an `outputs` section controlling where and how results are saved:

```yaml
features:
  outputs:
    path: "output/features.parquet"          # file path
    include: ["sample_id", "class"]          # columns to include from source data (features only)
    exclude: ["meta_*"]                      # columns to exclude from source (wildcard supported)
```

```yaml
metrics:
  outputs:
    path: "output/metrics.parquet"
```

```yaml
gap:
  outputs:
    path: "output/gap_{}-{}.parquet"         # {source}_{target} inserted for pairwise
    pairwise: true                           # add selection_source / selection_target columns (default: true)
```

> See [Metrics](metrics.md) for details on per-selection vs delta output formats.
