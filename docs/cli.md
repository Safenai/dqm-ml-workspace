# CLI Reference

Command-line interface for running DQM-ML pipelines.

## Command

```bash
dqm-ml process -p <config.yaml>
```

## Options

| Option | Description |
|--------|------------|
| `-p`, `--path-config` | Path to YAML configuration file (required) |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (missing file, invalid config, etc.) |

## Examples

### Basic Usage

```bash
dqm-ml process -p config.yaml
```

### With Relative Path

```bash
dqm-ml process -p examples/config/completeness.yaml
```

### Features Interface (Visual Features)

```bash
dqm-ml process -p examples/config/features_image.yaml
```

### Metrics Interface (Completeness)

```bash
dqm-ml process -p examples/config/metrics_completeness.yaml
```

### Gap Interface (Domain Gap)

```bash
dqm-ml process -p examples/config/gap_domain_gap.yaml
```

### Full Pipeline (All Three Interfaces)

```bash
dqm-ml process -p examples/config/full_pipeline.yaml
```

### List Available Processors

```bash
dqml-ml list
```

Shows available processors grouped by interface:
- **features**: `image_features`, `features_embeddings`
- **metrics**: `completeness`, `representativeness`, `diversity`
- **gap**: `domain_gap`
```

## Getting Help

```bash
dqm-ml --help
dqm-ml process --help
```

## Related Pages

- [Quick Start](quickstart.md) - Get started with DQM-ML
- [Configuration](configuration/overview.md) - Write configuration files
- [Configuration: Features](configuration/features.md) - Features interface
- [Configuration: Metrics](configuration/metrics.md) - Metrics interface
- [Configuration: Gap](configuration/gap.md) - Gap interface
- [Metrics](metrics.md) - Available metrics by interface