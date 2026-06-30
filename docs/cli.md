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

## Getting Help

```bash
dqm-ml --help
dqm-ml process --help
```

## Related Pages

- [Quick Start](quickstart.md) - Get started with DQM-ML
- [Configuration](configuration/overview.md) - Write configuration files
- [Metrics](metrics.md) - Available metrics