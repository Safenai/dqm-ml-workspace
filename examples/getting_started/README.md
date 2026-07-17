# Getting Started Fixtures

This directory contains example files used in the quickstart guide.

## Files

| File | Description |
|------|-------------|
| `data.csv` | Sample CSV data with missing values (nulls) for completeness testing |
| `config.yaml` | Configuration file for completeness metric |
| `README.md` | This file |

## Usage

These files are referenced in `docs/quickstart.md` for users to copy and try locally.

### Quick Test

```bash
# From the repository root or any directory with these files
dqm-ml process -p examples/getting_started/config.yaml
```

> Configuration file: [`config.yaml`](config.yaml)

### Data Description

The `data.csv` file contains:
- `name`: String column with 1 null value (Diana)
- `age`: Numeric column with 1 null value (Diana)
- `score`: Numeric column with 1 null value (row 3)

Expected completeness result: ~75% overall (3 out of 4 values present for each column)

## Adding More Examples

To add new examples:
1. Add data files (CSV, Parquet, etc.)
2. Add corresponding YAML config files
3. (removed — examples are in `examples/getting_started/`)
4. Update documentation in `docs/quickstart.md`