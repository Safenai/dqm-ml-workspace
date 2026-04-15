# Getting Started Fixtures

This directory contains test fixtures used in the quickstart guide and E2E tests.

## Files

| File | Description |
|------|-------------|
| `data.csv` | Sample CSV data with missing values (nulls) for completeness testing |
| `completeness.yaml` | Configuration file for completeness metric |
| `README.md` | This file |

## Usage

These files are referenced in `docs/quickstart.md` for users to copy and try locally.

### Quick Test

```bash
# From the repository root or any directory with these files
dqm-ml process -p tests/fixtures/getting_started/completeness.yaml
```

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
3. Add tests in `tests/cli/test_quickstart.py`
4. Update documentation in `docs/quickstart.md`