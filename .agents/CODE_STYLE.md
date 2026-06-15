# Code Style

## Python Version

Target Python version: >=3.12

## Docstrings

All Python docstrings must follow **Google Python Style Guide**. Docstrings should be meaningful and describe the purpose, args, and return values of functions.

Example:

```python
def process_data(data: list[int], threshold: float) -> dict[str, Any]:
    """Process input data and compute statistics.

    Args:
        data: List of numeric values to process.
        threshold: Minimum value threshold for filtering.

    Returns:
        Dictionary containing 'mean', 'median', and 'filtered_count'.

    Raises:
        ValueError: If data is empty or threshold is negative.
    """
```

## Linting

Run linting with:
```bash
uv run nox -s lint
```

Show fixable errors:
```bash
uv run nox -s lint_fix
```

Fix fixable errors:
```bash
uv run nox -s fmt
```

## Spell Checking

Run spell checking with:
```bash
uv run nox -s spell
```

Configuration (from pyproject.toml):
- Checks all files: `files = ["**/*"]`

## Type Checking

Run type checking with:
```bash
uv run nox -s type_check
```

Configuration:
- Strict mode enabled
- Error codes: `deprecated`, `exhaustive-match`, `explicit-override`
- Ignore missing imports: enabled

## PyArrow Rules

- Pipeline operates on `pa.RecordBatch`. Never `pd.DataFrame` in the data path.
- CSV bridge: `pd.read_csv()` → immediate `pa.RecordBatch.from_pandas()` (`pandas.py:69`).
- `pd.to_numeric` only as fallback for mixed-type coercion (Representativeness).
- Tests may use `.to_pandas()` for assertion convenience — not as a processing step.
- S3: use `dqm_ml_job.utils.s3.get_s3_filesystem()` (handles env vars and explicit config).
- Embeddings: `pa.FixedSizeListArray.from_arrays([list_of_arrays], type=pa.float32())`.
