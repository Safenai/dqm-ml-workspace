# Python Compatibility

This document describes the Python version compatibility of DQM-ML.

## Supported Versions

| Python Version | Support Status | Notes |
|----------------|---------------|-------|
| 3.11 | ✓ Full | Minimum supported version, tested in CI |
| 3.12 | ✓ Full | Primary version |
| 3.13 | ✓ Full | Tested in CI |
| 3.14 | ✓ Full | Tested in CI |

Python 3.10 is no longer supported (end-of-life in October 2026).

## Compatibility Notes

### typing.override

`typing.override` is available in the standard library from Python 3.12.
Python 3.11 still requires `typing_extensions.override`, so the codebase keeps
`typing_extensions` as a dependency until Python 3.11 support is dropped.

### fiftyone

`fiftyone` is only used by the test dependencies to download the COCO dataset.
The tests import it lazily, so it does not affect normal library usage.
Newer `fiftyone` releases no longer depend on the legacy `glob2` package.

## Running Tests by Version

```bash
# Test all versions
uv run nox -s compatibility

# Test specific version
uv run nox -s compatibility-3.11
uv run nox -s compatibility-3.12
uv run nox -s compatibility-3.13
uv run nox -s compatibility-3.14
```

## CI Configuration

The GitHub Actions CI pipeline tests compatibility on:

- 3.11
- 3.12
- 3.13
- 3.14

The full test suite runs on the primary version, 3.12. See
`.github/workflows/ci.yml` for details.

## Package Dependencies

Core packages require:

- `typing_extensions` - For the `override` decorator on Python 3.11
- `pyarrow` - For DataFrame and Parquet operations

## Related Pages

- [YAML Basics](yaml_basics.md)
- [Configuration](configuration/overview.md)
- [CLI Reference](cli.md)
