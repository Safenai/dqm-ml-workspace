# Python Compatibility

This document describes the Python version compatibility of DQM-ML.

## Supported Versions

| Python Version | Support Status | Notes |
|----------------|---------------|-------|
| 3.10 | ✓ Full | Tested in CI |
| 3.11 | ✓ Full | Tested in CI |
| 3.12 | ✓ Full | Primary version |
| 3.13 | ✓ Full | Tested in CI |

## Compatibility Notes

### Python 3.10/3.11 Limitations

Some test fixtures have limitations on older Python versions:

1. **fiftyone** - Used for COCO dataset downloads in tests
   - Works: Python 3.12+
   - Issue: `glob2` package has SyntaxError on 3.10/3.11
   - Fix: Lazy import to defer loading until needed

2. **typing.override** - Used for type hints
   - Python 3.13+ has `typing.override`
   - Fix: Use `from typing_extensions import override` for 3.10+

### Running Tests by Version

```bash
# Test all versions
uv run nox -s compatibility

# Test specific version
uv run nox -s compatibility-3.10
uv run nox -s compatibility-3.11
uv run nox -s compatibility-3.12
uv run nox -s compatibility-3.13
```

## CI Configuration

The GitHub Actions CI pipeline tests on:
- 3.10
- 3.11
- 3.12
- 3.13

See `.github/workflows/ci.yml` for details.

## Package Dependencies

Core packages require:
- `typing_extensions` - For `override` decorator compatibility
- `pyarrow` - For DataFrame operations, version requirements vary by Python version

## Related Pages

- [YAML Basics](yaml_basics.md)
- [Configuration](configuration.md)
- [CLI Reference](cli.md)