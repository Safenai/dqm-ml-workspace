# Testing

## Running Tests

Two nox test sessions are available:

**test_dev** — Runs all tests without coverage:
```bash
uv run nox -s test_dev
```
- Runs ALL tests in `tests/` (including slow tests)
- No coverage reporting

**test** — Runs tests with coverage (for PRs):
```bash
uv run nox -s test
```
- Runs only non-slow tests (`-m "not slow"`)
- Includes coverage reporting

**Running a single test:**
```bash
# Run a specific test file
uv run nox -s test_dev -- tests/cli/test_quickstart.py

# Run a specific test
uv run nox -s test_dev -- -v "tests/cli/test_quickstart.py::TestQuickstartCLI::test_completeness_cli_with_config"
```

**Adding extra pytest arguments:**
```bash
uv run nox -s test_dev -- -v           # verbose output
uv run nox -s test_dev -- -k "pattern" # filter by name
uv run nox -s test_dev -- -q           # quiet mode
```

## Test Configuration

- **pytest** is the test framework
- Use **fixtures** from `tests/conftest.py` when needed (e.g., `test_path`, `coco_data`, `uniform_dist`, `normal_dist`)
- Fixtures are session-scoped for expensive data creation
- Mark slow tests with `@pytest.mark.slow` (can be deselected with `-m "not slow"`)

### Pytest Options

- Timeout: 300 seconds per test
- Strict markers enabled
- Strict config enabled
- Warnings treated as errors
- `xfail_strict = true`

## Writing Integration Tests

Pattern — session fixture generates YAML config from template → `execute()` runs pipeline → read output Parquet → assert with `pytest.approx()`.

Fixture scopes:
- `session`: expensive data generation (COCO download, 1M-row Parquet files)
- `function`: temp output paths cleaned up after test

Canonical ref: `tests/integration/test_completeness.py:15`.

Rules:
- Parametrize by config variant.
- Mark inference-heavy tests `@pytest.mark.slow` (CI uses `-m "not slow"`).
- Read output back with `pq.read_table()`, convert to pandas for assertion convenience.

Example using fixtures:

```python
def test_representativeness(test_path: str, normal_dist: Any) -> None:
    """Test representativeness metric on normal distribution."""
    # test_path points to tests/ directory
    # normal_dist fixture creates test data
    ...
```

## Testing CI Workflows Locally (act)

The GitHub workflow can be tested locally using [act](https://github.com/nektos/act).

**Available jobs:**
- `quality` — Runs lint, spell, type_check (matrix job with 3 variants)
- `compatibility` — Runs tests on Python 3.12, 3.13 (matrix job)
- `test` — Runs full test suite on Python 3.12
- `pages` — Builds and deploys documentation

**Run a specific job:**
```bash
act -j quality -P ubuntu-24.04=ghcr.io/catthehacker/ubuntu:act-24.04
act -j test -P ubuntu-24.04=ghcr.io/catthehacker/ubuntu:act-24.04
```

> **Caution:** act requires significant system resources. Ensure no other Docker containers are running before executing. For quick local validation, prefer running nox sessions directly.

## Pipeline Timing Information

Pipeline execution in CI typically takes ~25 minutes:
- **code_quality stage**: ~5 minutes (linting, spell checking, type checking)
- **test stage**: ~20 minutes (running the full test suite)
