# AGENTS.md - Rules for Contributing Agents

This document outlines the rules and conventions that AI agents must follow when contributing to this project.

## Architecture

This is a Python monorepo using UV workspace. The project structure:

```
dqm-ml-workspace/
├── packages/
│   ├── dqm-ml-core/      # Core API & standard metrics (Completeness, Representativeness)
│   ├── dqm-ml-job/       # Orchestration, data loaders, output writers
│   ├── dqm-ml-images/   # Visual feature extraction
│   ├── dqm-ml-pytorch/  # PyTorch-based metrics (Domain Gap)
│   ├── dqm-ml-v2/      # Main wrapper & CLI entry point
│   └── dqm-ml/         # Legacy V1 (excluded from workspace)
├── tests/               # Test suite
└── src/                # Workspace-level code
```

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

## Coding Rules

### Linting

Run linting with:

```bash
uv run nox -s lint
```

Fixable error:
```bash
uv run nox -s lint_fix
```

Configuration (from pyproject.toml):
- Line length: 120
- Enabled rules: F, E, I, N, UP, RUF, B, C4, ISC, PIE, PT, PTH, SIM, TID

### Type Checking

Run type checking with:

```bash
uv run nox -s type_check
```

Configuration:
- Strict mode enabled
- Error codes: `deprecated`, `exhaustive-match`, `explicit-override`
- Ignore missing imports: enabled

## Testing Rules

### Running Tests

Run tests using:

```bash
uv run nox -s test
```

### Test Configuration

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

### Using Fixtures

Example test using fixtures:

```python
def test_representativeness(test_path: str, normal_dist: Any) -> None:
    """Test representativeness metric on normal distribution."""
    # test_path points to tests/ directory
    # normal_dist fixture creates test data
    ...
```

## Pre-commit Hooks

Before contributing, install pre-commit hooks:

```bash
uv run pre-commit install
```

## Dependencies

Install dependencies with:

```bash
uv sync
```

Install specific groups:

```bash
uv sync --group lint    # Linting tools
uv sync --group type_check  # Type checking tools
uv sync --group test   # Testing tools
```