# Contributing to DQM-ML

We welcome contributions! Whether you're fixing a bug, adding a new **Metric**, or improving documentation, your help makes DQM-ML better for everyone.

> **See also:** [Concepts](formal_concepts.md) for definitions of **Metric**, **Batch Metric**, **Processor**, and related terminology used throughout this page.

This guide walks you through setting up your development environment and adding new features.

## What Can You Contribute?

- 🐛 **Bug fixes** - Found an issue? Let us know and potentially fix it!
- 📊 **New metrics** - Add completeness, representativeness, or domain gap calculations
- 📝 **Documentation** - Improve docs, add examples, translate
- 🎨 **Better tests** - Increase test coverage, add edge cases
- 💡 **Ideas** - Open an issue with your suggestions

## Quick Start

```mermaid
flowchart TD
    A[Clone the Repo] --> B[Create Branch]
    B --> C[Make Your Changes]
    C --> D[Run Tests]
    D --> E[Run Lint & Type Check]
    E --> F[Submit Pull Request]
    F --> G[Code Review]
    G --> H[Merge & Celebrate!]
    
    style A fill:#e3f2fd
    style D fill:#fff3e0
    style E fill:#ffebee
    style G fill:#fff3e0
    style H fill:#e8f5e9
```

## Development Environment Setup

We use [uv](https://github.com/astral-sh/uv) for fast development and workspace management.

### 1. Prerequisites

```bash
# Clone the repository
git clone https://github.com/Safenai/dqm-ml-workspace
cd dqm-ml-workspace

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install git-lfs for large test files
sudo apt-get install git-lfs
git lfs pull
```

### 2. Initialize Workspace

```bash
# This installs all dependencies
uv sync
```

### 3. Install Pre-commit Hooks

```bash
# Runs checks before every commit
uv run pre-commit install
```

## Quality Standards

Before submitting a PR, please ensure all checks pass:

```bash
# Run all quality checks
uv run nox

# Or run individual checks:
uv run nox -s test       # Run tests
uv run nox -s lint       # Check code style
uv run nox -s lint_fix   # Auto-fix style issues
uv run nox -s type_check # Type checking
uv run nox -s docs       # Build documentation
```

## Adding a New Metric

Here's how to add your own metric to DQM-ML:

### 1. Create the Processor Class

Inherit from `DatametricProcessor` and implement the required methods:

```python
from dqm_ml_core.api.data_processor import DatametricProcessor
import pyarrow as pa

class MyNewMetric(DatametricProcessor):
    """A brief description of what this metric measures."""

    def compute_features(self, batch, prev_features=None):
        """
        Extract features from raw data.
        Optional: compute per-sample features.
        """
        return {}  # Return dict of feature arrays

    def compute_batch_metric(self, features):
        """Compute intermediate statistics for one batch."""
        # Example: count non-null values
        return {"count": pa.array([len(features)]), "sum": pa.array([...])}

    def compute(self, batch_metrics=None):
        """Aggregate batch results into final metric."""
        # Compute final score from accumulated batch stats
        return {"my_metric_score": pa.array([0.95])}

    def compute_delta(self, source, target):
        """Optional: Compare two datasets."""
        return {"delta_score": pa.array([...])}
```

### 2. Register via Entry Points

Add this to your package's `pyproject.toml`:

```toml
[project.entry-points."dqm_ml.metrics"]
my_new_metric = "my_package:MyNewMetric"
```

### 3. Add Tests

Create a test file in the `tests/` directory. Use existing tests as templates.

## Non-Code Contributions

You don't need to write code to contribute to DQM-ML!

**Documentation**

- Improve existing docs
- Add examples and tutorials
- Fix typos and improve clarity
- Translate documentation

**Testing**

- Report bugs you find
- Test on different platforms
- Suggest edge cases we haven't covered
- Review pull requests

**Community**

- Answer questions in discussions
- Share your use case
- Write blog posts or tutorials
- Speak at meetups or conferences

**Design**

- Suggest UI improvements for CLI
- Design better documentation layouts
- Create logos or graphics

## Submit changes for review

**Step 1: Create a Branch**

```bash
git checkout -b your-feature-name
```

**Step 2: Make Your Changes**

Follow the [quality standards](#quality-standards) below.

**Step 3: Submit a Pull Request**

1. Push: `git push origin your-feature-name`
2. Copy paste the link in the terminal into your browser
3. Select dev branch instead of main (default)
4. Click "Compare & pull request"
5. Fill out the PR template
6. Submit!
7. Reviewers will read your submission

**Tips for First-Timers**

- Start with documentation improvements (easier to review)
- Don't worry about making mistakes — we all started somewhere
- Ask questions in the PR if you're unsure
- It's okay if your first PR takes a few attempts

## Best Practices

Following these patterns keeps the codebase consistent:

| Practice | Why it matters |
|----------|----------------|
| **Streaming-friendly** | Keep `compute_batch_metric` lightweight - only compute what's needed for final aggregation |
| **Use PyArrow** | Ensures compatibility with the rest of the pipeline |
| **Add docstrings** | Helps others understand and use your metric |
| **Write tests** | Keeps bugs from being introduced |

### Good Docstring Example

```python
def compute(self, batch_metrics: dict | None = None) -> dict[str, pa.Array]:
    """Compute final dataset-level metric from batch statistics.

    Args:
        batch_metrics: Dictionary of aggregated batch statistics.

    Returns:
        Dictionary containing the final metric values.
    """
    # Your code here
```

## Testing Strategy

This section describes how tests are organized in DQM-ML and how to add new tests.

### Test Organization

```mermaid
flowchart TB

    subgraph "Test Types"
  
        U[Unit Tests <br/>tests/unit]
        I[Integration Tests<br/>tests/integration]
        C[CLI Tests <br/>tests/cli]
    end
    
    subgraph "Test Pyramid"
        TP[Unit Tests - Fast, isolated<br/>Core logic, processors]
        TI[Integration Tests - Real data<br/>Pipelines, loaders, metrics]
        TC[CLI Tests - End-to-end<br/>Commands, config files]
    end
    
    U --> TP
    I --> TI
    C --> TC

    F[Fixtures - Shared test data tests/integration/fixtures/ ]
    TP --> F
    TI --> F

```

### Test Directory Structure

```
tests/
├── conftest.py              # Pytest configuration & fixture imports
├── utils/                   # Utility functions for tests
│   ├── files.py            # File handling helpers
│   ├── jobs.py             # Job configuration helpers
│   └── plots.py            # Visualization helpers
├── unit/                    # Unit tests (fast, isolated)
│   ├── core/               # Core API tests (data_processor, metric_runner)
│   ├── pipeline/           # Pipeline tests (loaders, writers)
│   └── v2/                 # CLI wrapper tests
├── integration/             # Integration tests (synthetic data)
│   ├── fixtures/           # Test fixtures and data
│   │   ├── config.py      # Configuration fixtures
│   │   ├── data.py        # Data fixtures
│   │   ├── jobs.py        # Job configuration fixtures
│   │   └── paths.py       # Path fixtures
│   ├── test_completeness.py
│   ├── test_representativeness.py
│   ├── test_domain_gap.py
│   ├── test_visual_features.py
│   └── test_pandas_welding.py
└── cli/                     # CLI end-to-end tests
    ├── test_v2_wrapper.py
    └── test_job_cli.py
```

### Test Fixtures

DQM-ML uses pytest fixtures for reusable test data. Here's what's available:

| Fixture | Scope | Purpose | Usage |
|---------|-------|---------|-------|
| `test_path` | session | Tests directory path | All test files |
| `output_path` | session | Output directory for test results | Integration tests |
| `coco_data` | session | COCO dataset for domain gap tests | `test_domain_gap.py` |
| `normal_dist` | function | Normal distribution sample | Representativeness tests |
| `not_normal_dist` | function | Non-normal distribution | Statistical tests |
| `uniform_dist` | function | Uniform distribution | Statistical tests |
| `not_uniform_dist` | function | Non-uniform distribution | Statistical tests |
| `job_completeness` | function | Completeness job config | Pipeline tests |
| `job_representativeness` | function | Representativeness job config | Pipeline tests |
| `job_domain_gap` | function | Domain gap job config | Pipeline tests |
| `job_visual_features` | function | Visual features job config | Pipeline tests |

**Example using fixtures**:

```python
import pytest

def test_completeness_with_data(
    test_path: str,
    uniform_dist: Any
) -> None:
    """Test completeness metric with uniform distribution."""
    processor = CompletenessProcessor(
        name="test",
        config={"columns": {"input": ["feature"]}}
    )
    result = processor.compute({})
    assert result is not None
```

### Running Tests

```bash
# All tests with coverage report
uv run nox -s test

# Fast mode (skip slow tests)
uv run pytest -m "not slow"

# Specific test file
uv run pytest tests/integration/test_completeness.py

# With verbose output
uv run pytest -v tests/

# Run only unit tests
uv run pytest tests/unit/

# Run with coverage for specific package
uv run pytest --cov=packages/dqm-ml-core tests/
```

### Adding a New Test

1. **Choose test type**:
   - **Unit tests**: `tests/unit/package_name/` - for fast, isolated tests
   - **Integration tests**: `tests/integration/` - for real data and pipeline tests
   - **CLI tests**: `tests/cli/` - for end-to-end command testing

2. **Follow naming conventions**:
   - Test files: `test_*.py`
   - Test functions: `test_*`
   - Use descriptive names: `test_completeness_returns_valid_score`

3. **Use existing fixtures**:
   ```python
   def test_my_feature(test_path: str, uniform_dist: Any) -> None:
       # Your test code using fixtures
       pass
   ```

4. **Mark slow tests** (if your test takes >30s):
   ```python
   @pytest.mark.slow
   def test_slow_operation() -> None:
       # This test will be skipped with -m "not slow"
       pass
   ```

### Test Data Sources

| Type | Source | When to Use |
|------|--------|--------------|
| **Synthetic** | Generated via fixtures (normal_dist, uniform_dist) | Most unit/integration tests |
| **Real datasets** | COCO-2017 via `fiftyone.zoo` | Domain gap tests |
| **Example data** | `examples/config/` | CLI tests |

### Test Coverage & Results

After running tests, reports are generated:

| Report | Location | Description |
|--------|----------|-------------|
| Coverage HTML | `docs/reports/htmlcov/index.html` | Line-by-line coverage |
| Test Results HTML | `docs/reports/pytest/pytest_report.html` | Test execution report |
| Live Coverage | [GitHub Pages](https://safenai.github.io/dqm-ml-workspace/reports/htmlcov/) | Published coverage |

> **Tip**: Build the docs to generate these reports: `uv run nox -s docs_offline`

### CI/CD Testing

Tests run automatically on every push via GitHub Actions:

- **Lint**: Code style with ruff
- **Type Check**: Type safety with mypy
- **Test**: Full test suite with pytest
- **Docs**: Documentation build

See the [README](https://github.com/anomalyco/dqm-ml-workspace#readme) for current status badges.

## Getting Help

- 📖 Check the [Documentation](https://safenai.github.io/dqm-ml-workspace/) - Start here!
- 💬 Open an [Issue](https://github.com/Safenai/dqm-ml-workspace/issues) - For bugs or features
- 💭 Start a [Discussion](https://github.com/Safenai/dqm-ml-workspace/discussions) - For questions
- ⭐ Star us on [GitHub](https://github.com/Safenai/dqm-ml-workspace) - Motivate the team!

Thanks for considering contributing to DQM-ML!
