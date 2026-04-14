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
│   ├── dqm-ml/        # Main wrapper & CLI entry point
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

### Spell Checking

Run spell checking with cspell:

```bash
uv run nox -s spell
```

Configuration (from pyproject.toml):
- Checks all files: `files = ["**/*"]`

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

Two nox test sessions are available:

**test_dev** - Runs all tests without coverage:
```bash
uv run nox -s test_dev
```
- Runs ALL tests in `tests/` (including slow tests)
- No coverage reporting

**test** - Runs tests with coverage (for PRs):
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

## Documentation

Generate documentation in HTML format using mkdocs:

```bash
# Build documentation (outputs to docs/site/)
uv run nox -s docs

# Serve documentation locally (live reload)
uv run nox -s docs_serve
```

The documentation site is automatically built and deployed by the CI/CD pipeline to GitHub Pages.

## Documentation Guidelines

When editing documentation (README, docs/, etc.), follow these guidelines:

### Audience

The audience is the open source community, grouped by role:

#### Technical Users (primary)
- **Data Scientists** — ML practitioners checking dataset quality
- **ML Engineers / MLOps** — Building and monitoring data pipelines
- **Data Engineers** — Building reliable ETL pipelines
- **Software Engineers** — Integrating metrics into applications

#### Researchers
- **Research Scientists** — Academic papers on data quality methodology
- **Academics / Students** — Learning and teaching data quality concepts

#### Decision Makers
- **Tech Leads / Architects** — Deciding on data infrastructure
- **Product Managers** — Defining data quality requirements
- **Startup Founders** — Building AI products

#### Specialized Roles
- **Domain Experts** — Healthcare, finance — validating domain-specific data
- **AI Ethics / Governance** — Checking for bias, ensuring fairness
- **Enterprise Users** — Compliance, governance, audit

#### Community
- **Open Source Contributors** — Integrating metrics into other tools
- **Python Enthusiasts** — Exploring data quality metrics

### Tone and Wording

- **Welcoming and friendly** — Write as if explaining to a colleague
- **Accessible** — Don't assume deep technical knowledge of DQM-ML internals
- **Practical** — Focus on "how to" and "why" before details
- **Inclusive** — Avoid jargon; explain technical terms briefly
- **Respect technical levels** — Some know ML, others know Python, some neither

### Technical Level Guidelines

| Context | Example | Approach |
|---------|---------|----------|
| **Quick Start** | "Install and run in 2 minutes" | Keep simple |
| **API docs** | `CompletenessProcessor` | Explain, show minimal example |
| **Architecture** | Streaming pipeline | Explain "why" before "how" |
| **Configuration** | YAML examples | Copy-paste friendly |

### Best Practices

1. **Lead with the goal** — Tell readers what they'll learn/do
2. **Use concrete examples** — "Run this command"
3. **Link for depth** — README overview, docs/ details
4. **Explain abbreviations** — First use: "Maximum Mean Discrepancy (MMD)"
5. **Keep it scannable** — Tables, bullet points, code blocks
6. **Respect all levels** — Don't assume expertise

### What to Avoid

- **Assuming expertise** — New users may not know uv, Docker, or ML
- **Being patronizing** — Explain once, not repeatedly
- **Over-simplifying** — Respect intelligence
- **Inconsistent terminology** — Same term throughout

### Structure for Markdown Files

- **README.md**: High-level overview, quick start, key references
- **docs/*.md**: Detailed explanations, full examples, background
- **docs/metrics/*.md**: Detailed metric docs (configuration, parameters, usage)
- **docs/metrics.md**: Overview table, links to detailed metric pages

## Contributing

Agents are welcomed to contribute to this project. To contribute:

1. Follow all the rules defined in this document
2. Make your changes following the coding, linting, and testing rules
3. Ensure all tests pass before proposing a pull request
4. Submit a pull request for review

When contributing:
- Run linting: `uv run nox -s lint`
- Run type checking: `uv run nox -s type_check`
- Run tests: `uv run nox -s test`
- Generate documentation: `uv run nox -s docs`

### Testing GitHub Workflows Locally

The GitHub workflow can be tested locally using [act](https://github.com/nektos/act).

**Available jobs (from CI):**
- `quality` - Runs lint, spell, type_check (matrix job with 3 variants)
- `compatibility` - Runs tests on Python 3.12, 3.13 (matrix job)
- `test` - Runs full test suite on Python 3.12
- `pages` - Builds and deploys documentation

**List all jobs:**
```bash
act --list
```

**Run a specific job:**
```bash
# Run quality checks (lint, spell, type_check)
act -j quality -P ubuntu-24.04=ghcr.io/catthehacker/ubuntu:act-24.04

# Run compatibility tests
act -j compatibility -P ubuntu-24.04=ghcr.io/catthehacker/ubuntu:act-24.04

# Run test suite
act -j test -P ubuntu-24.04=ghcr.io/catthehacker/ubuntu:act-24.04
```

> **Caution:** act requires significant system resources and may take several minutes to run. Ensure no other Docker containers are running before executing act to avoid conflicts. For quick local validation, prefer running nox sessions directly (e.g., `uv run nox -s spell`).

Note: The `-P` flag is needed to specify the platform image for ubuntu-24.04.

### Committing and Pushing
- NEVER commit or push without user approval
- Always propose a meaningful commit title and message
- The message should summarize what changed (1-2 sentences), not list every file
- Use imperative mood ("Add", "Fix", "Update" - not "Added", "Fixed")
- Keep title under 72 characters
- Ask "commit?" before executing after proposing
- After commit succeeds, ask "push?" before pushing
- When pushing:
  - First check the current branch with `git branch -vv`
  - If on dev or main branch, refuse to push and ask user to create a new branch
  - Use `git push origin <branch_name>`

### Checking Quality Gates

All quality gates (lint, spell, type_check, tests, SonarCloud) must pass before merging.

**1. Check via GitHub CLI:**
```bash
gh pr view <MR_number> --json state,mergeable,statusCheckRollup
```

**2. Check SonarQube issues (required):**
```bash
curl -s "https://sonarcloud.io/api/issues/search?componentKeys=Safenai_dqm-ml-workspace&pullRequest=<MR_number>&statuses=OPEN,CONFIRMED" | jq '.total'
```

If `.total` > 0, there are issues to fix before merging.