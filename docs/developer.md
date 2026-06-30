# Developer guide 


## how to modify and or contribute to dqm-ml 

[Contribution Guide](./contributing.md)

## What is currently planned

The [Roadmap](./ROADMAP.md), do not hesitate to propose adjustments

## Packages documentations

- [Core](./packages/dqm-ml-core.md)
- [Job](./packages/dqm-ml-job.md)
- [Images](./packages/dqm-ml-images.md)
- [PyTorch](./packages/dqm-ml-pytorch.md)
- [CLI](./packages/dqm-ml.md)

## Code quality checks

All checks run via [mise](https://mise.jdx.dev/) tasks defined in `.mise.toml`.

```bash
mise code_quality   # Lint (ruff) + type check (pyright)
mise spell          # Spell check (cspell)
mise test           # Test with coverage
mise test_custom    # Run specific tests with custom args (logs to logs/)
mise complexity     # Code complexity (complexipy, max McCabe 15)
```

### Spell check

**Prerequisite** &mdash; `cspell` depends on the `hunspell` C extension:

```bash
sudo apt install libhunspell-dev
```

Then run:

```bash
mise spell
```

### Complexity

Uses [complexipy](https://pypi.org/project/complexipy/) to enforce max McCabe complexity of 15 across `packages/`:

```bash
mise complexity
```

# Tests reports 

- [Results](./reports/pytest/pytest_report.html)
- [Coverage](./reports/htmlcov/coverage_report.html)
- [Performance](./TBD1.md)