# Project

## Repository Architecture

Python monorepo using UV workspaces. Package dependency chain:

```
dqm-ml-core  (API, base metrics: completeness/representativeness/diversity)
  ← dqm-ml-job  (orchestration, I/O: dataloaders / output writers)
    ← dqm-ml-images  (visual features)
    ← dqm-ml-pytorch  (image embedding, domain gap)
      ← dqm-ml  (CLI entry point)
```

## Directory Structure

```
dqm-ml-workspace/
├── packages/
│   ├── dqm-ml-core/      # Core API & standard metrics
│   ├── dqm-ml-job/       # Orchestration, data loaders, output writers
│   ├── dqm-ml-images/    # Visual feature extraction
│   ├── dqm-ml-pytorch/   # PyTorch-based metrics (Domain Gap)
│   └── dqm-ml/           # Main wrapper & CLI entry point
├── tests/                # Test suite
├── docs/                 # MkDocs documentation
├── examples/             # Example configs and scripts
└── .agents/              # Agent guidelines (this directory)
```

## Plugin System

Three plugin groups via Python `[project.entry-points]`:

| Entry Point Group | Registration Location |
|---|---|
| `dqm_ml.metrics` | Each metric package's `pyproject.toml` |
| `dqm_ml.dataloaders` | `packages/dqm-ml-job/pyproject.toml:32` |
| `dqm_ml.outputwriter` | `packages/dqm-ml-job/pyproject.toml:37` |

## CLI

Entry point: `dqm_ml` → `dqm_ml.__main__:execute`.
