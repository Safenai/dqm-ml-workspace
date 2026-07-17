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

Five plugin groups via Python `[project.entry-points]`:

| Entry Point Group | Purpose | Registration Location |
|---|---|---|
| `dqm_ml.metrics` | Metrics Processors (Completeness, Representativeness, Diversity) | `packages/dqm-ml-core/pyproject.toml` |
| `dqm_ml.features` | Features Processors (VisualFeatures, ImageEmbedding) | `packages/dqm-ml-images/pyproject.toml`, `packages/dqm-ml-pytorch/pyproject.toml` |
| `dqm_ml.gap` | Gap Processors (DomainGap) | `packages/dqm-ml-pytorch/pyproject.toml` |
| `dqm_ml.dataloaders` | Data Loaders (Parquet, CSV) | `packages/dqm-ml-job/pyproject.toml` |
| `dqm_ml.outputwriter` | Output Writers (Parquet) | `packages/dqm-ml-job/pyproject.toml` |

## CLI

Entry point: `dqm_ml` → `dqm_ml.__main__:execute`.
