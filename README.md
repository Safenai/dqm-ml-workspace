# DQM ML repository

[![License: Apache 2.0][license-badge]](https://opensource.org/license/apache-2-0)
![Python][python-badge]
![Repo Size][size-badge]

[![CI][github-actions-badge]](https://github.com/Safenai/dqm-ml-workspace/actions)
[![Ruff][ruff-badge]](https://github.com/astral-sh/ruff)
[![uv][uv-badge]](https://github.com/astral-sh/uv)
[![Nox][nox-badge]](https://nox.thea.codes/en/stable/)
[![Checked with mypy][mypy-badge]](https://mypy-lang.org/)

[license-badge]: https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg
[size-badge]: https://img.shields.io/github/repo-size/Safenai/dqm-ml-workspace
[python-badge]: https://img.shields.io/badge/python-3.12%20|%203.13-blue.svg
[pypi-core-badge]: https://badge.fury.io/py/dqm-ml-core.svg
[github-actions-badge]: https://github.com/Safenai/dqm-ml-workspace/actions/workflows/ci.yml/badge.svg
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[nox-badge]: https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[mypy-badge]: https://www.mypy-lang.org/static/mypy_badge.svg

[![PyPI dqm-ml-core version][pypi-core-badge]](https://badge.fury.io/py/dqm-ml-core)
[![PyPI dqm-ml-pipeline version][pypi-pipeline-badge]](https://badge.fury.io/py/dqm-ml-pipeline)
[![PyPI dqm-ml-images version][pypi-images-badge]](https://badge.fury.io/py/dqm-ml-images)
[![PyPI dqm-ml-pytorch version][pypi-pytorch-badge]](https://badge.fury.io/py/dqm-ml-pytorch)

[pypi-core-badge]: https://badge.fury.io/py/dqm-ml-core.svg
[pypi-pipeline-badge]: https://badge.fury.io/py/dqm-ml-pipeline.svg
[pypi-images-badge]: https://badge.fury.io/py/dqm-ml-images.svg
[pypi-pytorch-badge]: https://badge.fury.io/py/dqm-ml-pytorch.svg

This repository groups all packages derived from [dqm-ml](https://github.com/IRT-SystemX/dqm-ml/blob/main/README.md) to initiate what shall become dqm-ml v2.0.0.

Documentation remains in this repository as we deliver the migration to the new API.

The library was originally developed in the program:

<div align="center">
    <img src="static/images/Logo_ConfianceAI.png" width="20%" alt="ConfianceAI Logo" />
    <h1 style="font-size: large; font-weight: bold;">dqm-ml v2</h1>
</div>

## Documentation

* **[Architecture & Rational](./docs/dqm-ml-v2.md)**: The "why" and "how" of V2.
* **[Metrics Guide](./docs/metrics.md)**: Detailed list of available metrics and their configurations.
* **[Configuration Guide](./docs/configuration.md)**: How to write pipeline configuration files.
* **[Roadmap & Limitations](./docs/roadmap.md)**: Known issues and planned evolutions.
* **[Contributing](./docs/contributing.md)**: How to set up the development environment and contribute.

## Installation

Install the DQM-ML V2 framework using pip:

```bash
pip install "dqm-ml-v2[all]"
```

## Execution

Run a metric processing job using a configuration file:

```bash
dqm-ml process -p examples/config/completeness.yaml
```

Other configuration examples can be found in the `examples/config/` directory.

## Workspace Structure

* `packages/dqm-ml-v2`: Main wrapper and CLI entry point.
* `packages/dqm-ml-core`: Core API and standard metrics (Completeness, Representativeness).
* `packages/dqm-ml-pipeline`: Orchestration, streaming data loaders, and output writers.
* `packages/dqm-ml-images`: Visual feature extraction metrics.
* `packages/dqm-ml-pytorch`: Advanced metrics requiring PyTorch (Domain Gap).
* `packages/dqm-ml`: **Legacy** version (V1), excluded from the active workspace.

## Usage from your python code

* [jupyter notebook](packages/dqm-ml/examples/multiple_metrics_tests_v2.ipynb)
