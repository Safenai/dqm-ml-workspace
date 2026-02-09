# DQM ML repository

## Workspace common Badge and CI / CD informations

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

[github-actions-badge]: https://github.com/Safenai/dqm-ml-workspace/actions/workflows/ci.yml/badge.svg
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[nox-badge]: https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[mypy-badge]: https://www.mypy-lang.org/static/mypy_badge.svg

## Package last version available on pypi

* [![PyPI dqm-ml-core version][pypi-core-badge]](https://badge.fury.io/py/dqm-ml-core) :  `packages/dqm-ml-core`: Core API and standard metrics (Completeness, Representativeness).
* [![PyPI dqm-ml-pipeline version][pypi-pipeline-badge]](https://badge.fury.io/py/dqm-ml-pipeline) : `packages/dqm-ml-pipeline`: Orchestration, streaming data loaders, and output writers.
* [![PyPI dqm-ml-images version][pypi-images-badge]](https://badge.fury.io/py/dqm-ml-images) :`packages/dqm-ml-images`: Visual feature extraction metrics.
* [![PyPI dqm-ml-pytorch version][pypi-pytorch-badge]](https://badge.fury.io/py/dqm-ml-pytorch) : `packages/dqm-ml-pytorch`: Advanced metrics requiring PyTorch (Domain Gap).
* (not yet delivered as a package)`packages/dqm-ml-v2`: Main wrapper and CLI entry point.
* `packages/dqm-ml`: **Legacy** version (V1) delivered from original repository, excluded from the active workspace.

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
* **[Roadmap & Limitations](./docs/ROADMAP.md)**: Known issues and planned evolutions.
* **[Contributing](./docs/contributing.md)**: How to set up the development environment and contribute.

## Installation

Install the DQM-ML V2 framework with all available metrics and helpers using pip:
> :warning: **NOT YET AVAILABLE**, ONLY ON v2.0.0 but all functionality are available with detail install bellow

```bash
pip install "dqm-ml-v2[all]" 
```

Install the DQM-ML V2 framework by passing only needed optional dependecy:
> :warning: **NOT YET AVAILABLE**, ONLY ON v2.0.0 but all functionality are available with detail install bellow

```bash
pip install "dqm-ml-v2[notebooks, pytorch, job, images ]" 
```

Manualy install all packages:
> :warning: for version <v2.0.0> the dqm-ml version installed is the legacy version, you have access to the **process** command

```bash
pip install dqm-ml, dqm-ml-pipeline, dqm-ml-pytorch, dqm-ml-images" 
```

## Execution with cli provided **dqm-ml**

Run a metric processing job using a configuration file:

```bash
dqm-ml process -p examples/config/completeness.yaml
```

Other configuration examples can be found in the `examples/config/` directory.

## Call the same process from your script / code

```python
def compute_metric() -> None:
    """Example script to compute a metric using a YAML configuration."""

    # Load configuration file or create a dictionary structure with the same keys
    cur_file_path = os.path.abspath(__file__)
    config_path = os.path.join(os.path.dirname(cur_file_path), "../config/completeness.yaml")

    config: dict[str, Any] = {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

        # Execute the job with the loaded configuration, output are directlu saved to disk
        exec_qml_job(config["pipeline_config"])

        # A more granular API will be provided in future releases to access intermediate results

if __name__ == "__main__":
    compute_metric()
```

```bash
python examples/script/completness.py
```

this example can be found in `examples/script/completness.py'` and executed with

## Direct usage of metrics from your python code on data

* [jupyter notebook](packages/dqm-ml/examples/multiple_metrics_tests_v2.ipynb)

## Workspace Structure

