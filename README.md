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

This repository group all package derived from [dqm-ml](https://github.com/IRT-SystemX/dqm-ml/blob/main/README.md) to intiate what shall become dqm-ml v2.0.0

Documentation still remain in the main repository as we only deliver partial migration to the new API [dqm-ml](https://github.com/IRT-SystemX/dqm-ml/blob/main/README.md)

The library was originally developped in the programme

<div align="center">
    <img src="static/images/Logo_ConfianceAI.png" width="20%" alt="ConfianceAI Logo" />
    <h1 style="font-size: large; font-weight: bold;">dqm-ml</h1>
</div>

## other usefull documentations

- The [Rational](./docs/dqm-ml-v2.md) behind creation of V2 fro [dqm-ml](https://github.com/IRT-SystemX/dqm-ml)
- A [demonstration](TODO) of dqm usage to generate informations regarding exinsting datas using welding challenge results.
- [known limitation and evolution roadmap](./docs/roadmap.md)

## Dependency and bootsrap

### uv

We rely on uv for development for speed (multiple python versions, shared cache between venv, ...),
We also us git lfs for test / data files
We include legacy dqm-ml as a submodule

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install git-lfs
git lfs pull

git submodule update --init --recursive
```

The following command synchronize the workspace and allow to compute metrics

``` bash
uv sync 
source .venv/bin/activate
mkdir output
dqm-ml process -p packages/dqm-ml-pipeline/config/completeness.yaml

```

Other configuration exemple files can be found in this [directory](packages/dqm-ml-pipeline/tests/config/)

## Contents

- packages/*: the sources of packages build by this environment
- docs/*: documentation for dqml-ml-v2 no documentation in partial package to prevent documentation segmentation (TO BE MOVED FOR V2)
- .github: CI configuration
- src/*: empty workspace content for workspace
- uv.lock: reference version used for the environment

## Available commands

- uv sync : install workspace dependency (with --no-sync if you want to rely on uv.lock file)
- uv build --package <package_name> : build the define package
- uv run nox : execute by default the following sessions [lint, type_check, test]
- uv run nox -s <session_name> whith one of the available sessions
  - lint : check lint issues
  - lint_fix : use to correct several lint (TODO : check complementarity with fmt)
  - fmt : auto reformat code (use to correct lint warnings)
  - type_check : check type control with mypy
  - test_dev : execute basic tests
  - test : perform test on all packages with coverage results
  - compatibility : execute test for different python versions
  - licenses : check licences dependency of the repository
- uv publish (see publishing guide)

## Usage from your python code

- [jupyter notebook](packages/dqm-ml/examples/multiple_metrics_tests_v2.ipynb)
