# DQM ML repository

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/license/apache-2-0)![Python](https://img.shields.io/badge/python-3.9%20|3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)
![Repo Size](https://img.shields.io/github/repo-size/Safenai/dqm-ml-workspace)
[![PyPI version](https://badge.fury.io/py/dqm-workspace.svg)](https://badge.fury.io/py/dqm-workspace)
[![CI](https://github.com/Safenai/dqm-ml-workspace/actions/workflows/ci.yml/badge.svg)](https://github.com/Safenai/dqm-ml-workspace/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

This repository group all package derived from [dqm-ml](https://github.com/IRT-SystemX/dqm-ml/blob/main/README.md) to intiate what shall become dqm-ml v2.0.0

Documentation still remain in the main repository as we only deliver partial migration to the new API [dqm-ml](https://github.com/IRT-SystemX/dqm-ml/blob/main/README.md)

The library was originally developped in the programme

<div align="center">
    <img src="docs/static/images/Logo_ConfianceAI.png" width="20%" alt="ConfianceAI Logo" />
    <h1 style="font-size: large; font-weight: bold;">dqm-ml</h1>
</div>

The [Rational](./docs/dqm-ml-v2.md) behind creation of V2 fro dqm-ml

A [demonstration](TODO) of dqm usage to generate informations regarding exinsting datas using welding challenge results.

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
# Need parquet with image_bytes
# dqm-ml process -p packages/dqm-ml-pipeline/config/visual_features.yaml
# Need output of visual_features
# dqm-ml process -p packages/dqm-ml-pipeline/config/representativness.yaml
```

Other configuration exemple files can be found in this [directory](packages/dqm-ml-pipeline/tests/config/)

## Contents

* packages/*: the sources of packages build by this environment
* docs/*: documentation for dqml-ml-v2 no documentation in partial package to prevent documentation segmentation (TO BE MOVED FOR V2)
* .github: CI configuration
* src/*: empty workspace content (see TODO)
* uv.lock: version really used for the environment

## Available commands

* uv sync : install workspace dependency (with --no-sync if you want to rely on uv.lock file)
* uv build --package <package_name> : build the define package
* uv run nox : execute by default the following sessions [lint, type_check, test]
* uv run nos -s <session_name> whith one of the available sessions
  * lint : check lint issues
  * lint_fix : use to correct several lint (TODO : check complementarity with fmt)
  * fmt : auto reformat code (use to correct lint warnings)
  * type_check : check type control with mypy
  * test_dev : execute basic tests
  * test : perform test on all packages with coverage results
  * compatibility : execute test for different python versions
  * licenses : check licences dependency of the repository

* uv publish (see publishing guide)

## Usage from your python code

* [jupyter notebook](packages/dqm-ml/examples/multiple_metrics_tests_v2.ipynb)

## ROADMAP

* Finalize dqm-ml migration
  * doc integration in this repository
  * mark original repository as deprecated and reference this one
  * integrate the last metrics not implemented
  * put version as V2.0.0

* Support of other format as inputs and outputs
  * write output in json / yaml format for metrics

* New metrics, and improvement of current metrics configurations, base on next month experimentations.
