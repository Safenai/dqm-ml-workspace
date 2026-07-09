# DQM-ML Project TODOs

This file tracks planned improvements, bug fixes, and refactoring tasks for the DQM-ML V2 project.

## General & Workspace

* [x] Improve test management in `noxfile.py`.
* [x] Finalize Autodoc configuration in `mkdocs.yml`.
* [x] Mark original repository as deprecated and reference this one. (Legacy dqm-ml submodule removed, now using dqm-ml as main CLI)
* [X] Set version to V2.0.0-rc upon completion of core migration. (CLI renamed from dqm-ml-v2 to dqm-ml)
* [ ] Choose between `pyyaml` and `ruamel.yaml` to standardize dependency.

## Core API (`dqm-ml-core`)

* [ ] Check if `scipy` dependency in `requirements.txt` is strictly necessary.
* [ ] Implement configuration-driven output metric names in `CompletenessProcessor`.
* [ ] Implement configuration-driven output metric names in `RepresentativenessProcessor`.
* [ ] Refactor `RepresentativenessProcessor` to support multi-sample tests (currently one-sample).
* [ ] Optimize Kolmogorov-Smirnov (KS) test to handle data without loading everything into memory.
* [ ] Make metadata export optional in `RepresentativenessProcessor`.
* [ ] Refactor `RepresentativenessProcessor` implementation for better performance.
* [ ] Refactor `registry.py` to use a common base class for all registries.
* [ ] Move dataloader/writer type checker to core once stabilized.
* [ ] Extend metrics to compute representativity of a set of features (multi-column), not just single columns.

## Pipeline & Orchestration (`dqm-ml-job`)

* [x] Rename `dqm-ml-pipeline` to `dqm-ml-job` to better reflect its role in MLOps.
* [ ] Add parameters to the CLI to pass file or directory paths directly as inputs for loaders.
* [ ] Improve generic parameter and log handling in `cli.py`.
* [ ] Implement proper metric ordering based on dependencies in `DatasetPipeline`.
* [ ] Validate input order of metric computation in the pipeline.
* [ ] Add specific command-line arguments for pipeline orchestration features.
* [ ] Standardize the combination format of classical and delta metrics.
* [ ] Implement chunking for output writing (Parquet) to handle cases where features exceed memory limits.
* [ ] Extend experimental usage of DuckDB to allow computation of metrics directly on groups without creating temporary files.

## Domain Specific Packages

### Image Processing (`dqm-ml-images`)

* [ ] Vectorize or parallelize visual feature computation in `visual_features.py`.
* [ ] Clean up `noqa` and type-checking workarounds in `visual_features.py`.
* [ ] Improve performance of image feature computation.

### Deep Learning (`dqm-ml-pytorch`)

* [ ] Test image path support in `image_embedding.py`.
* [ ] Fix type-checking errors in `image_embedding.py` and `domain_gap.py` (e.g., `vec` function).
* [ ] Improve configuration and available metrics check in `domain_gap.py`.
* [ ] Add proper error return codes to the Domain Gap API.
* [ ] Re-implement missing V1 metrics: Gini-Simpson, Simpson indices, Relative Diversity, PAD, CMD.
* [ ] Investigate result variations between V1 and V2 for KLMVN and FID metrics.

### New Domains

* [ ] Create a new package `dqm-ml-timeseries` dedicated to time series features.

## Data Formats & I/O

* [x] Support reading from CSV files in the pipeline.
* [ ] Support writing metric results in JSON and YAML formats.
* [ ] Implement database read/write support (SQL).

## Documentation

* [ ] Create a dedicated "How to create a new metric" guide (building on the contributing guide).
* [ ] Document performance configuration and optimization strategies for large-scale usage.

## Wrapper & CLI (`dqm-ml`)

* [ ] Move logging configuration to a dedicated module.
* [ ] Forward generic parameters from `init_log` to `logging.basicConfig`.
* [ ] Add filtering and extra parameters support for the dependency display command.
* [ ] Refine integration of packages to ensure only minimal required dependencies are installed.

## Scientific & Community

* [ ] Re-implement the Diversity metric (pending scientific discussion on target content and entropy position). See ROADMAP for full V1 metrics list.
* [ ] Rationalize the level of parametrization across all metrics.
* [ ] Scientific committee review: allocation of methods between Diversity and Representativeness.
* [ ] Expand supported reference distributions for Representativeness to reflect real-world data.
* [ ] Use base statistics (min, max, mean, std) to improve precision of batch-based metric implementations.

## Tests

* [ ] Handle long parameter lists in CLI tests (`test_v2_wrapper.py`, `test_job_cli.py`).
* [ ] Investigate high variance in metrics for the Pandas welding integration test.
* [ ] Add tests to validate filters on DataLoaders and combinations of `filter_row`.

## Configuration


