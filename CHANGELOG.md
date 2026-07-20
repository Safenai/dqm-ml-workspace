# Changelog

## 2.0.1 (2026-07-20) — Fix pytorch extra, add meta-package scenarios

* **Fix:** `pytorch` extra in `pyproject.toml` incorrectly depended on `dqm-ml-images` instead of `dqm-ml-pytorch`
* **Added:** Meta-package test scenarios (10–13) exercising `dqm-ml[job]`, `[pytorch]`, `[images]`, and `[all]` extras end-to-end
* **Removed:** `test_pypi_prerelease.sh` — temporary script used while test.pypi.org configuration was not ready; now superseded by `test_testpypi_prerelease.sh`
* **Improved:** `test_pypi_release.sh` now uses explicit `--index-url` for PyPI resolution
* **Updated:** Packaging documentation to reflect 13 scenarios and script removal

## 2.0.0 (2026-07-17) Release

* merged in main
* tagged 2.0.0

## 2.0.0-rc4 (2026-07-17) — ProcessorRunner improvements, package isolation testing, configurable seeds

This release addresses review feedback from 2.0.0-rc3, enhancing ProcessorRunner with direct image-to-domain-gap support, adding package isolation smoke tests, and introducing configurable test seeds.

### ProcessorRunner Enhancements

* `ProcessorRunner.run_gap()` now accepts raw images directly via optional `features` parameter — compute domain gaps from image bytes without manual embedding extraction.
* New `_df_to_record_batch()` helper for robust DataFrame-to-PyArrow conversion, handling `FixedSizeListArray` for embeddings.
* Renamed `metrics_processors` parameter to `processors` in `run()` for clarity.
* Separated `select_columns` intermediate results from final `batch_features` to avoid metric pollution.

### Package Isolation Testing

* New `tests/packaging/` directory with smoke tests for all 5 packages: dqm-ml, dqm-ml-core, dqm-ml-images, dqm-ml-job, dqm-ml-pytorch.
* Smoke scripts verify imports, basic functionality, and YAML-based execution in isolated virtual environments.
* New `tests/fixtures/packaging_fixtures.py` with `isolated_venv` and `wheels_dir` fixtures.
* New `docs/packaging-tests.md` and `docs/testing.md` documentation.

### Configurable Test Seeds

* New `tests/utils/seeds.py` with `get_test_seed()` function reading `DQM_ML_TEST_SEED` env var (default 42).
* Session-scoped `test_seed` fixture in `tests/conftest.py`.
* Updated ~35 test files and YAML templates to use configurable seeds instead of hardcoded values.
* Seeds injected via `tests/utils/jobs.py:generate_job()` for YAML template generation.

### CI Improvements

* tqdm progress bars suppressed in CI via `TQDM_DISABLE=1` in GitHub Actions and GitLab CI.
* `noxfile.py` now sets `DQM_ML_TEST_SEED=42` in all test sessions.
* `.mise.toml` exports `DQM_ML_TEST_SEED=42` for local development.

### Documentation

* New `docs/testing.md` with test organization, directory structure, and adding tests guide.
* New `docs/packaging-tests.md` with package isolation testing documentation.
* Improved READMEs for dqm-ml-core (+134 lines), dqm-ml-images (+100 lines), dqm-ml-pytorch (+252 lines).

### Bug Fixes

* S3 support fixes in CLI and processor (`dqm_ml_job.cli`, `dqm_ml_job.utils.s3`).
* CI tag regex and PyPI publish workflow fixes.docs/index.md 

## 2.0.0-rc3 (2026-06-30) — Pipeline DAG, Pydantic configs, example notebooks, and documentation

This release introduces a consistent configuration with Pydantic models and validation, adds a topological processor DAG, example notebooks and a user guide, improves domain-gap algorithms and image embedding, and refactors tests to generate synthetic data inline.

### Issue ticket number and link

* fix: [#13](https://github.com/Safenai/dqm-ml-workspace/issues/13) - Example Notebooks
* fix: [#10](https://github.com/Safenai/dqm-ml-workspace/issues/10) - User guide

### Pipeline Refactoring

* Topological sort for processor DAG — generators before consumers — with configurable execution order via `DatasetJob.execute()` → `topological_sort()`.
* Accumulate-then-flush mode for single-path output patterns.
* Memory-threshold parsing and configurable `compute_max_memory`.

### Configuration Models (Pydantic)

* New `dqm_ml_core.models` package with Pydantic models for processor, output, dataloader, and global configs.
* Refactored parquet output writer to use Pydantic config with column exclude list, S3 path-prefix fallback, and `FixedSizeListArray` metric filtering.
* Refactored S3 filesystem with `StorageConfig` model, retry strategies, role-based access, env-var fallback, and checksum validation.

### Domain Gap Improvements

* KLMVN variance-eps dampening and FID epsilon regularization.
* PAD with `CalibratedClassifierCV`.
* CMD multi-channel spatial moments.
* Wildcard embedding column patterns and per-selection path-prefix injection.
* New `dqm_ml_core.utils.matching` with include/exclude pattern resolution.

### Image Embedding

* Multi-input-column support, lazy model loading, auto device resolution.
* Column prefix/suffix, S3 per-column path prefixes, configurable failure handling with rate thresholds.

### Representativeness

* Mean-std estimation, expected-counts method, per-column distribution parameters, improved model validation, path-prefix for sample paths.

### Documentation & Examples

* Restructured docs: flat files → `docs/configuration/*.md`, new metrics documentation, formal concepts page.
* Added scenario example configs, debug pipeline script, and example notebooks for each metric.
* Updated user guide in README with installation, metrics, configuration, and usage sections.
* Schema generation utilities and `docs/schema/config.json`.
* Google-format docstrings across source packages and test files.

### Testing

* New integration tests: batch invariance, pipeline ordering, pipeline scenarios, path prefix, output columns, pipeline data flow, full story, features embeddings.
* New unit tests: matching, registry, representativeness, error policy, output configs, job, pandas loader, embedding edge cases, visual feature columns, domain gap processor, domain gap synthetic ordering.
* New property-based tests: diversity, domain gap, representativeness, and visual features properties, plus visual features stress.
* Stress image fixtures, expanded expected output YAML.
* Updated config templates and benchmark stability improvements.

### Code Quality

* Tests refactored to reduce cognitive complexity
* Synthetic data generated by tests at runtime, replacing 36 stale LFS-tracked test data files (parquet + JPGs).
* Type checking improvments.
* Removed LFS caching steps from CI workflows.
* Cleaned up legacy CLI tests, stale example configs, and outdated documentation.

## 2.0.0-rc2 (2026-06-03) - S3 support, pipeline refactoring, new algorithms, code quality

This release adds S3 filesystem support, refactors the metric pipeline with dependency ordering, introduces diversity metrics and 4 new domain gap algorithms, and improves code quality.

### Issue ticket number and link

* fix: [#18](https://github.com/Safenai/dqm-ml-workspace/issues/18) - Integrate lexical and visual diversity metrics to v2
* fix: [#17](https://github.com/Safenai/dqm-ml-workspace/issues/17) - Integrate categorical data diversity metric to v2
* fix: [#16](https://github.com/Safenai/dqm-ml-workspace/issues/16) - Integrate Domain Gap metric MMD to v2
* fix: [#15](https://github.com/Safenai/dqm-ml-workspace/issues/15) - Integrate Domain Gap metric PAD to v2
* fix: [#14](https://github.com/Safenai/dqm-ml-workspace/issues/14) - Integrate Domain Gap metric CMD to v2
* fix: [#4](https://github.com/Safenai/dqm-ml-workspace/issues/4) - Computed values differ between V1.1.2 and new implementation proposed for domain gap

### S3 Filesystem Support

* Shared S3 utility (`dqm_ml_job.utils.s3.get_s3_filesystem`) with env-var and dict-based configuration.
* S3 support in Parquet data loading, output writing, image embedding, and visual features processors.
* `region` parameter for providers like OVH S3.
* Image loading improvements: S3-first fallback, Parquet-to-Pillow `BytesIO` fix.

### Pipeline Refactoring

* Topological sort for processor ordering (generators before consumers).
* Delta metrics rewritten with PyArrow `concat_tables` instead of manual dict merging.
* Accumulate-then-flush mode for single-path output patterns.
* Dataloader column injection in feature outputs.

### Diversity Metrics (new)

* New `DiversityProcessor` with streaming batch-level computation.
* Four indices: Simpson, Gini-Simpson, Shannon, and Richness.
* Integration test fixtures and synthetic data generators.

### Domain Gap Algorithms (new)

* 4 new delta metrics: **MMD-RBF**, **MMD-Poly**, **PAD**, **CMD** with multi-layer support.
* Image embedding refactoring and visual_features simplification for domain gap pipeline.
* Updated representativeness processor for compatibility.
* New domain gap benchmark suite and expanded integration fixtures.
* Updated configuration templates and documentation.

### .agents/ Refactoring

* Replaced 364-line inline `AGENTS.md` with a shim pointing to `.agents/AGENTS.md`.
* 7 standalone agent files: architecture, code style, git workflow, testing, project docs, and agent guidelines.

### Code Quality & Bug Fixes

* Renamed cryptic variables (`self.fx` → `self.feature_extractor`, `vec` → `sum_fixed_to_vector`, etc.).
* Fixed h1/h2 variable-shadowing bug in Wasserstein loop (was using only first histogram dimension).
* Fixed RepresentativenessProcessor double-count bug (`exp_probs * total_count` squaring total).
* Reduced cognitive complexity below 15 in 5 modules by extracting focused helper methods (diversity, visual_features, domain_gap, image_embedding, job).
* Replaced legacy `np.random.*` calls with seeded `default_rng()` Generator API for reproducible results.
* Fixed type/security issues: missing `gamma="auto"` in SVC, `logger.error` → `logger.exception`, hardcoded `/tmp` → `tempfile.gettempdir()`.
* Fixed `dataset_root_path` being set to string `"None"` instead of Python `None`.
* Consolidated duplicate string literals into module constants; fixed shellcheck `[` → `[[`.
* Added missing docstrings; corrected error message grammar; translated French comments.
* Removed unused Dockerfile and container CI workflow.

### Packaging & Versioning

* Removed pinned version requirements for workspace-level setuptools-scm.
* Fixed package discovery in `pyproject.toml`.
* Configured PyTorch UV index for CPU torch downloads.

## 2.0.0-rc (2026-04-10) - Release Candidate for V2

This release is the V2.0.0 release candidate, featuring major refactoring, security fixes, and documentation improvements.

### Issue ticket number and link

* fix: [#61](https://github.com/Safenai/dqm-ml-workspace/issues/61) - Python 3.10/3.11 compatibility
* fix: [#60](https://github.com/Safenai/dqm-ml-workspace/issues/60) - Fix parquet dataloader with image_path column
* fix: [#59](https://github.com/Safenai/dqm-ml-workspace/issues/59) - Data selection tests for CLI
* fix: [#41](https://github.com/Safenai/dqm-ml-workspace/issues/41) - Security warnings addressed
* fix: [#31](https://github.com/Safenai/dqm-ml-workspace/issues/31) - Test package dependencies upgraded
* fix: [#29](https://github.com/Safenai/dqm-ml-workspace/issues/29) - dqm-ml-pipeline renamed to dqm-ml-job
* fix: [#28](https://github.com/Safenai/dqm-ml-workspace/issues/28) - Tests moved to workspace root
* fix: [#27](https://github.com/Safenai/dqm-ml-workspace/issues/27) - Multi data selection configuration
* fix: [#26](https://github.com/Safenai/dqm-ml-workspace/issues/26) - Domain gap computation for multiple selections
* fix: [#25](https://github.com/Safenai/dqm-ml-workspace/issues/25) - Multiple data selection in same job
* fix: [#24](https://github.com/Safenai/dqm-ml-workspace/issues/24) - Progress messages during post-processing
* fix: [#22](https://github.com/Safenai/dqm-ml-workspace/issues/22) - Output folder auto-creation
* fix: [#11](https://github.com/Safenai/dqm-ml-workspace/issues/11) - MkDocs documentation generation

### Breaking Changes

* **Package rename**: `dqm-ml-v2` → `dqm-ml` (CLI entry point)
* **CLI command**: `dqm-ml-v2` → `dqm-ml`
* **Legacy removal**: Removed dependency on legacy `dqm-ml` submodule

### Security Fixes (10 vulnerabilities fixed)

| Package | Old Version | New Version |
|---------|-------------|-------------|
| strawberry-graphql | 0.287.3 | 0.314.3 |
| tornado | 6.5.4 | 6.5.5 |
| cryptography | 46.0.4 | 46.0.7 |
| requests | - | 2.33.0 |
| pygments | - | 2.20.0 |

CVEs addressed: CVE-2026-35526, CVE-2026-35523, CVE-2026-31958, CVE-2025-47287, CVE-2026-26007, CVE-2026-34073, CVE-2026-37990, CVE-2026-25645

### New Features

* **Test Strategy Documentation**: Comprehensive testing docs in `contributing.md` with mermaid diagrams
* **Git LFS Guide**: Added to `quickstart.md` for development setup
* **Test Fixtures**: Session-scoped fixtures (`coco_data`, `normal_dist`, `uniform_dist`, etc.)
* **Test Utils**: New `tests/utils/` with helpers for files, jobs, plots
* **tqdm dependency added**: Added to `dqm-ml-job` for progress display

### Documentation Improvements

* **AGENTS.md**: New contribution guidelines for AI agents
* **Test Strategy**: Complete testing documentation with diagrams
* **Git LFS Section**: Added to quickstart for large file handling
* **TODO/ROADMAP Sync**: Aligned V1 metrics lists and version references
* **Mermaid Diagrams**: Fixed dependency direction (core ← job, not →)
* **All docs updated**: Replaced `dqm-ml-v2` references with `dqm-ml`

### Documentation Restructuring

* **New CLI Reference**: `docs/cli.md`
* **New YAML Basics**: `docs/yaml_basics.md`
* **New Data Loaders**: `docs/dataloaders.md`
* **New Metrics Computation**: `docs/metrics_computation.md`
* **New Python Compatibility**: `docs/python_compatibility.md`
* **Mermaid diagrams**: Config structure visualization
* **Navigation**: Organized into separate pages in mkdocs.yml

### Technical Changes

* Pinned `mkdocs-jupyter<0.26` to fix api-autonav compatibility
* Refactored `conftest.py` - restructured fixtures
* Added deprecation warning filters for strawberry and fiftyone

### New CLI and Integration Tests

* Test domain gap matrix between class pairs
* Test quickstart examples with completeness config
* New test fixtures and config files in `tests/fixtures/getting_started/`
* Test data: COCO dataset with class information (102+ classes)
* 18 tests added (unit + CLI)

### Python Compatibility (Issue #61)

* **Python 3.10/3.11 support added**
* Fixed `typing.override` import: Changed from `typing` to `typing_extensions` in 9 package files
* Fixed fiftyone lazy import in test fixtures (glob2 SyntaxError workaround)
* Fixed domain gap delta metrics: Handle both pa.Array and scalar returns from compute_delta
* Added `_to_pa_array` helper for robust type conversion in job.py
* CI now tests: Python 3.10, 3.11, 3.12, 3.13
* New documentation: `docs/python_compatibility.md`

### Data Selection Features (Issues #25, #27, #59, #60)

* **split_by configuration**: Create multiple selections from single dataloader based on column values
* **split_values**: Specify values to create selections (e.g., split_by: class, split_values: [dog, cat])
* **filter configuration**: Filter rows by column values
* Fixed parquet dataloader with image_path column handling
* New fixtures: `coco_classes`, `bird`, `elephant` for domain gap testing
* Example configs: `domain_gap_split_2classes.yaml`, `domain_gap_split_top10.yaml`, `domain_gap_matrix.yaml`

### Dependencies Updated

* fiftyone: 1.14.1 → 1.13.0
* torch: 2.10.0 → 2.11.0
* torchvision: 0.25.0 → 0.26.0
* Many other transitive dependencies upgraded

### Testing

* 72 tests passing (18 new tests)
* Integration test fixtures for real data testing
* CLI end-to-end tests
* Test coverage reports available in `docs/reports/`
* Python compatibility tests: 3.10, 3.11, 3.12, 3.13

## 1.1.6 (2026-02-10) - upgrade dependency and security check

This release is dedicated to rename dqm-ml-pipeline as sqm-ml-job, correct security issues detected by github, as well as initiated template for issues, feature and merge request

### Issue ticket number and link

* fix: [#42](https://github.com/Safenai/dqm-ml-workspace/issues/42)
* fix: [#31](https://github.com/Safenai/dqm-ml-workspace/issues/31)
* fix: [#31](https://github.com/Safenai/dqm-ml-workspace/issues/31)

### Other details on several changes

* Security warning regarding dependency used in uv.lock as reference package version
  * `urllib3 "2.6.2" => "2.6.3"` : Decompression-bomb safeguards bypassed when following HTTP redirects (streaming API) High
  * `torch 2.7.1 => 2.10.0` : PyTorch Improper Resource Shutdown or Release vulnerability Moderate
  * `filelock "3.20.1" => "3.20.3"` : filelock Time-of-Check-Time-of-Use (TOCTOU) Symlink Vulnerability in SoftFileLock Moderate
  * `pynacl "1.6.1" => "1.6.2"` : libsodium has Incomplete List of Disallowed Inputs Moderate
  * `virtualenv "20.35.4" => "20.36.1"` : virtualenv Has TOCTOU Vulnerabilities in Directory Creation Moderate

* As `fiftyone` use a deprecated version of `strawberry-graphql` we pin the version to "`strawberry-graphql==0.287.3`

* Other version change associate to uv sync --upgrade
  * default package version used for development and test upgraded, but without increasing min value dependencies

* Remove dependency needed for test from default installation, only install when targetted nox command are executed

* github-code-quality[bot] findings
  * SECURITY.md update
  * `Unkow comand`=> `Unknown command` miss spelling

* add default issues template to github workflow

* documentation presentation improvment
  * Add light/dark mode
  * Add dynamic horizontal size for content
  * Disable expand by default in navigation
  * Add navigation tab
  * Add navigation path
   
* connection to sonarqube for code quality

## 1.1.5 (2026-02-09)

* Quality correction / ci improvements **Fixed**
  * `noxfile.py`: Added a `docs` session to build the documentation. (Fixes #6)
  * `noxfile.py`: The `format` nox session now runs independently from the `lint` session. (Fixes #8)
  * `dqm_ml_job/outputwriter/parquet.py`: Ensured output directory is created if it does not exist, preventing crashes. (Fixes #22)  
  * `.github\ci.yml` : adjust security permissions and other code quality checks

* Repository organization changes :
  * Fixes : #28 : test moves to root of the workspace, and cover improved to 80%
  * fixes : #11 #12 : publication of mkdocs content document integrated

* Documentation :
  * First documentation structure proposed, to be upgraded with feedback, it integrate fixes for #9 and #10

* Conceptual change:
  * Notion of data selection introduced, and metrics are computed on a data selection, not a data loader.
  * delta metrics are compare between all dataselection available in the data selection.

* Configuration changes
  * parquet data loader  (fix : #27)
  * comparison of multiple data selection metrics fix : #25 and fix : #26
  * progression bar (optional) has been added to follow computation in the cli (fix : #24)

* Limitations not delivered in v1.1.5
  * Change of dml-ml-pipeline to dqm-ml-job in future release for consistency #29

## 1.1.4 (2025-12-18)

* github ci
* add changelog
* initiate release note and roadmap
* match of dqm-ml version number

## 0.0.5 (2025-12-17)

Version use to check compatibility with legacy dqm-ml, after this we will match for this repository the versions with dqm-ml
Version content :

* dqm-ml-core : API proposal for future generic V2 api to allow generic usage of API
* dqm-ml-images : beta version of feature computation for images in order to simplify computation on features computed
* dqm-ml-pytorch : beta implementation of 3 of existing domain gap metrics relying on pytorch
* dqm-ml-job : beta implementation of of metric computation cli from files, with grouping strategy (see roadmap)

This version allow use to generate metrics on welding uses case to demonstrate values of such an structural evolution
