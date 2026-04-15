# Changelog

## 2.0.0-rc (2026-04-10) - Release Candidate for V2

This release is the V2.0.0 release candidate, featuring major refactoring, security fixes, and documentation improvements.

### Issue ticket number and link

* fix: #61 - Python 3.10/3.11 compatibility
* fix: #60 - Fix parquet dataloader with image_path column
* fix: #59 - Data selection tests for CLI
* fix: #41 - Security warnings addressed
* fix: #31 - Test package dependencies upgraded
* fix: #29 - dqm-ml-pipeline renamed to dqm-ml-job
* fix: #28 - Tests moved to workspace root
* fix: #27 - Multi data selection configuration
* fix: #26 - Domain gap computation for multiple selections
* fix: #25 - Multiple data selection in same job
* fix: #24 - Progress messages during post-processing
* fix: #22 - Output folder auto-creation
* fix: #11 - MkDocs documentation generation

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

* fix: #42
* fix: #31
* fix: #31

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
