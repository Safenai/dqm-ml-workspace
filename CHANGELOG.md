# Changelog

## 1.1.6 (2026-02-10) - upgrade dependency and security check

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
