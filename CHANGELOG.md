# Changelog

## 1.1.5 (2026-02-09)

* Quality correction / ci improvments **Fixed**
  * `noxfile.py`: Added a `docs` session to build the documentation. (Fixes #6)
  * `noxfile.py`: The `format` nox session now runs independently from the `lint` session. (Fixes #8)
  * `.github\ci.yml` : adjust security permissions and other code quality checks

* Repository organization changes :
  * Fixes : #28 : test moves to root of the workspace, and cover improved to 80%
  * fixes : #11 #12 : publication of mkdocs content document integrated

* Documentation :
  * First documentation strucutre proposed, to be upgraded with feedback, it integrate fixes for #9 and #10

* Conceptual change:
  * Notion of data selection introduced, and metrics are computed on a data selection, not a data loader.
  * delta metrics are compare betweenn all dataselection available in the data selection.

* Configuration changes
  * parquet data loader  (fix : #27)
  * comparison of multiple data selection metrics fix : #25 and fix : #26
  * progression bar (optional) has been added to follow computation in the cli (fix : #24)

* Limitations not delivered in v1.1.5
  * Change of dml-ml-pipeline to dqm-ml-job in futur release for consistency #29

## 1.1.4 (2025-12-18)

* github ci
* add changelog
* initiate release note and roadmap
* match of dqm-ml version number

## 0.0.5 (2025-12-17)

Version use to check compability with legacy dqm-ml, after this we will match for this repository the versions with dqm-ml
Version content :

* dqm-ml-core : API proposal for futur generic V2 api to allow generic usage of API
* dqm-ml-images : beta version of feature computation for images in order to simplify compuration on features computed
* dqm-ml-pytorch : beta implementation of 3 of existing domain gap metrics relying on pytorch
* dqm-ml-pipeline : beta implementation of of metric computation cli from files, with grouping strategy (see roadmap)

This version allow use to generate metrics on welding uses case to demonstrate values of such an structural evolution
