# Known Limitations and Roadmap

This document outlines the current limitations of DQM-ML V2 and provides a high-level view of its future strategic direction.

## Current Status and Limitations

The V2 release focuses on a complete architectural overhaul, introducing a unified API, a modular plugin system, and streaming capabilities for large datasets. While functional, several limitations remain as we transition from the V1 codebase.

### Architectural Limitations

* **Beta Phase**: Both `dqm-ml-job` and `dqm-ml-images` are currently in beta. Configuration schemas and CLI parameters are subject to refinement based on community feedback.
* **CLI Renaming**: To better align with its role in MLOps, the pipeline component is slated to be renamed to `dqm-ml-job`.
* **Submodule Dependency**: The project still maintains a dependency on the legacy `dqm-ml` package for comparison and reference, which will be phased out once full parity is achieved.

### Functional and Scientific Limitations

* **Metric Parity**: A few metrics from V1, notably the **Diversity** metric and certain **Domain Gap** variants (PAD, CMD), are awaiting re-implementation under the new streaming API.
* **Result Consistency**: Minor variations in results for complex metrics like FID and KLMVN between V1 and V2 are under investigation to ensure mathematical equivalence.
* **Single-Column Focus**: Most current representativeness metrics operate on a per-column basis. Extending these to multi-dimensional feature sets requires further scientific contribution and validation.

## Roadmap

The DQM-ML roadmap is divided into three main phases: Finalization, Optimization, and Extension.

### Phase 1: Finalization of V2.0.0

* **Feature Parity**: Complete the porting of all remaining V1 metrics to the V2 API.
* **Stability**: Freeze the core API (`dqm-ml-core`) to provide a stable foundation for community-contributed metrics.
* **Standalone Release**: Transition out of the transitionary workspace into a finalized V2.0.0 release, marking the original repository as deprecated.

### Phase 2: Performance and Scalability

* **Advanced Streaming**: Enhance the streaming engine to handle even more complex data types and larger feature arrays by implementing disk-backed accumulators.
* **Parallelization**: Leverage multi-core processing for computationally intensive tasks like image feature extraction and deep-learning-based metrics.
* **Database Integration**: Enable the pipeline to read from and write directly to industrial databases, moving beyond flat-file processing.

### Phase 3: Domain Extensions

* **Time Series**: Introduce a dedicated `dqm-ml-timeseries` package for quality metrics on sequential data.
* **Multi-Modal Support**: Expand the framework to support multi-modal datasets (e.g., text-image pairs) and cross-domain quality metrics.
* **Enhanced Interoperability**: Provide deeper integration with MLOps platforms and experimental support for direct SQL-based metric computation via DuckDB.
