# Rational & Architecture of DQM-ML V2

DQM-ML V2 is a modular evolution of the original `dqm-ml` library, designed to address the challenges of large-scale data quality assessment.

## Why V2?

The transition to V2 was driven by several key requirements:

* Unified API: Providing a consistent interface for all metrics, whether they are simple statistical checks or complex deep-learning-based analyses.
* Scalability: Enabling the computation of metrics on datasets that far exceed available system memory.
* Extensibility: Making it easy for developers to add new metrics, data loaders, or output formats without modifying the core engine.
* Decoupled Dependencies: Allowing users to install only the necessary components (e.g., core metrics without PyTorch if not needed).
* Integrated Feature Extraction: Building feature computation (like visual features) directly into the pipeline to streamline quality analysis on unstructured data.

## Core Architecture

The system is built on a **Modular Plugin Architecture** using Python entry points. It decouples three main concerns:

1. **Data Loading (`DataLoader` & `DataSelection`)**: Responsible for discovering data and providing it in manageable batches.
2. **Metric Processing (`DatametricProcessor`)**: Implements the logic for computing statistics or extracting features from data batches.
3. **Output Writing (`OutputWriter`)**: Handles the persistence of computed features or metrics.

### Streaming Data Flow

Unlike V1, which often loaded entire datasets into memory (e.g., using large Pandas DataFrames), V2 utilizes a **Streaming Pipeline**:

1. **Batch Iteration**: The `DataLoader` creates a `DataSelection` which iterates over the dataset in batches (typically using PyArrow or optimized Pandas chunks).
2. **Incremental Aggregation**: Each `DatametricProcessor` implements `compute_batch_metric()`. This method updates intermediate statistics for each batch.
3. **Finalization**: Once all batches are processed, the `compute()` method is called to aggregate the intermediate results into final dataset-level metrics.
4. **Memory Efficiency**: This approach ensures that memory usage remains constant regardless of the total dataset size.

## Interoperability & Migration

V2 maintains a high level of interoperability with V1 concepts while optimizing the underlying implementation. 

* Comparative Performance: V2 implementations of Completeness and Representativeness show significantly lower memory footprints and faster execution times on large Parquet/CSV files due to the streaming architecture and PyArrow integration.
* Legacy Support: The legacy `dqm-ml` package is included as a submodule for reference and side-by-side comparison, though new development should strictly follow the V2 API.

## Implementation Details

The orchestration is handled by the `DatasetPipeline` class in `dqm-ml-pipeline`. It manages the lifecycle of the processors and ensures that only the required data columns are loaded from the source, further optimizing I/O performance.
