# DQM-ML Core

This package defines the foundational API and core metrics for the DQM-ML V2 framework.

## Key Concepts

### `DatametricProcessor`

The base class for all metrics and feature extractors. It supports a streaming architecture by splitting computation into two phases:
1. Batch Level: `compute_batch_metric()` updates intermediate statistics for a single chunk of data.
2. Dataset Level: `compute()` aggregates these statistics into final scores.

## Included Metrics

* Completeness: Analyzes null/missing values.
* Representativeness: Statistical distribution analysis (Chi-Square, KS, etc.).

## For Developers

To create a new metric:
1. Subclass `dqm_ml_core.api.data_processor.DatametricProcessor`.
2. Define `needed_columns()`, `generated_features()`, and `generated_metrics()`.
3. Implement the streaming logic in `compute_batch_metric()` and `compute()`.
