# Concepts

This page defines the concepts used throughout DQM-ML documentation.

> **Note:** These definitions are under review and will be updated to be consistent with definitions released by ETAIA. They come from the [Confiance.ai program](https://docs.google.com/presentation/d/12bquOoLMtKzVpsgUQeCoytPXDunbCJyzsXMjB3le_4E/edit?slide=id.p12#slide=id.p12) without modification, with one addition: **Embedding**.

## Formal Concepts

These are the theoretical concepts that form the foundation of the data quality methodology.

### Sample

A **Sample** is a collection of **Features** corresponding to one observation or experience that will be inferred by one AI Model. May contain data fields (image, signal, text, numerical values) and metadata fields.

### Feature

A **Feature** refers to information extracted from data associated with each **Sample**. May be raw (acquired image, signal, token) or derived from other features.

### Sample Selection

A **Sample Selection** is a set of samples chosen for processing or analysis. May represent full datasets, filtered subsets, or programmatically defined selections.

See [Data Selection](configuration/data_selection.md) for how sample selections are configured.

### Metric

A **Metric** is a quantitative measure computed on a group of samples from a **Sample Selection**. May be global, feature-level, absolute (single selection), or relative (comparing two selections).

See [Metrics Guide](metrics.md) for available metrics.

### Batch

A **Batch** is a memory-fit subset of a **Sample Selection** used for efficient incremental processing.

### Batch Metric

A **Batch Metric** is an intermediate metric computed on a **Batch**, designed to be aggregated into a final **Metric** without loading all samples at once.

See [Metrics](configuration/metrics.md) for how batch metrics work.

### Domain Gap

A **Domain Gap** is a relative **Metric** comparing two **Sample Selections** (typically training vs. deployment), measuring distributional divergence or mismatch.

See [Domain Gap](metrics/domain_gap.md) for configuration details.

### Embedding

An **Embedding** is a representation of a high-dimensional **Sample** into a lower-dimensional **Feature** of numerical vectors.

See [Embedding Features](metrics/features_embeddings.md) for how embeddings are computed.

## Core Concepts

These are DQM-ML-specific concepts that implement the formal framework above.

### Data Selection

A **Data Selection** is a named subset of your data that can be used to compute **Metrics**, **Features**, or **Domain Gap**. It defines specific portions of your dataset for targeted quality assessments and comparisons.

See [Data Selection](configuration/data_selection.md) for configuration details.

### Processor

A **Processor** is a set of tools that process **Batches** or **Samples** to compute **Features**, **Metrics**, or **Domain Gaps**.

See [Configuration Guide](configuration/overview.md) for how processors are configured.
