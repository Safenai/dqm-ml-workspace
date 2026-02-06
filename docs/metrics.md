# Metrics Guide

DQM-ML V2 provides a set of modular metrics organized by package. Each metric is designed to handle large-scale datasets using a streaming architecture.

## Available Metrics

### Core Metrics (`dqm-ml-core`)

* **[Completeness](./metrics/completeness.md)**: Analyzes the presence of non-null values.
* **[Representativeness](./metrics/representativeness.md)**: Compares data distributions against a reference.

### Visual Metrics (`dqm-ml-images`)

* **[Visual Features](./metrics/visual_features.md)**: Extracts image quality indicators (luminosity, blur, etc.).

### Advanced Metrics (`dqm-ml-pytorch`)

* **[Domain Gap](./metrics/domain_gap.md)**: Measures statistical distance between source and target distributions.

## Configuration

Metrics are configured in the `metrics_processor` section of the pipeline YAML file. For detailed configuration options for each metric, follow the links above.
