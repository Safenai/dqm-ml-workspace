# Available Metrics

DQM-ML V2 provides a set of core metrics to evaluate the quality of your Machine Learning datasets.

## Completeness

Evaluates the ratio of non-null to total values in your dataset.

- **Class**: `CompletenessProcessor`
- **Package**: `dqm-ml-core`
- **Output**:
  - `completeness_overall`: Average completeness across all analyzed columns.
  - `completeness_<column_name>`: Completeness score for a specific column.

**Configuration:**
```yaml
metrics_processor:
  my_completeness:
    type: completeness
    input_columns: ["col1", "col2"]
    include_per_column: true
    include_overall: true
    include_metadata: false
```

## Representativeness

Compares the distribution of your data to a target reference distribution (Normal or Uniform).

- **Class**: `RepresentativenessProcessor`
- **Package**: `dqm-ml-core`
- **Methods**: `chi-square`, `grte`, `shannon-entropy`, `kolmogorov-smirnov`
- **Target Distributions**: `normal`, `uniform`

**Configuration:**
```yaml
metrics_processor:
  my_representativeness:
    type: representativeness
    input_columns: ["numeric_col"]
    distribution: "normal"
    metrics: ["chi-square", "grte"]
    bins: 10
    distribution_params:
      mean: 0.0
      std: 1.0
```

## Visual Features

Computes basic image quality features for image datasets.

- **Class**: `VisualFeaturesProcessor`
- **Package**: `dqm-ml-images`
- **Features**:
  - `luminosity`: Mean gray level.
  - `contrast`: RMS contrast (standard deviation of gray levels).
  - `blur`: Variance of Laplacian (measures sharpness).
  - `entropy`: Shannon entropy of the gray histogram.

**Configuration:**
```yaml
metrics_processor:
  my_visual:
    type: visual_metrique
    input_columns: ["image_bytes"]
    dataset_root_path: "/path/to/images" # Optional if bytes are not provided
    grayscale: true
    normalize: true
```

## Domain Gap

Computes statistical distances between two datasets (source and target) based on their embeddings.

- **Class**: `DomainGapProcessor`
- **Package**: `dqm-ml-pytorch`
- **Metrics**:
  - `klmvn_diag`: KL-Divergence assuming Diagonal Multivariate Normal distribution.
  - `mmd_linear`: Maximum Mean Discrepancy (linear kernel).
  - `fid`: Frechet Inception Distance.
  - `wasserstein_1d`: 1D Wasserstein distance aggregated over dimensions.

**Configuration:**
```yaml
metrics_processor:
  my_domain_gap:
    type: domain_gap
    INPUT:
      embedding_col: "embedding"
    DELTA:
      metric: "fid"
```
