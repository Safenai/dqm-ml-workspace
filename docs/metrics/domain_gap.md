# Domain Gap Metric

The Domain Gap metric measures the statistical distance between a source and a target distribution based on their embeddings. It is used to quantify drift or differences between datasets.

## What It Measures

Domain Gap measures how different two datasets are from each other. Use it to:

- **Compare train/test distributions** — Ensure test data resembles training data
- **Detect data drift over time** — Monitor distribution shifts in production
- **Validate data augmentation** — Check if augmented data maintains original characteristics

### Available Distance Metrics

| Metric | Full Name | Best For |
|--------|-----------|----------|
| **FID** | Fréchet Inception Distance | Image embeddings |
| **MMD** | Maximum Mean Discrepancy | General kernel-based |
| **Wasserstein** | Earth Mover's Distance | 1D distributions |
| **KLMVN** | Kullback-Leibler Multivariate Normal | Gaussian distributions |
| **H-Divergence** | Hypothesis-based Divergence | Binary classification |

### Use Cases

- Validate train/test split quality
- Monitor data drift in production pipelines
- Compare different data versions
- Evaluate data augmentation strategies

## Processor Information

* **Class**: `DomainGapProcessor`
* **Package**: `dqm-ml-pytorch`
* **Type Name**: `domain_gap`

## Supported Distance Metrics

* `fid`: Frechet Inception Distance.
* `klmvn_diag`: KL-Divergence assuming Diagonal Multivariate Normal distribution.
* `mmd_linear`: Maximum Mean Discrepancy with a linear kernel.
* `wasserstein_1d`: 1D Wasserstein distance aggregated over dimensions.

## Configuration Parameters

* `INPUT`:
  * `embedding_col`: Column containing pre-computed feature embeddings.
* `DELTA`:
  * `metric`: The specific distance metric to compute.

## Example YAML Configuration

```yaml
metrics_processor:
  domain_drift:
    type: domain_gap
    INPUT:
      embedding_col: "resnet_embeddings"
    DELTA:
      metric: "fid"
```

## Output

The processor returns the computed distance value:

* `domain_gap_<metric_name>`: The calculated statistical distance.
