# Domain Gap Metric

The Domain Gap metric measures the statistical distance between a source and a target distribution based on their embeddings. It is used to quantify drift or differences between datasets.

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
