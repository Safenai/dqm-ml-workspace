### 2.6 Gap Processors

Gap processors take **Embeddings** from two **Sample Selections** and compute a **Domain Gap** between them.

```yaml
gap:
  processors:
    - name: klmvn_diag
      type: domain_gap
      columns:
        input: ["embedding"]
      distance:
        metric: klmvn_diag

    - name: mmd_linear
      type: domain_gap
      distance:
        metric: mmd_linear

    - name: fid
      type: domain_gap
      distance:
        metric: fid
      summary:
        collect_sum_outer: true

    - name: wasserstein_1d
      type: domain_gap
      distance:
        metric: wasserstein_1d
      summary:
        histogram:
          dims: 64
          bins: 32
          range: [-3.0, 3.0]

    - name: mmd_rbf
      type: domain_gap
      distance:
        metric: mmd_rbf
        kernel_params:
          gamma: 1.0
      summary:
        store_embeddings: true

    - name: mmd_poly
      type: domain_gap
      distance:
        metric: mmd_poly
        kernel_params:
          degree: 3.0
          gamma: 1.0
          coefficient0: 1.0
      summary:
        store_embeddings: true

    - name: pad
      type: domain_gap
      distance:
        metric: pad
        evaluator: mse    # or mae
      summary:
        store_embeddings: true

    - name: cmd
      type: domain_gap
      distance:
        metric: cmd
        k: 5
        feature_weights: [1.0]
      summary:
        store_embeddings: true
```

| Distance Metric | Tunable Parameters | Summary Options |
|----------------|-------------------|-----------------|
| `mmd_rbf` | `gamma` | `store_embeddings` |
| `mmd_poly` | `degree`, `gamma`, `coefficient0` | `store_embeddings` |
| `cmd` | `k`, `feature_weights` | `store_embeddings` |
| `wasserstein_1d` | `histogram.dims`, `histogram.bins`, `histogram.range` | None |
| `pad` | `evaluator` (mse/mae) | `store_embeddings` |
| `fid` | None | `collect_sum_outer` |
| `mmd_linear` | None | None |
| `klmvn_diag` | None | None |

> See [Domain Gap](../metrics/domain_gap.md) for per-parameter practical effects and interpretation thresholds.
