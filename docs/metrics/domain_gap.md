# Domain Gap Metric

The Domain Gap metric measures the statistical distance between a source and a target distribution based on image embeddings. It is used to quantify drift or differences between datasets.

## What It Measures

Domain Gap measures how different two datasets are from each other. Use it to:

- **Compare train/test distributions** — Ensure test data resembles training data
- **Detect data drift over time** — Monitor distribution shifts in production
- **Validate data augmentation** — Check if augmented data maintains original characteristics

### Available Distance Metrics

| Metric | Full Name | Best For |
|--------|-----------|----------|
| **FID** | Fréchet Inception Distance | Comparing image distributions (assumes Gaussian) |
| **MMD-Linear** | Maximum Mean Discrepancy (linear kernel) | Fast linear comparison |
| **MMD-RBF** | Maximum Mean Discrepancy (RBF kernel) | Non-linear distribution comparison |
| **MMD-Poly** | Maximum Mean Discrepancy (polynomial kernel) | Non-linear with polynomial interactions |
| **Wasserstein-1D** | Earth Mover's Distance (1D projections) | Robust distribution comparison |
| **KLMVN-Diag** | KL Divergence (diag. Multivariate Normal) | Gaussian distributions with diagonal cov |
| **PAD** | Proxy A-Distance | Binary classification-based comparison |
| **CMD** | Central Moment Discrepancy | Multi-layer feature distribution comparison |

### Use Cases

- Validate train/test split quality
- Monitor data drift in production pipelines
- Compare different data versions
- Evaluate data augmentation strategies

## Processor Information

- **Class**: `DomainGapProcessor`
- **Package**: `dqm-ml-pytorch`
- **Type Name**: `domain_gap`

## Computation Modes

The processor uses three strategies depending on the metric:

### Summary-based (FID, MMD-Linear, Wasserstein-1D, KLMVN-Diag)

Accumulates compact statistics per batch — count, sum, sum-of-squares, optionally outer products (FID) or histograms (Wasserstein-1D). Memory-efficient, works with any dataset size.

### Full-embedding (MMD-RBF, MMD-Poly, PAD)

Stores raw embeddings (via `summary.store_embeddings`) and computes the metric from the full embedding matrices in a single step. Memory scales with the number of samples × embedding dimension.

### Streaming CMD (CMD)

Accumulates per-batch raw moments (`sum(X^j)`) for each embedding layer, then derives central moments at the end using moment identities. Single-pass, memory proportional to number of moments × feature dimensions.

## Supported Distance Metrics

* `fid`: Frechet Inception Distance — compares mean and covariance via `sqrtm`.
* `mmd_linear`: Maximum Mean Discrepancy with a linear kernel — `||μ_src - μ_tgt||²`.
* `mmd_rbf`: MMD with RBF kernel — `exp(-γ·‖x-y‖)`, biased estimator `mean(K_xx)+mean(K_yy)-2·mean(K_xy)`.
* `mmd_poly`: MMD with polynomial kernel — `(γ·⟨x,y⟩+c)^d`, biased estimator (same form).
* `wasserstein_1d`: 1D Wasserstein distance aggregated over embedding dimensions via histograms.
* `klmvn_diag`: KL-Divergence assuming diagonal Multivariate Normal distribution.
* `pad`: Proxy A-Distance — trains a linear SVM (`SVC(C=1, probability=True)`), returns `2·(1-2·error)` where error is MSE or MAE of predicted probabilities.
* `cmd`: Central Moment Discrepancy — per-layer RMSE of up to 5 central moments, weighted and averaged.

## Configuration Parameters

### `input`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embedding_col` | string | `"embedding"` | Column containing pre-computed feature embeddings |
| `embedding_cols` | list[string] | `[embedding_col]` | List of embedding columns for CMD (multi-layer) |

### `summary`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `collect_sum_outer` | bool | auto | Compute outer products (auto-enabled for FID) |
| `collect_hist_1d` | bool | auto | Compute 1D histograms (auto-enabled for Wasserstein-1D) |
| `store_embeddings` | bool | auto | Store raw embeddings (auto-enabled for MMD-RBF, MMD-Poly, PAD) |
| `hist_dims` | int | 64 | Number of embedding dimensions to histogram (Wasserstein) |
| `hist_bins` | int | 32 | Number of bins per dimension (Wasserstein) |
| `hist_range` | list[float] | `[-3.0, 3.0]` | Histogram range (Wasserstein) |

Auto-detection: `collect_sum_outer`, `collect_hist_1d`, and `store_embeddings` default to `true` automatically when the selected metric requires them. Explicitly setting them overrides the auto-detection.

### `delta`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | string | `"klmvn_diag"` | Target metric name (see supported metrics above) |
| `k` | int | 5 | Number of moments (CMD only) |
| `feature_weights` | list[float] | `[1.0, ...]` | Per-layer weights, one per `embedding_cols` entry (CMD only) |
| `kernel_params` | dict | `{}` | Kernel parameters for MMD-RBF (`gamma`) and MMD-Poly (`degree`, `gamma`, `coefficient0`) |

### `method`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `evaluator` | string | `"mse"` | Error metric for PAD: `"mse"` or `"mae"` |

## Algorithm Details

### MMD-RBF

```
K(x, y) = exp(-γ · ‖x - y‖)
MMD² = mean(K_xx) + mean(K_yy) - 2 · mean(K_xy)
```

Uses **non-squared** Euclidean distance (matching `torch.cdist(p=2.0)`) and the **biased** estimator (includes diagonal elements), exactly matching the legacy implementation.

Default `gamma`: `1.0`.

### MMD-Poly

```
K(a, b) = (γ · ⟨a, b⟩ + c)^d
MMD² = mean(K_xx) + mean(K_yy) - 2 · mean(K_xy)
```

Same biased estimator as MMD-RBF. Defaults: `degree=3.0`, `gamma=1.0`, `coefficient0=1.0`.

### PAD (Proxy A-Distance)

1. Concatenate source and target embeddings with labels (0 = source, 1 = target).
2. Train `sklearn.svm.SVC(C=1, kernel="linear", probability=True, random_state=42)`.
3. Compute error (MSE or MAE) between predicted probabilities and one-hot labels.
4. `PAD = 2 · (1 - 2 · error)`.

### CMD (Central Moment Discrepancy)

1. Per batch, per layer: accumulate raw moment sums `Σ(X^j)` for `j = 1..k`.
2. Aggregate across batches: `E[X^j] = Σ(X^j) / N`.
3. Derive central moments from raw moments using exact identities:

| Order | Central Moment Identity |
|---|---|
| 1 | `μ = E[X]` |
| 2 | `E[(X-μ)²] = E[X²] - μ²` |
| 3 | `E[(X-μ)³] = E[X³] - 3μE[X²] + 2μ³` |
| 4 | `E[(X-μ)⁴] = E[X⁴] - 4μE[X³] + 6μ²E[X²] - 3μ⁴` |
| 5 | `E[(X-μ)⁵] = E[X⁵] - 5μE[X⁴] + 10μ²E[X³] - 10μ³E[X²] + 4μ⁵` |

4. Per layer: `loss = sum(RMSE(src_cm_j, tgt_cm_j) for j in 0..k-1) / k`.
5. Final score: weighted average across layers.

This single-pass streaming approach is mathematically equivalent to the standard two-pass centered computation and is numerically stable for typical ResNet embedding magnitudes.

## YAML Configuration Examples

### FID (summary-based, outer products)

```yaml
metrics_processor:
  domain_gap:
    type: domain_gap
    input:
      embedding_col: "embedding"
    delta:
      metric: "fid"
```

### MMD-RBF (full-embedding with kernel params)

```yaml
metrics_processor:
  domain_gap:
    type: domain_gap
    input:
      embedding_col: "embedding"
    summary:
      store_embeddings: true
    delta:
      metric: "mmd_rbf"
      kernel_params:
        gamma: 1.0
```

### MMD-Poly (full-embedding with kernel params)

```yaml
metrics_processor:
  domain_gap:
    type: domain_gap
    input:
      embedding_col: "embedding"
    summary:
      store_embeddings: true
    delta:
      metric: "mmd_poly"
      kernel_params:
        degree: 3.0
        gamma: 1.0
        coefficient0: 1.0
```

### PAD (full-embedding with evaluator)

```yaml
metrics_processor:
  domain_gap:
    type: domain_gap
    input:
      embedding_col: "embedding"
    summary:
      store_embeddings: true
    delta:
      metric: "pad"
    method:
      evaluator: "mse"
```

### CMD (multi-layer streaming)

```yaml
metrics_processor:
  domain_gap:
    type: domain_gap
    input:
      embedding_cols: ["emb_layer1", "emb_layer2", "emb_layer3"]
    delta:
      metric: "cmd"
      k: 5
      feature_weights: [1.0, 0.5, 0.5]
```

### Wasserstein-1D (summary-based, histograms)

```yaml
metrics_processor:
  domain_gap:
    type: domain_gap
    input:
      embedding_col: "embedding"
    summary:
      hist_dims: 64
      hist_bins: 32
    delta:
      metric: "wasserstein_1d"
```

### KLMVN-Diag (summary-based)

```yaml
metrics_processor:
  domain_gap:
    type: domain_gap
    input:
      embedding_col: "embedding"
    delta:
      metric: "klmvn_diag"
```

### MMD-Linear (summary-based)

```yaml
metrics_processor:
  domain_gap:
    type: domain_gap
    input:
      embedding_col: "embedding"
    delta:
      metric: "mmd_linear"
```

### All Metrics in One Config

The following example runs all eight domain gap metrics in a single pipeline, sharing a common `image_embedding` processor for efficiency.

**CMD requires a separate image processor.** CMD needs multi-resolution features from five ResNet-18 layers (maxpool, layer1–layer4), while the other seven metrics use a single avgpool embedding. Adding those five layers to the shared processor would produce five large embedding columns for every metric, increasing memory and I/O cost. Instead, a dedicated `image_embedding_cmd` processor runs an independent inference pass that outputs the five multi-layer columns.

| Metric | Computation Mode | Image Processor |
|--------|-----------------|-----------------|
| FID | Summary-based (outer products) | `image_embedding` |
| MMD-Linear | Summary-based | `image_embedding` |
| MMD-RBF | Full-embedding | `image_embedding` |
| MMD-Poly | Full-embedding | `image_embedding` |
| Wasserstein-1D | Summary-based (histograms) | `image_embedding` |
| KLMVN-Diag | Summary-based | `image_embedding` |
| PAD | Full-embedding (SVM) | `image_embedding` |
| CMD | Streaming (per-layer moments) | `image_embedding_cmd` |

Values will differ from the per-metric examples above because all metrics here use ResNet-18 (e.g. FID normally uses Inception-v3). This config demonstrates pipeline structure, not exact equivalence.

```yaml
metrics_processor:
  image_embedding:
    type: image_embedding
    data:
      image_column: "image_path"
      mode: "path"
    model:
      arch: resnet18
      n_layer_feature: -2
      device: cpu
    infer:
      batch_size: 32
      width: 224
      height: 224
      norm_mean: [0.485, 0.456, 0.406]
      norm_std: [0.229, 0.224, 0.225]

  image_embedding_cmd:
    type: image_embedding
    data:
      image_column: "image_path"
      mode: "path"
    model:
      arch: resnet18
      n_layer_feature:
        - maxpool
        - layer1.1.relu_1
        - layer2.1.relu_1
        - layer3.1.relu_1
        - layer4.1.relu_1
      device: cpu
    infer:
      batch_size: 32
      width: 224
      height: 224
      norm_mean: [0.485, 0.456, 0.406]
      norm_std: [0.229, 0.224, 0.225]

  domain_gap_fid:
    type: domain_gap
    input:
      embedding_col: "embedding"
    delta:
      metric: "fid"

  domain_gap_mmd_linear:
    type: domain_gap
    input:
      embedding_col: "embedding"
    delta:
      metric: "mmd_linear"

  domain_gap_mmd_rbf:
    type: domain_gap
    input:
      embedding_col: "embedding"
    summary:
      store_embeddings: true
    delta:
      metric: "mmd_rbf"
      kernel_params:
        gamma: 1.0

  domain_gap_mmd_poly:
    type: domain_gap
    input:
      embedding_col: "embedding"
    summary:
      store_embeddings: true
    delta:
      metric: "mmd_poly"
      kernel_params:
        degree: 3.0
        gamma: 1.0
        coefficient0: 1.0

  domain_gap_wasserstein_1d:
    type: domain_gap
    input:
      embedding_col: "embedding"
    delta:
      metric: "wasserstein_1d"

  domain_gap_klmvn_diag:
    type: domain_gap
    input:
      embedding_col: "embedding"
    delta:
      metric: "klmvn_diag"

  domain_gap_pad:
    type: domain_gap
    input:
      embedding_col: "embedding"
    summary:
      store_embeddings: true
    delta:
      metric: "pad"
    method:
      evaluator: "mse"

  domain_gap_cmd:
    type: domain_gap
    input:
      embedding_cols:
        - emb_maxpool
        - emb_layer1_1_relu_1
        - emb_layer2_1_relu_1
        - emb_layer3_1_relu_1
        - emb_layer4_1_relu_1
    summary:
      store_embeddings: true
    delta:
      metric: "cmd"
      k: 5
      feature_weights: [1.0, 1.0, 1.0, 1.0, 1.0]
```

## Output

The processor returns the computed distance value:

| Column | Type | Description |
|--------|------|-------------|
| `fid` | float | Fréchet Inception Distance |
| `mmd_linear` | float | MMD with linear kernel |
| `mmd_rbf` | float | MMD with RBF kernel |
| `mmd_poly` | float | MMD with polynomial kernel |
| `wasserstein_1d` | float | Average 1D Wasserstein distance across dimensions |
| `klmvn_diag` | float | KL divergence (diagonal Multivariate Normal) |
| `pad` | float | Proxy A-Distance |
| `cmd` | float | Central Moment Discrepancy |
