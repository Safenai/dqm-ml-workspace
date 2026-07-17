# Domain Gap Metric

The **Domain Gap** **Metric** measures the statistical distance between a source and a target distribution based on image **Embeddings**. It is used to quantify distribution differences between two datasets.

> **See also:** [Concepts](../formal_concepts.md) for definitions of **Domain Gap**, **Metric**, **Embedding**, and related terminology.

## What It Measures

Domain Gap measures how different two datasets are from each other. Use it to:

- **Compare train/test distributions** — Ensure test data resembles training data
- **Compare acquisition subsets** — Determine if different batches, sessions, or sources share the same distribution
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
- Compare different data versions
- Evaluate data augmentation strategies
- Compare acquisition sessions for consistency

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

### `columns`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | list[string] | `["embedding"]` | Column(s) containing pre-computed feature embeddings |

### `distance`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | string | `"klmvn_diag"` | Target metric name (see supported metrics above) |
| `k` | int | 5 | Number of moments (CMD only) |
| `feature_weights` | list[float] | `[1.0, ...]` | Per-layer weights, one per `columns.input` entry (CMD only) |
| `kernel_params` | dict | `{}` | Kernel parameters for MMD-RBF (`gamma`) and MMD-Poly (`degree`, `gamma`, `coefficient0`) |
| `evaluator` | string | `"mse"` | Error metric for PAD: `"mse"` or `"mae"` (PAD only) |
| `epsilon` | float | `1e-6` | Regularization epsilon for covariance-based metrics (FID). Increase if you encounter singular matrix warnings. |
| `klmvn_var_eps` | float | `0.0` | Variance regularization for klmvn_diag. When > 0, var is replaced by `var + klmvn_var_eps * mean(var)`. Set to e.g. `1e-6` when comparing datasets with near-constant embedding dimensions. |

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

## Algorithm Details

### MMD-RBF

```
K(x, y) = exp(-γ · ‖x - y‖)
MMD² = mean(K_xx) + mean(K_yy) - 2 · mean(K_xy)
```

Uses **non-squared** Euclidean distance (matching `torch.cdist(p=2.0)`) and the **biased** estimator (includes diagonal elements), exactly matching the legacy implementation.

Default `gamma`: `1.0`.

### Practical Effect of `gamma`

The RBF kernel bandwidth `gamma` determines how "wide" the similarity window is. To build intuition:

> **Small gamma** = squint your eyes — only coarse structure visible.
> **Large gamma** = magnifying glass — tiny differences get amplified.
> **Default (1.0)** = typical ResNet-18 embedding scale.

#### Effect on common data quality tasks

**1. Comparing classes to find regroupable clusters**

You want to know which classes are similar enough to merge into one model.

| `gamma` | Behavior | What it tells you |
|---------|----------|-------------------|
| 0.01–0.1 | Classes must be very different to register a gap | Over-merges — distinct classes look similar |
| 1.0 (default) | Good separation at typical embedding scales | Reliable class hierarchy |
| 10–100 | Even minor differences register | Splits classes that could be safely merged |

*Tip:* Start at 1.0, then lower `gamma` slightly (0.1) to see which class pairs stay close — those are your best merge candidates.

**2. Comparing acquisition subsets to detect environmental drift**

Your dataset was collected across different sessions, cameras, or lighting conditions. You want to know if those subsets drifted apart.

- Keep `gamma` at **1.0** — the absolute value doesn't matter; you only care about the **ranking** of subset pairs (subset A vs B vs C).
- A rising gap over acquisition order suggests hardware drift (e.g. sensor degradation).
- A high gap between day/night subsets confirms lighting is a confounder — consider balancing or re-collecting.

**3. Comparing synthetic vs real data for augmentation candidates**

You have a synthetic dataset (rendered, GAN-generated, etc.) and want to know if it's close enough to real data to be useful for training.

| `gamma` | Scenario | Result | Action |
|---------|----------|--------|--------|
| 1.0 | Synthetic too different from real | MMD > 0.3 | Synthetic data may hurt accuracy |
| 1.0 | Synthetic close to real | MMD < 0.1 | Safe to augment |
| Sweep [0.01, 0.1, 1, 10] | Synthetic similar on coarse scale only | Small gap at 0.01, large at 10 | Good for low-level features, bad for high-level semantics |

#### Choosing `gamma` in practice

1. Start with `gamma: 1.0`.
2. If MMD-RBF ≈ MMD-Linear (poor separation), `gamma` is too small. Try `gamma: 0.1` or scale embeddings.
3. If MMD-RBF is near-constant for all pairs, `gamma` is too large. Try `gamma: 10`.
4. *Heuristic:* sample 500 embeddings, compute median pairwise Euclidean distance → `gamma ≈ 1 / median`. For ResNet-18 avgpool, this is usually ~1.0.
5. Once chosen, **lock `gamma`** — the ranking of scores is what matters, not the absolute value.

#### Interpretation rules of thumb

| MMD-RBF range vs reference | What it suggests |
|----------------------------|------------------|
| **< 0.05** | Distributions are nearly identical — safe to merge / augment |
| **0.05 – 0.2** | Moderate gap — investigate further; may still be usable |
| **> 0.2** | Significant gap — separate models or data filtering recommended |

> **Important:** These thresholds are guidelines for typical ResNet-18 embeddings at `gamma=1.0`. The absolute values depend on your embedding model, embedding dimension, and `gamma`. Always calibrate thresholds on a known-good vs known-bad pair from your own pipeline.

### MMD-Poly

```
K(a, b) = (γ · ⟨a, b⟩ + c)^d
MMD² = mean(K_xx) + mean(K_yy) - 2 · mean(K_xy)
```

Same biased estimator as MMD-RBF. Defaults: `degree=3.0`, `gamma=1.0`, `coefficient0=1.0`.

### Practical Effect of `degree`, `gamma` and `coefficient0`

The polynomial kernel `(γ·⟨a,b⟩ + c)^d` has three knobs with distinct roles:

| Parameter | Effect | Low value | High value |
|-----------|--------|-----------|------------|
| `degree` | Polynomial order of feature interactions | ~ linear (d=1) — coarse grouping | Captures complex interactions but emphasises large dot products (d≥4) — noise-prone |
| `gamma` | Dot-product scaling (same name, different meaning from MMD-RBF) | Kernel response compressed — may under-estimate gaps | Kernel response amplified — may over-estimate gaps |
| `coefficient0` | Additive bias term | Only angular similarity matters (c=0) — sensitive to direction only | All similarities amplified, even tiny dot products (c>0) — less discriminative |

#### Effect on common data quality tasks

**1. Comparing classes for regrouping**

| `degree` | Behavior | Best for |
|----------|----------|----------|
| 2 | Quadratic interactions — captures pairwise feature correlations | Broad class grouping, rough hierarchy |
| 3 (default) | Cubic interactions — good balance | General class separation |
| 4+ | Very high-order interactions | Rarely useful — tends to over-emphasise large-magnitude outliers |

*Tip:* Run with degree=2 and degree=3. If the ranking is similar, the extra polynomial order doesn't matter for your data. If degree=3 separates pairs that degree=2 considers identical, those pairs differ in higher-order feature interactions.

**2. Comparing acquisition subsets**

- Keep `degree=3` (default) — changing it rarely helps for drift detection.
- If you suspect the drift is a simple brightness/contrast shift (linear transformation), try `degree=1`: it acts like MMD-Linear but with an intercept term.
- `coefficient0=0` removes the bias, making the kernel purely angle-based — useful when you want to ignore magnitude differences (e.g. comparing normalised vs unnormalised data).

**3. Comparing synthetic vs real data**

- If the gap is large at degree=3 but small at degree=2, the synthetic data matches real data on pairwise feature correlations but differs on triple-wise interactions. Check if those interactions matter for your task.
- Defaults (`degree=3`, `gamma=1.0`, `coefficient0=1.0`) are a reasonable starting point.

#### Choosing MMD-Poly parameters in practice

1. Start with defaults (`degree=3.0`, `gamma=1.0`, `coefficient0=1.0`).
2. If results are too noisy, try `degree=2` — fewer interaction terms, smoother.
3. If results lack discrimination, try `degree=4` — but verify on known-different pairs that the gap is real, not driven by outliers.
4. `gamma` and `coefficient0` are best left at defaults unless you have a specific reason to change them (e.g. `coefficient0=0` for angle-only comparison).

### PAD (Proxy A-Distance)

1. Concatenate source and target embeddings with labels (0 = source, 1 = target).
2. Train `sklearn.svm.SVC(C=1, kernel="linear", probability=True, random_state=42)`.
3. Compute error (MSE or MAE) between predicted probabilities and one-hot labels.
4. `PAD = 2 · (1 - 2 · error)`.

### Practical Effect of `evaluator`

PAD trains a linear SVM to distinguish source from target, then measures how well it fails. The `evaluator` controls how prediction error is computed.

| Evaluator | Sensitive to | Best for |
|-----------|-------------|----------|
| `"mse"` (default) | Squared errors → penalises confident mistakes heavily | Detecting any separable difference, even subtle |
| `"mae"` | Absolute errors → more robust to outliers | Noisy embedding spaces; when a few extreme points differ but the bulk is similar |

*Tip:* If PAD with `"mse"` gives a large gap but you suspect it's driven by a handful of outlier samples, re-run with `"mae"`. If the gap stays large, the drift is pervasive and real. If it shrinks significantly, the gap was outlier-driven and may not matter for training.

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

### Practical Effect of `k`

The number of moments `k` controls how much distributional shape information is captured.

> **Small k (= 2)** = compare only mean and spread — like describing a mountain by its position and width.
> **Large k (= 10)** = also capture finer shape — peak sharpness, tail heaviness, asymmetry.

Internally, each embedding column (e.g. a ResNet layer) is treated as per-channel spatial maps, sigmoid is applied (mapping values to 0–1), and power sums for orders 1..k are accumulated. Central moments are derived via binomial expansion, then averaged across channels via RMSE.

#### Effect on common data quality tasks

**1. Comparing classes for regrouping**

| `k` | What it captures | Impact on class hierarchy |
|-----|------------------|---------------------------|
| 2 | Mean + variance only | May over-merge classes with similar position/width but different shape |
| 5 (default) | Up to 5th moment (skewness, kurtosis) | Good balance — classes with different tail behavior stay separated |
| 10 | Very fine shape details | May split semantically similar classes differing only in rare extreme values |

*Tip:* Run at k=2 and k=5. If the ranking stays the same, higher moments don't add information for your data. If it changes, those extra moments capture meaningful differences.

**2. Comparing acquisition subsets for environmental drift**

| Drift type | Moments affected | Recommendation |
|---|---|---|
| Brightness shift (mean) | k=1–2 enough | Low `k` sufficient |
| Contrast change (variance) | k=2 captures it | `k=3` safe |
| Color cast / white balance (skewness) | k=3+ needed | Keep default `k=5` |
| Sensor noise (tail behavior) | k=4–5 | Stick with `k=5+` |

For general drift detection, `k=5` (default) is a safe catch-all.

**3. Comparing synthetic vs real data**

Synthetic data often matches real data on mean and variance (k=2) but differs on higher-order moments (k=5+). Run with both k=2 and k=5: if the gap is small at k=2 but large at k=5, your synthetic data captures the "main shape" but not fine texture — useful for augmenting coarse features, risky for fine-grained tasks.

#### Choosing `k` in practice

1. Start with `k=5` (default).
2. If comparisons seem too noisy (expected-similar pairs look different), try `k=3`.
3. If comparisons seem too coarse (expected-different pairs look similar), try `k=7`.
4. For dataset selection, lock `k` — the ranking matters, not the absolute value.

### Practical Effect of `feature_weights`

CMD can operate on multiple embedding columns simultaneously (e.g. five ResNet layers). Each layer is assigned a weight in the final weighted average.

| `feature_weights` pattern | Effect |
|---|---|
| `[1.0, 1.0, 1.0, 1.0, 1.0]` (default) | All layers equally important — balanced multi-scale view |
| `[0.0, 1.0, 1.0, 1.0, 1.0]` | Exclude early layer (maxpool: edges and textures) |
| `[0.5, 0.5, 0.5, 1.0, 1.0]` | Down-weight early layers, emphasize deep semantic features |
| `[0.0, 0.0, 0.0, 0.0, 1.0]` | Only deepest layer — pure semantics, ignoring texture |

*Tip:* Run once with all weights = 1.0, inspect per-layer losses, then re-weight to focus on the layers that matter for your task (e.g. deeper layers for semantic differences, earlier layers for texture/acquisition differences).

### Wasserstein-1D

The 1D Wasserstein distance (Earth Mover's Distance) measures how much probability mass must be moved to turn one distribution into another. For high-dimensional embeddings, it is computed per dimension and averaged — like slicing the embedding space into 1D projections and measuring each separately.

1. For each of the first `hist_dims` embedding dimensions, build a 1D histogram with `hist_bins` equal-width bins spanning `[hist_range[0], hist_range[1]]`.
2. Normalise counts to probability distributions.
3. Compute cumulative distribution functions (CDFs).
4. Per-dimension Wasserstein = `sum(|CDF_src - CDF_tgt|) · bin_width`.
5. Final score: mean across all non-empty dimensions.

Histograms are accumulated per batch and aggregated at the end (summary mode, no raw embeddings stored). Values outside `hist_range` are silently assigned to the outermost bins.

### Practical Effect of histogram parameters

> **`hist_bins`** = how fine-grained your measurement ruler is.
> **`hist_range`** = where you place the ruler — values outside it land in the edge bins without warning.
> **`hist_dims`** = how many of the 512 embedding dimensions you measure.

#### Effect on common data quality tasks

**1. Comparing classes for regrouping**

| `hist_bins` | Effect on class separation |
|---|---|
| 8–16 | Coarse — only big differences register; may over-merge fine-grained classes |
| 32 (default) | Good resolution for typical ResNet embeddings |
| 64–128 | Finer discrimination — needs enough samples per bin for reliable estimation |

*Heuristic:* With < 100 samples per class, keep `hist_bins ≤ 16`. With > 1000 samples per class, `hist_bins=64` is safe.

**2. Comparing acquisition subsets (critical: `hist_range`)**

This is the most important parameter — and the most common trap:

| `hist_range` scenario | Result |
|---|---|
| Too narrow (e.g. `[-1, 1]` for `[-5, 5]` data) | Values crammed into edge bins → **falsely small gap** — "clipping collapse" |
| Too wide (e.g. `[-10, 10]` for `[-1, 1]` data) | Bins are coarse → **underestimates the gap** — "resolution starved" |
| Default `[-3.0, 3.0]` appropriate | Typical for ResNet-18 avgpool embeddings |

*Critical:* If your embedding model produces different scales (e.g. ViT, CLIP), you **must** adjust `hist_range`. Run `min()` / `max()` on a sample of embeddings to verify. If values overflow, the metric silently degrades without warning.

**How to set `hist_range` correctly:**

1. Compute embeddings for a sample (100–1000 images).
2. Find global min and max across all dimensions.
3. Set `hist_range: [min - 0.5, max + 0.5]` (add a margin).
4. Or use percentiles: `[p1, p99]` to clip extreme outliers.

**3. Comparing synthetic vs real data**

Synthetic data often has narrower distributions than real data:

| Parameter | Risk | Mitigation |
|---|---|---|
| `hist_range` too narrow | Both distributions look clipped → gap under-estimated | Use wide enough range to cover both |
| `hist_bins` too low | Both look similar to bin resolution | Increase until the gap stabilises |

*Tip:* Double `hist_bins` and see if the Wasserstein distance changes significantly. If not, your bin count is adequate.

#### Choosing Wasserstein-1D parameters in practice

1. Start with defaults (`hist_dims=64`, `hist_bins=32`, `hist_range=[-3.0, 3.0]`).
2. Validate `hist_range` against actual embedding values. Adjust if needed.
3. Quick convergence check: run with 16, 32, and 64 bins. If the value stabilises, you're fine. If it keeps changing, increase further.
4. `hist_dims` at 64 is sufficient for most purposes — beyond that, extra dimensions add marginal information at linear compute cost.
5. Lock all parameters once chosen — the ranking matters, not the absolute value.

#### Interpretation rules of thumb

| Wasserstein-1D range | What it suggests |
|---|---|
| **< 0.05** | Nearly identical per-dimension distributions — safe to merge / augment |
| **0.05 – 0.3** | Moderate gap — investigate further; may still be usable |
| **> 0.3** | Significant gap — separate models or data filtering recommended |

> **Important:** These thresholds are guidelines for typical ResNet-18 embeddings at `hist_bins=32`, `hist_range=[-3.0, 3.0]`. Calibrate on known-good vs known-bad pairs from your own pipeline.

## YAML Configuration Examples

### FID (summary-based, outer products)

```yaml
gap:
  processors:
    - name: domain_gap
      type: domain_gap
      columns:
        input: ["embedding"]
      distance:
        metric: "fid"
        epsilon: 1e-6
```

### MMD-RBF (full-embedding with kernel params)

```yaml
gap:
  processors:
    - name: domain_gap
      type: domain_gap
      columns:
        input: ["embedding"]
      summary:
        store_embeddings: true
      distance:
        metric: "mmd_rbf"
        kernel_params:
          gamma: 1.0
```

### MMD-Poly (full-embedding with kernel params)

```yaml
gap:
  processors:
    - name: domain_gap
      type: domain_gap
      columns:
        input: ["embedding"]
      summary:
        store_embeddings: true
      distance:
        metric: "mmd_poly"
        kernel_params:
          degree: 3.0
          gamma: 1.0
          coefficient0: 1.0
```

### PAD (full-embedding with evaluator)

```yaml
gap:
  processors:
    - name: domain_gap
      type: domain_gap
      columns:
        input: ["embedding"]
      summary:
        store_embeddings: true
      distance:
        metric: "pad"
        evaluator: "mse"
```

### CMD (multi-layer streaming)

```yaml
gap:
  processors:
    - name: domain_gap
      type: domain_gap
      columns:
        input: ["emb_layer1", "emb_layer2", "emb_layer3"]
      distance:
        metric: "cmd"
        k: 5
        feature_weights: [1.0, 0.5, 0.5]
```

### Wasserstein-1D (summary-based, histograms)

```yaml
gap:
  processors:
    - name: domain_gap
      type: domain_gap
      columns:
        input: ["embedding"]
      summary:
        hist_dims: 64
        hist_bins: 32
      distance:
        metric: "wasserstein_1d"
```

### KLMVN-Diag (summary-based)

```yaml
gap:
  processors:
    - name: domain_gap
      type: domain_gap
      columns:
        input: ["embedding"]
      distance:
        metric: "klmvn_diag"
```

### Limitations

**klmvn_diag is unstable when either dataset has near-constant embedding dimensions.** The diagonal-covariance KL divergence divides by target variance and computes variance ratios. If any dimension is nearly constant (e.g. small synthetic images, highly compressed crops, or post-processed data), these terms dominate the metric with arbitrarily large values. The result is unreliable when `klmvn_diag` returns values > 1e6 — this indicates variance collapse.

**Remedy:** Set `klmvn_var_eps` to a small positive value (e.g. `1e-6`) to regularize near-zero variances towards the global mean variance. See the `klmvn_var_eps` parameter in the `distance` table.

### MMD-Linear (summary-based)

```yaml
gap:
  processors:
    - name: domain_gap
      type: domain_gap
      columns:
        input: ["embedding"]
      distance:
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
dataloaders:
  loaders:
    - name: source
      type: parquet
      path: data/source.parquet

features:
  processors:
    - name: image_embedding
      type: features_embeddings
      columns:
        input: ["image_path"]
      model:
        arch: resnet18
        n_layer_feature: -2
      infer:
        batch_size: 32
        width: 224
        height: 224
        norm_mean: [0.485, 0.456, 0.406]
        norm_std: [0.229, 0.224, 0.225]

    - name: image_embedding_cmd
      type: features_embeddings
      columns:
        input: ["image_path"]
      model:
        arch: resnet18
        n_layer_feature:
          - maxpool
          - layer1.1.relu_1
          - layer2.1.relu_1
          - layer3.1.relu_1
          - layer4.1.relu_1
      infer:
        batch_size: 32
        width: 224
        height: 224
        norm_mean: [0.485, 0.456, 0.406]
        norm_std: [0.229, 0.224, 0.225]

gap:
  processors:
    - name: domain_gap_fid
      type: domain_gap
      columns:
        input: ["embedding"]
      distance:
        metric: "fid"

    - name: domain_gap_mmd_linear
      type: domain_gap
      columns:
        input: ["embedding"]
      distance:
        metric: "mmd_linear"

    - name: domain_gap_mmd_rbf
      type: domain_gap
      columns:
        input: ["embedding"]
      summary:
        store_embeddings: true
      distance:
        metric: "mmd_rbf"
        kernel_params:
          gamma: 1.0

    - name: domain_gap_mmd_poly
      type: domain_gap
      columns:
        input: ["embedding"]
      summary:
        store_embeddings: true
      distance:
        metric: "mmd_poly"
        kernel_params:
          degree: 3.0
          gamma: 1.0
          coefficient0: 1.0

    - name: domain_gap_wasserstein_1d
      type: domain_gap
      columns:
        input: ["embedding"]
      distance:
        metric: "wasserstein_1d"

    - name: domain_gap_klmvn_diag
      type: domain_gap
      columns:
        input: ["embedding"]
      distance:
        metric: "klmvn_diag"

    - name: domain_gap_pad
      type: domain_gap
      columns:
        input: ["embedding"]
      summary:
        store_embeddings: true
      distance:
        metric: "pad"
        evaluator: "mse"

    - name: domain_gap_cmd
      type: domain_gap
      columns:
        input:
          - emb_maxpool
          - emb_layer1_1_relu_1
          - emb_layer2_1_relu_1
          - emb_layer3_1_relu_1
          - emb_layer4_1_relu_1
      summary:
        store_embeddings: true
      distance:
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
