# Representativeness Metric

The Representativeness **Metric** checks whether individual feature columns in your dataset match a specified reference distribution (Normal or Uniform). It is used to validate feature health, preprocessing correctness, and synthetic data quality — feature by feature.

Unlike **Domain Gap** (which compares two datasets globally), Representativeness works on a single **Sample Selection** and tells you whether each column "looks right" for its expected distribution.

> **See also:** [Concepts](../formal_concepts.md) for definitions of **Metric**, **Domain Gap**, **Sample Selection**, and related terminology.

## What It Measures

Representativeness answers: *"Does this feature column follow the distribution I expect?"*

### Available Statistical Tests

| Test | What It Computes | Interpretation |
|------|-----------------|----------------|
| **Chi-Square (χ²)** | Goodness-of-fit between observed binned frequencies and expected frequencies | p-value < alpha → feature **does not** follow the distribution |
| **Kolmogorov-Smirnov (KS)** | Maximum distance between empirical and theoretical CDFs | p-value < alpha → feature **does not** follow the distribution (uses random sampling) |
| **Shannon Entropy** | Information diversity of the binned distribution | Higher values → more spread across bins; lower → concentrated in few bins |
| **GRTE** | Granular Relative Theoretical Entropy — exponential gap between observed and theoretical entropy | Higher values → more representative; lower → poorly representative |

### Use Cases

**1. Sanity-check preprocessing**

After scaling, you expect each feature to match a known distribution:

| Transformation | Expected distribution | Representativeness config |
|---------------|----------------------|--------------------------|
| Standard scaling (z-score) | Normal(0,1) | `distribution: normal` |
| Min-max scaling | Uniform(0,1) | `distribution: uniform` |
| Quantile transformation | Uniform(0,1) | `distribution: uniform` |
| Box-Cox / Yeo-Johnson | Approximately normal | `distribution: normal` |

If a feature fails the test after preprocessing, it signals a problem — constant column, extreme outlier, or incorrect transformation.

**2. Validating synthetic tabular features**

When augmenting with synthetic data (GAN, VAE, oversampling), each generated feature should match the real distribution. Representativeness pinpoints *which specific features* deviate — unlike Domain Gap which gives a single global score.

- If all features pass → synthetic data is distributionally faithful.
- If a few features fail → those features need better modelling or post-processing.
- If many features fail → the synthetic data is unlikely to help training.

**3. Feature selection for model compatibility**

Some models perform better when features follow certain distributions:

| Model family | Prefers | Why |
|-------------|---------|-----|
| Linear / Logistic regression, LDA | Normal | Assumes normally distributed predictors |
| SVM with RBF kernel | Normal or near-normal | Sensitive to feature scale and skew |
| k-NN, distance-based | Uniform (after scaling) | Avoids distance domination by skewed features |
| Tree-based (RF, GBM) | No assumption | Robust to any distribution |

Run Representativeness across your feature set to decide which features to keep or transform for your chosen model.

**4. Checking feature engineering outputs**

After applying custom transformations (log, polynomial features, interactions, binning), verify the output matches expectation:

- Log-transform of a positive feature should produce roughly normal (if original was log-normal).
- Interaction features (product of two normals) follow a more complex distribution — but you can check against a uniform reference after rank-based scaling.

**5. Detecting acquisition artifacts across data sources**

When your dataset is assembled from multiple sources, the same feature may behave differently per source. Compare Representativeness results per-source:

- Source A: feature follows normal → healthy
- Source B: same feature fails the test → likely acquisition artifact (sensor difference, clipping, saturation)

This helps decide whether to re-collect, re-normalize, or drop a source for specific features.

## Processor Information

- **Class**: `RepresentativenessProcessor`
- **Package**: `dqm-ml-core`
- **Type Name**: `representativeness`

## Configuration Parameters

### `columns`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | list[string] | — | Column name patterns (fnmatch) to analyze |
| `exclude` | list[string] | `None` | Column patterns to exclude |
| `rename` | list[dict] | `None` | Rename output columns (`from` → `to`) |
| `prefix` | string | `None` | Prefix prepended to output column names |
| `suffix` | string | `None` | Suffix appended to output column names |

### Top-level

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distribution` | string | `"normal"` | Reference distribution: `"normal"` or `"uniform"` |
| `metrics` | list[string] | all four | Which tests to run: `chi-square`, `kolmogorov-smirnov`, `shannon-entropy`, `grte` |
| `alpha` | float | `0.05` | Significance threshold for chi-square and KS p-values |
| `epsilon` | float | `1e-9` | Numerical stability constant (rarely needs tuning) |

### `histogram`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bins` | int | `10` | Number of bins for histogram-based tests |

### `shannon`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | `2.0` | Entropy above this → "high diversity" |

**Note:** Shannon entropy is computed from **expected** counts (the theoretical target distribution), not observed data. With PPF-based bins (normal target) or user-provided params, expected counts are always equal across bins, yielding `H = log₂(bins)`. For the default 10 bins, `H ≈ 3.32 ≥ 2.0`, so the interpretation is always `"high_diversity"` regardless of input data. See [Shannon entropy measures expected diversity](#4-shannon-entropy-measures-expected-diversity).

### `grte`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | `0.5` | GRTE above this → "high representativeness" |
| `scaling_factor` | float | `-2.0` | Controls how sharply GRTE penalizes entropy gaps |

### `ks`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_size` | int | `500` | Max per-batch samples for KS test |
| `min_sample_size` | int | `50` | Min per-batch samples for KS test |
| `sample_divisor` | int | `20` | Batch size ÷ divisor = target samples per batch |

### `expected_counts_method` (optional, default: `"cdf"`)

Controls how expected bin counts are computed for chi-square, GRTE, and Shannon entropy.

| Method | Behaviour | When to use |
|--------|-----------|-------------|
| `cdf` | Exact expected counts via the CDF of the reference distribution. Deterministic — same result regardless of batch size or RNG seed. | **Default.** Recommended for all use cases. Enables batch-invariance testing. |
| `monte_carlo` | Draws `total_count` random samples from the reference distribution and histograms them. Stochastic — varies with batch size and RNG state. | Legacy behaviour. May be useful for comparing against previous results. |

### `distribution_params` (optional)

Per-column list of explicit distribution parameters used when `mean_std_estimation` is set to `"user_provided"`:

```yaml
distribution_params:
  - column: feature_x
    mean: 0.0
    std: 1.0
  - column: feature_y
    min: 0.0
    max: 1.0
```

Each entry supports fields:

| Field | Applicable distribution | Description |
|-------|------------------------|-------------|
| `column` | both | Column name this entry applies to (required). |
| `mean` | normal | Mean of the reference normal distribution. |
| `std` | normal | Standard deviation of the reference normal distribution. |
| `min` | uniform | Lower bound of the reference uniform distribution. |
| `max` | uniform | Upper bound of the reference uniform distribution. |

Omit to auto-estimate from data (default behaviour).

### `mean_std_estimation` (optional, default: `"from_first_batch"`)

Controls how distribution parameters (mean/std for normal, min/max for uniform) are estimated for generating expected counts in the chi-square test. Expected counts are then computed using the method specified by [`expected_counts_method`](#expected_counts_method-optional-default-cdf).

| Strategy | Behaviour | When to use |
|----------|-----------|-------------|
| `from_first_batch` | Estimate from the first batch's data and reuse for all batches of the same selection. Bin edges and expected distribution are always consistent. | **Default.** Safe for most workloads. Recommended when feature variance is stable across the dataset. |
| `per_batch` | Re-estimate from each batch's KS-sampled data. | High-variance data or streaming scenarios where the first batch is not representative of the whole. ⚠️ Risks `insufficient_bins` / NaN when per-batch feature variance is very low (bin edges don't match the re-estimated distribution). |
| `user_provided` | Use the explicit per-column parameters from `distribution_params`. | Known reference distribution (e.g. standard-scaled Normal(0,1)). Eliminates data-dependent estimation entirely. See [`distribution_params`](#distribution_params-optional) for format. |
| `from_all_data` | Estimate from the full dataset (requires two passes or materialized features). | ⏳ Not yet implemented — raises `NotImplementedError`. |

## Practical Effect of Parameters

### `distribution`: normal vs uniform

This is the foundational choice — it changes binning strategy, reference generation, and what "passing" means.

| distribution | Bin edges | When to use |
|-------------|-----------|-------------|
| `"normal"` | Quantile-based (equal-probability bins from normal PPF) | Standard-scaled features, residuals, biological measurements, any naturally Gaussian process |
| `"uniform"` | Linear-spaced across the data range | Min-max scaled features, rank-transformed features, categorical encodings, controlled parameters |

*Tip:* If you're unsure, run both. If a feature passes both tests, it is well-behaved. If it passes normal but not uniform (or vice versa), the pattern tells you something about the data generation process.

### `histogram.bins`

Controls the granularity of all histogram-based tests.

| bins | Effect |
|------|--------|
| 5–10 (default: 10) | Coarse — reliable with fewer samples, but may miss subtle deviations |
| 15–30 | Good balance for 500–5000 samples |
| 50+ | Fine-grained — requires many samples (>10k) to avoid sparse bins and unreliable chi-square p-values |

*Heuristic:* `bins ≈ sqrt(n_samples) / 2` is a common starting point. For 1000 samples, `bins ≈ 16`. For 10000 samples, `bins ≈ 50`.

**⚠️ Warning for large datasets.** At very large n (> 50k), the heuristic gives >100 bins, which produces many sparse bins. Cap bins at 50–100 for n > 100k. Conversely, too *few* bins at large n inflates chi-square sensitivity (see [`alpha`](#alpha) caveat).

**Effect on specific metrics:**

| Metric | Effect of more bins |
|--------|---------------------|
| Chi-square | More degrees of freedom → requires more samples per bin for reliable p-values |
| Shannon entropy | Maximum possible entropy increases (log₂(bins)) — a threshold of 2.0 at 10 bins means something different than at 50 bins |
| GRTE | More bins → finer entropy comparison → potentially larger gaps |

*Tip:* Double bins and re-run. If pass/fail decisions change, you needed more resolution. If they stay the same, bin count is adequate.

### `alpha`

The significance threshold for chi-square and KS p-values. Standard statistical convention, but with two practical twists for dataset building:

**1. Multiple-testing correction.** Testing many features increases false positives — with 100 features at alpha=0.05, expect ~5 false failures by chance.

**2. Sample-size correction.** Chi-square power grows with sample size. At large n, the test detects *tiny* deviations (often noise, not signal) as statistically significant:

| Sample size | Observed behavior with perfectly healthy N(0,1) | Recommended alpha |
|---|---|---|
| ≤ 1,000 | False positive rate ≈ alpha | 0.05 (default) |
| 1,000 – 10,000 | Slightly inflated | 0.01 |
| 10,000 – 100,000 | p ≈ 0.04 at alpha=0.05 (false positive) | 0.001 |
| > 100,000 | Nearly always p < 0.05 | Fewer bins, or use statistic directly |

At n=100k with 10 bins (~10k expected/bin), perfectly healthy N(0,1) data produces χ² ≈ 18 with p ≈ 0.036 — a false positive. This is not a bug; it is correct statistical behavior. The test is powerful enough to detect the tiny sampling noise that always exists. The issue is that at large n, *statistical significance* diverges from *practical significance*.

These two effects compound: with 100 features at 100k samples, expect far more false failures than the nominal alpha.

**Recommendation:** when using `user_provided` distribution params (known reference), scale alpha inversely with √n. Or use fewer bins (see [`histogram.bins`](#histogrambins)). Or track the chi-square statistic directly for trend monitoring — it is less sample-size-dependent than the p-value.

### GRTE: `threshold` and `scaling_factor`

GRTE is a custom metric unique to this framework. It converts the gap between observed entropy and expected entropy into a 0–1-ish score:

```
GRTE = exp(scaling_factor × |h_obs - h_exp|)
```

**`scaling_factor`** (default: `-2.0`):

| scaling_factor | Behavior |
|----------------|----------|
| `-1.0` | Gentle penalty — moderate entropy gaps still score ~0.37 |
| `-2.0` (default) | Moderate — entropy gap of 1.0 → GRTE ≈ 0.14 |
| `-4.0` | Aggressive — even small gaps produce near-zero scores |

A negative value means larger entropy gaps produce smaller GRTE scores (approaching 0). The more negative, the more aggressively small gaps are penalized.

*Tip:* If you find all features score very high or very low with little discrimination, adjust `scaling_factor`. Make it more negative (-4.0) to spread scores apart, or less negative (-1.0) to compress them.

**`threshold`** (default: `0.5`):

| threshold | Classifies as "high_representativeness" when |
|-----------|----------------------------------------------|
| 0.3 | Lenient — most features pass |
| 0.5 (default) | Balanced |
| 0.7 | Strict — only very close matches pass |

*Tip:* Calibrate on a known-good feature (one you trust) and a known-bad feature (e.g. a constant column). Set `threshold` between their GRTE scores.

### `shannon.threshold`

The Shannon entropy threshold depends on the number of bins. Maximum entropy for a distribution with `b` equal-probability bins is `log₂(b)`:

| bins | Max entropy | `threshold=2.0` means |
|------|-------------|----------------------|
| 10 | ~3.32 bits | Above mid-range — accepts reasonably spread distributions |
| 20 | ~4.32 bits | Below mid-range — stricter |
| 50 | ~5.64 bits | Quite strict — only well-spread distributions pass |

*Tip:* If you change `bins`, adjust `threshold` proportionally. A common heuristic: `threshold ≈ log₂(bins) × 0.6`.

### KS sampling parameters

The KS test uses random sampling per batch to stay efficient on large datasets. Three parameters control the tradeoff:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `sample_size` | 500 | Upper bound — larger = more accurate KS, slower |
| `min_sample_size` | 50 | Lower bound for small batches |
| `sample_divisor` | 20 | Per-batch target = max(50, batch_size ÷ 20), capped at 500 |

For a batch of 100,000 rows: `max(50, 100000÷20=5000)` capped at 500 → 500 samples per batch.

Only needs tuning if KS p-values are unstable across runs — increase `sample_size` to 2000–5000 for more stable estimates.

**⚠️ Stochasticity warning.** At the default 500 samples, the KS D-statistic has a standard error of ~1.36/√500 ≈ 0.06. Small distributional shifts (e.g., 5% contamination) produce D-changes < 0.02, which are swamped by sampling noise. The KS test may not reliably distinguish subtle DQ issues at default settings. Consider increasing `sample_size` to 5000–10000 when stable results are critical.

## Interpretation Rules of Thumb

### Chi-square and KS

| p-value vs alpha | Interpretation | Action |
|------------------|---------------|--------|
| p ≥ alpha | Follows the distribution | Feature is healthy for its expected distribution |
| p < alpha | Does not follow the distribution | Investigate — preprocessing issue, artifact, or genuinely non-conforming feature |

### Shannon Entropy

| Entropy vs threshold | Interpretation |
|---------------------|----------------|
| > threshold | High diversity — feature values are well spread across bins |
| ≤ threshold | Low diversity — feature values concentrate in few bins (potentially constant or skewed) |

### GRTE

| GRTE vs threshold | Interpretation |
|-------------------|---------------|
| > threshold | High representativeness — observed distribution closely matches expected |
| ≤ threshold | Low representativeness — observed distribution deviates meaningfully from expected |

> **Important:** Thresholds (alpha=0.05, shannon=2.0, grte=0.5, bins=10) are defaults that work for many cases but should be calibrated on your specific data. Always confirm with a known-good feature from your own pipeline.

## Caveats and Practical Limitations

### 1. Chi-square over-power at large sample sizes

As noted in the [`alpha` section](#alpha), chi-square power grows with sample size.
At large n, *statistical significance* diverges from *practical significance* — a feature
may fail the chi-square test even though it is perfectly healthy.

**Example from controlled testing** (100k samples, 10 bins, normal target):

| Dataset | χ² statistic | p-value | Interpretation at α=0.05 |
|---------|-------------|---------|--------------------------|
| `N(0, 1)` — perfectly healthy | 18.0 | 0.036 | ❌ does_not_follow_distribution |
| `N(0.2, 1)` — slight mean shift | 3,935 | ≈ 0 | ❌ does_not_follow_distribution |
| `N(0, 1.5)` — wrong variance | 23,940 | ≈ 0 | ❌ does_not_follow_distribution |

The first row is a **false positive** — the data *is* N(0,1), but the test flags it anyway.

**Recommendations:**

- Scale alpha inversely with √n (see alpha recommendations table).
- Or cap bins at a lower number (see [`histogram.bins`](#histogrambins)).
- Or track the chi-square *statistic* as a trend metric — it is less affected by sample
  size than the p-value. A sudden jump in the statistic between runs is more meaningful
  than the absolute p-value at any single run.
- When possible, include a known-good control feature to calibrate what "passing" looks
  like at your dataset's sample size.

### 2. KS test stochasticity at default settings

The KS test uses random subsampling (default: 500 samples per batch). At this sample size:

```
SE(D) ≈ 1.36 / √500 ≈ 0.06
```

From controlled testing, two mild contamination levels (5% vs 10% corrupted data) produce
D-statistics within ~0.002 of each other — well inside the noise floor. The KS test cannot
differentiate them at default settings.

**Recommendations:**

- Increase `ks.sample_size` to 5000–10000 for stable estimates when KS results inform
  critical decisions.
- Use chi-square for primary DQ screening — it is deterministic and uses all data.
- Compare KS D-statistics as trends over time rather than as absolute pass/fail signals.

### 3. Sensitivity mismatch: GRTE vs chi-square / KS

GRTE measures entropy gaps, not exact distribution fit. It is intentionally less sensitive
to small deviations:

| Contamination level | Chi-square | KS | GRTE (threshold 0.5) |
|---|---|---|---|
| Clean N(0,1) | ✅ pass | ✅ pass | ✅ 1.00 — high_representativeness |
| 5% corrupted | ❌ p ≈ 0 | ❌ p < 0.05 (if enough KS samples) | ✅ 0.98 — still high_representativeness |
| 20% corrupted | ❌ p ≈ 0 | ❌ p < 1e-9 | ✅ 0.79 — still high_representativeness |
| 50% corrupted | ❌ p ≈ 0 | ❌ p < 1e-80 | ❌ 0.32 — low_representativeness |

GRTE only flags distributions that are ~30%+ contaminated as `"low_representativeness"`
at the default scaling factor (-2.0). A feature can fail chi-square and KS while GRTE
still reports `"high_representativeness"` — this is **not a contradiction**, it reflects
that GRTE captures a different property (entropy similarity, not exact distribution match).

**Recommendations:**

- Do not expect GRTE and chi-square/KS to agree on marginal DQ issues.
- Calibrate the GRTE threshold on known-good and known-bad features (see
  [`threshold and scaling_factor`](#grte-threshold-and-scaling_factor)).
- Use GRTE for tracking *large* distributional shifts over time (drift monitoring),
  not for precise goodness-of-fit checks.
- Lower `scaling_factor` (e.g., -4.0) to make GRTE more sensitive, or raise it (e.g.,
  -1.0) to compress scores toward 1.0.

### 4. Shannon entropy measures expected diversity

The `shannon-entropy` metric computes entropy from **expected** counts (the theoretical
target distribution), not **observed** counts. With PPF-based binning (normal target)
or user-provided params, expected counts are always equal across bins:

```
H = log₂(n_bins)

Default (10 bins): H = log₂(10) ≈ 3.32 bits
Default threshold:  2.0 bits
→ Interpretation is always "high_diversity"
```

The metric provides no discriminatory information about the observed data in this
configuration. The interpretation is constant regardless of input data quality.

**Recommendations:**

- Disable `shannon-entropy` in your config if you don't have a specific use for it.
- Or use the `monte_carlo` expected_counts_method if you want entropy to reflect
  the stochasticity of sampling (note: this adds noise, not a signal about the data).
- Consider evaluating observed entropy as a separate custom metric if data diversity
  is a DQ concern.
- The default threshold (2.0) was chosen because `log₂(10) > 2.0`, ensuring the
  default interpretation is `"high_diversity"` — the metric acts as a sanity check
  that binning is working correctly, not as a data quality discriminator.

### 5. Chi-square blind spot: within-bin distribution changes

Chi-square only counts values per bin — it cannot detect changes within a bin.
This creates a blind spot for certain DQ issues:

| DQ issue | Chi-square (10 bins) | KS test |
|---|---|---|
| Clipping at ±1.5 (PPF bins from N(0,1)) | ❌ Blind — bin counts unchanged, clipped values stay in outermost bins | ✅ Detects via CDF jump at clip boundaries |
| Outliers at ±10 | ✅ Detects — mass shifts to outermost bins | ✅ Detects |
| Mode collapse (bimodal generator) | ✅ Detects — central bins lose mass | ✅ Detects |
| Sensor saturation at boundary | ❌ Blind if boundary is inside outermost bin edge | ✅ Detects |

**Recommendation:** always run KS alongside chi-square. Their blind spots are
complementary. The configuration example in the next section shows a recommended
setup with both enabled and tuned parameters.

## YAML Configuration Examples

### Minimal — check all numeric columns against normal distribution

```yaml
metrics:
  processors:
    - name: repr_check
      type: representativeness
      columns:
        input: ["*"]
```

### Full — all parameters customized

```yaml
metrics:
  processors:
    - name: repr_check
      type: representativeness
      columns:
        input: ["feat_*", "score_*"]
        exclude: ["feat_id"]
      distribution: normal
      metrics: ["chi-square", "kolmogorov-smirnov", "shannon-entropy", "grte"]
      alpha: 0.01
      histogram:
        bins: 20
      shannon:
        threshold: 3.0
      grte:
        threshold: 0.6
        scaling_factor: -2.5
      ks:
        sample_size: 2000
        min_sample_size: 100
        sample_divisor: 10
```

### Validating synthetic data

```yaml
metrics:
  processors:
    - name: synthetic_check
      type: representativeness
      columns:
        input: ["*"]
      distribution: normal
      metrics: ["chi-square", "grte"]
      histogram:
        bins: 20
      grte:
        threshold: 0.5
```

### Checking min-max scaling (uniform expected)

```yaml
metrics:
  processors:
    - name: scaling_check
      type: representativeness
      columns:
        input: ["*"]
      distribution: uniform
      metrics: ["chi-square", "kolmogorov-smirnov"]
      alpha: 0.01
```

## Output

For each input column `X` and each enabled metric, the processor produces:

| Output column | Type | Description |
|---------------|------|-------------|
| `X_chi-square_p_value` | float | Chi-square p-value |
| `X_chi-square_statistic` | float | Chi-square test statistic |
| `X_chi-square_interpretation` | string | "follows_distribution" or "does_not_follow_distribution" |
| `X_kolmogorov-smirnov_p_value` | float | KS p-value |
| `X_kolmogorov-smirnov_statistic` | float | KS test statistic |
| `X_kolmogorov-smirnov_interpretation` | string | "follows_distribution" or "does_not_follow_distribution" |
| `X_shannon-entropy_entropy` | float | Shannon entropy in bits |
| `X_shannon-entropy_interpretation` | string | "high_diversity" or "low_diversity" |
| `X_grte_grte_value` | float | GRTE score |
| `X_grte_interpretation` | string | "high_representativeness" or "low_representativeness" |
