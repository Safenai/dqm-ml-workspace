# Full Pipeline

[📄 Config](../config/full_story.yaml) · [📖 Docs](../overview.md)

End-to-end pipeline combining all metric types on the synthetic images dataset. Runs visual feature extraction, deep embeddings, completeness, diversity, representativeness, and three domain-gap metrics (FID, MMD-RBF, Wasserstein-1D) in a single `dqm-ml process` invocation.

## Data

1200 synthetic images across 4 classes (elephant, lion, giraffe, zebra) and 3 sources (zoo, safari, reserve), with source-dependent null rates in `quality_score` (zoo 2%, safari 8%, reserve 15%) and distinct pixel-generation parameters per source.

## Data Loaders

Three loaders partition the image dataset by acquisition source, then split each source by animal class. This creates 6 selections — `safari_elephant`, `safari_zebra`, `reserve_elephant`, `reserve_zebra`, `zoo_elephant`, `zoo_zebra` — each flowing through every downstream processor.

```yaml
dataloaders:
  loaders:
    - name: safari
      type: parquet
      path: examples/data/samples_with_images.parquet
      batch_size: 500
      filters:
        - column: source
          values: [safari]
      split:
        by: class_name
        values: [elephant, zebra]

    - name: reserve
      type: parquet
      path: examples/data/samples_with_images.parquet
      batch_size: 500
      filters:
        - column: source
          values: [reserve]
      split:
        by: class_name
        values: [elephant, zebra]

    - name: zoo
      type: parquet
      path: examples/data/samples_with_images.parquet
      batch_size: 500
      filters:
        - column: source
          values: [zoo]
      split:
        by: class_name
        values: [elephant, zebra]
```

## Feature Extraction

Two feature processors run sequentially: first `image_features` computes per-image quality indicators, then `features_embeddings` extracts 512-d deep feature vectors.

```yaml
features:
  outputs:
    path: examples/outputs/full_story_features.parquet
    include:
      - sample_id
      - source
      - class_name

  processors:
    - name: visual_features
      type: image_features
      columns:
        input: [image_bytes]
      features: [luminosity, contrast, blur, entropy]
      grayscale: true
      normalize: true
      histogram:
        bins: 256
      laplacian_kernel: 3x3

    - name: embedding
      type: features_embeddings
      columns:
        input: [image_bytes]
      model:
        arch: resnet18
        n_layer_feature: -2
      infer:
        batch_size: 32
        width: 64
        height: 64
        norm_mean: [0.485, 0.456, 0.406]
        norm_std: [0.229, 0.224, 0.225]
```

### Features Output

Per-sample visual features and embeddings (605 rows — images that loaded successfully from the 6 selections):

| Feature | Safari (n=185) | Reserve (n=219) | Zoo (n=201) |
|---|---|---|---|
| Luminosity (mean ± std) | 0.518 ± 0.016 | 0.389 ± 0.022 | 0.501 ± 0.037 |
| Contrast (mean ± std) | 0.199 ± 0.005 | 0.180 ± 0.009 | 0.154 ± 0.010 |
| Blur (mean ± std) | 0.816 ± 0.050 | 0.666 ± 0.072 | 0.505 ± 0.064 |
| Entropy (mean ± std) | 5.160 ± 0.020 | 4.902 ± 0.020 | 4.197 ± 0.022 |

The three sources show distinct visual profiles matching the engineered generator parameters:
- **Zoo** samples have lower contrast and less blur (enclosure shots, controlled conditions).
- **Safari** samples have the highest contrast and blur (open savanna, vehicle-based photography).
- **Reserve** samples have the lowest luminosity and mid-range blur (wild conditions, long-distance tourist photos).

## Metrics Computation

Three metric processors consume the extracted feature columns along with pre-existing metadata. Completeness checks both generated visual features and the pre-existing `quality_score` (which carries source-dependent nulls); representativeness validates visual feature distributions against a normal reference; diversity measures class balance within each selection.

```yaml
metrics:
  outputs:
    path: examples/outputs/full_story_metrics.parquet
  processors:
    - name: completeness
      type: completeness
      columns:
        input: [quality_score, image_bytes_luminosity, image_bytes_contrast, image_bytes_blur, image_bytes_entropy]
      include_per_column: true
      include_overall: true

    - name: diversity
      type: diversity
      columns:
        input: [class_name]
      metrics: [simpson, gini, shannon, richness]

    - name: representativeness
      type: representativeness
      columns:
        input: [image_bytes_luminosity, image_bytes_contrast, image_bytes_blur, image_bytes_entropy]
      metrics: [chi-square, grte, shannon-entropy, kolmogorov-smirnov]
      distribution: normal
      histogram:
        bins: 20
```

### Metrics Output

| Selection | Count | Completeness<br>Overall | Completeness<br>`quality_score` | Diversity<br>Richness | Diversity<br>Shannon |
|---|---|---|---|---|---|
| `safari_elephant` | 86 | 0.995 | 0.965 | 1 | -0 |
| `safari_zebra` | 99 | 0.991 | 0.939 | 1 | -0 |
| `reserve_elephant` | 104 | 0.975 | 0.827 | 1 | -0 |
| `reserve_zebra` | 115 | 0.971 | 0.800 | 1 | -0 |
| `zoo_elephant` | 103 | 0.996 | 0.971 | 1 | -0 |
| `zoo_zebra` | 98 | 0.997 | 0.980 | 1 | -0 |

**Completeness** — The `quality_score` column shows source-dependent missingness that mirrors the engineered null rates: reserve (15% null → ~0.80–0.83 completeness), safari (8% null → ~0.94–0.97), zoo (2% null → ~0.97–0.98). Generated feature columns (luminosity, contrast, blur, entropy, embedding) are always present after pipeline generation, so their per-column completeness is 1.0 across all selections.

**Diversity** — Each selection is filtered to a single class (the split key), yielding trivial per-selection diversity (richness=1, Shannon entropy ≈ 0). Diversity becomes meaningful when comparing selections with mixed classes or examining cross-selection distributions.

**Representativeness** — Correctly analyzes all 4 visual feature columns (confirmed in `_metadata.columns_analyzed`). The `total_samples_rep` field (344–460 across selections) aggregates the per-column reference distribution samples. Chi-square results vary: some selections' features are flagged as "does_not_follow_distribution", while GRTE consistently reports "high_representativeness" (0.63–1.00). The KS test (random-sampling approximation) scores all features as "follows_distribution" — the larger KS p-values suggest the resampled reference loses sensitivity to distributional differences that chi-square detects.

## Gap Computation

Three domain-gap processors each use a different distance metric, comparing the embedding vectors across all 6 selections (15 pairwise combinations each).

```yaml
gap:
  outputs:
    path: examples/outputs/full_story_gap.parquet
  processors:
    - name: fid_gap
      type: domain_gap
      columns:
        input: [image_bytes_embedding]
      distance:
        metric: fid
        epsilon: 1e-6

    - name: mmd_rbf_gap
      type: domain_gap
      columns:
        input: [image_bytes_embedding]
      distance:
        metric: mmd_rbf
        kernel_params:
          gamma: 1.0

    - name: wasserstein_gap
      type: domain_gap
      columns:
        input: [image_bytes_embedding]
      distance:
        metric: wasserstein_1d
```

### Gap Output (pivoted, sorted by FID descending)

| Source | Target | FID | MMD-RBF | Wasserstein-1D |
|---|---|---|---|---|
| `safari_elephant` | `zoo_zebra` | 635.9 | 0.0230 | 0.379 |
| `safari_elephant` | `zoo_elephant` | 624.0 | 0.0218 | 0.353 |
| `safari_zebra` | `zoo_elephant` | 541.8 | 0.0203 | 0.311 |
| `safari_zebra` | `zoo_zebra` | 532.7 | 0.0215 | 0.328 |
| `reserve_elephant` | `zoo_elephant` | 408.1 | 0.0198 | 0.266 |
| `reserve_elephant` | `zoo_zebra` | 406.4 | 0.0210 | 0.297 |
| `reserve_zebra` | `zoo_elephant` | 387.9 | 0.0189 | 0.231 |
| `reserve_zebra` | `zoo_zebra` | 362.1 | 0.0201 | 0.252 |
| `safari_elephant` | `reserve_zebra` | 181.3 | 0.0204 | 0.192 |
| `safari_elephant` | `reserve_elephant` | 156.9 | 0.0213 | 0.178 |
| `safari_zebra` | `reserve_elephant` | 121.8 | 0.0198 | 0.163 |
| `safari_zebra` | `reserve_zebra` | 121.4 | 0.0189 | 0.156 |
| `safari_elephant` | `safari_zebra` | 45.6 | 0.0217 | 0.076 |
| `reserve_elephant` | `reserve_zebra` | 36.3 | 0.0183 | 0.072 |
| `zoo_elephant` | `zoo_zebra` | 35.9 | 0.0211 | 0.073 |

The three domain-gap metrics consistently rank pairs by distributional difference:

- **FID** provides the widest dynamic range (36–636) and clearest separation between categories. Same-source, different-species pairs score lowest (36–46); cross-source pairs of the same species score higher (122–624). The zoo↔safari pairs rank highest (FID 533–636), reflecting the largest distribution shift between controlled enclosures and open safari conditions.
- **MMD-RBF** (γ=1.0) shows minimal variation (0.018–0.023) across all pairs — the fixed kernel bandwidth is insensitive to these embedding distributions.
- **Wasserstein-1D** follows the FID ranking (0.073–0.379) with moderate separation.

**Actionable insight**: For this dataset, FID alone suffices to detect and rank domain gaps. MMD-RBF with the default γ=1.0 adds no discriminative power; tuning γ via median heuristic or using multiple kernel bandwidths would improve sensitivity.

## Run

```bash
dqm-ml process -p examples/config/scenario/full_story.yaml
```

## Output Files

| File | Rows | Description |
|---|---|---|
| `full_story_features.parquet` | 605 | Per-image visual features + embeddings |
| `full_story_metrics.parquet` | 6 | Per-selection completeness, diversity, representativeness |
| `full_story_gap.parquet` | 45 | Pairwise gap (3 processors × 15 pairs) |

```python
import pandas as pd

features = pd.read_parquet("examples/outputs/full_story_features.parquet")
metrics = pd.read_parquet("examples/outputs/full_story_metrics.parquet")
gap = pd.read_parquet("examples/outputs/full_story_gap.parquet")
```
