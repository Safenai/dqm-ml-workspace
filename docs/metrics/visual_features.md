# Visual Features Processor

The Visual Features **Processor** extracts per-**Sample** image quality indicators (luminosity, contrast, blur, entropy) from image data. It is a **Feature Processor** — its outputs are columns consumed by downstream **Metrics** like Diversity and Representativeness.

> **See also:** [Concepts](../formal_concepts.md) for definitions of **Sample**, **Feature**, **Metric**, **Processor**, and related terminology.

## What It Produces

Four float-valued feature columns per image:

| Feature | What It Measures | Typical range | Interpretation |
|---------|-----------------|---------------|----------------|
| **Luminosity** | Mean intensity (brightness) | 0.0 (black) – 1.0 (white) | Under-exposed < 0.2, over-exposed > 0.9 |
| **Contrast** | RMS contrast (std of intensities) | 0.0 (flat) – ~0.5 (high) | Low contrast < 0.1 — image may lack detail |
| **Blur** | Variance of Laplacian (sharpness proxy) | 0 (perfectly flat) – high (sharp edges) | Blurry < 50, sharp > 500 (on 0–255 scale) |
| **Entropy** | Shannon entropy of the grayscale histogram | 0 (single tone) – ~8 (256 bins, uniform) | Low entropy < 4 — image has little information |

> **Note:** These ranges assume `normalize: true` and default settings. Changing `grayscale`, `clip_percentiles`, or `normalize` shifts the numerical ranges.

### Use Cases

**1. Filter low-quality images before training**

Set thresholds to remove problematic samples from your dataset:

| Condition | Visual Features rule |
|-----------|---------------------|
| Blurry images | `blur < threshold` — find the right threshold by plotting the blur histogram and looking for a natural cutoff |
| Under/over-exposed | `luminosity < 0.1` or `luminosity > 0.95` |
| Low information | `entropy < 3.0` — may indicate damaged or degenerate images |
| Low contrast | `contrast < 0.05` — may indicate flat-field images |

*Tip:* Don't hard-code thresholds. Compute visual features on your full dataset, plot histograms, and pick data-driven cutoffs based on percentiles (e.g. remove bottom 1% blurriest images).

**2. Check dataset balance across visual conditions**

A training set where 80% of images are dark (low luminosity) will bias a model toward low-light performance. Visual features let you audit:

- Is brightness balanced across classes? (e.g. "zebra" might be consistently bright, "bat" consistently dark)
- Is blur evenly distributed, or does one source consistently produce soft images?
- Does one acquisition session have systematically lower contrast?

These imbalances signal a need for re-balancing, additional collection, or per-source normalization.

**3. Validate augmentation and preprocessing**

After applying augmentations (brightness jitter, random crop, blur, compression), run Visual Features on the output:

- Does the brightness distribution still match the original? If not, augmentation may be too aggressive.
- Does blur increase uniformly (good — simulates out-of-focus) or only on a subset (bad — pipeline artifact)?
- Does entropy drop? A significant drop suggests information loss (over-compression, excessive downscaling).

**4. Feed downstream metrics (Diversity, Representativeness)**

Visual features are numeric columns that Diversity and Representativeness can analyze:

| Downstream metric | What it reveals |
|------------------|-----------------|
| **Diversity** on `luminosity` | Are brightness values spread across the full range or concentrated? |
| **Diversity** on `blur` | Is there variety in sharpness, or are all images similarly blurry/sharp? |
| **Representativeness** on `contrast` | Does contrast follow a normal distribution? Skewed or bimodal contrast may indicate multiple acquisition sources. |

> **⚠️ Important:** Changing Visual Features parameters (`grayscale`, `normalize`, `clip_percentiles`, `luminosity_weights`) changes the numerical values of all four output columns. This means **Diversity and Representativeness results will also change** — even with identical image data and metric config. Always document your Visual Features settings alongside downstream metric results for reproducibility.

## Processor Information

- **Class**: `VisualFeaturesProcessor`
- **Package**: `dqm-ml-images`
- **Type Name**: `image_features`

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns.input` | list[string] | `["image_bytes"]` | Column(s) with image data (bytes or file paths) |
| `features` | list[string] | all four | Which features to compute: `luminosity`, `contrast`, `blur`, `entropy` |
| `grayscale` | bool | `true` | Convert to grayscale before processing |
| `normalize` | bool | `true` | Normalise pixel values to [0, 1] |
| `laplacian_kernel` | string | `"3x3"` | Laplacian kernel size for blur: `"3x3"` or `"5x5"` |
| `clip_percentiles` | list[int] | `None` | e.g. `[1, 99]` — clip extreme pixel outliers before normalization |
| `histogram.bins` | int | `256` | Bins for entropy calculation |
| `luminosity_weights` | string or list | `"bt709"` | RGB → grayscale weights: `"bt601"`, `"bt709"`, `"bt2020"`, or `[R, G, B]` |
| `batch_size` | int | `64` | Images processed per batch (performance only) |

## Practical Effect of Parameters

### `grayscale`

| Setting | Effect |
|---------|--------|
| `true` (default) | Converts to luminance via weighted RGB average — single-channel, faster |
| `false` | Computes all four features on full color data, then reduces to single channel using `luminosity_weights` |

Use `grayscale: false` when you care about color-dependent quality (e.g. color bleeding, chromatic aberration). The processing is ~3x slower.

### `luminosity_weights`

Controls how RGB channels are weighted to produce the grayscale image. Different standards reflect different display technologies:

| Standard | Weights (R, G, B) | Emphasis |
|----------|-------------------|----------|
| `"bt601"` | (0.299, 0.587, 0.114) | Standard-def TV — older, green-heavy |
| `"bt709"` (default) | (0.2126, 0.7152, 0.0722) | HDTV — even more green-heavy |
| `"bt2020"` | (0.2627, 0.6780, 0.0593) | UHD / 4K — less blue sensitivity |
| Custom | e.g. `[0.33, 0.33, 0.34]` | Equal channel weight |

Only matters when `grayscale: false`. Use BT.709 for general purpose, custom weights for domain-specific needs (e.g. medical: weight green channel higher for tissue contrast).

### `laplacian_kernel`

| Kernel | Sensitivity | Best for |
|--------|-------------|----------|
| `"3x3"` (default) | Fine edges — responds to small-scale detail | General-purpose blur detection |
| `"5x5"` | Coarser edges — ignores fine texture, focuses on larger structures | High-resolution images where small-scale texture is expected |

*Tip:* If your dataset has naturally low texture (e.g. medical X-rays, satellite imagery), try `"5x5"` — the 3x3 kernel may falsely mark sharp low-texture images as blurry.

### `clip_percentiles`

When `None` (default): uses raw pixel values. When set, e.g. `[1, 99]`: clips the bottom 1% and top 1% of pixel values before computing features, then normalizes the remaining range to [0, 1].

| Scenario | Without clipping | With `[1, 99]` |
|----------|-----------------|----------------|
| Clean images | Correct | Slightly different (outliers trimmed) |
| Images with hot pixels / specular highlights | Luminosity inflated by outliers | Robust — outliers removed |
| Under-exposed with a few bright spots | Contrast under-estimated | More accurate contrast |

*Tip:* Start without clipping. If you see suspicious outliers in feature histograms (spikes at the extremes), add `clip_percentiles: [1, 99]` and compare.

### `features`

Select only the features you need to reduce compute:

```yaml
features: ["luminosity", "contrast"]
```

Omitting `blur` saves the Laplacian convolution (the most expensive single operation). Omitting `entropy` saves the histogram computation.

## YAML Configuration Examples

### Minimal — all defaults

```yaml
features:
  processors:
    - name: image_quality
      type: image_features
      columns:
        input: ["image_data"]
```

### Color with custom luminosity weights

```yaml
features:
  processors:
    - name: image_quality_color
      type: image_features
      columns:
        input: ["image_data"]
      grayscale: false
      luminosity_weights: [0.33, 0.33, 0.34]
```

### Outlier-robust with percentile clipping

```yaml
features:
  processors:
    - name: image_quality_robust
      type: image_features
      columns:
        input: ["image_data"]
      clip_percentiles: [1, 99]
      laplacian_kernel: "5x5"
```

### Lightweight — only luminosity and contrast, no entropy/blur

```yaml
features:
  processors:
    - name: fast_quality
      type: image_features
      columns:
        input: ["image_data"]
      features: ["luminosity", "contrast"]
```

## Output

For each image, the processor generates one column per enabled feature:

| Column | Type | Description |
|--------|------|-------------|
| `luminosity` | float | Mean intensity in [0, 1] (after normalization) |
| `contrast` | float | RMS contrast in [0, ~0.5] |
| `blur` | float | Variance of Laplacian (higher = sharper) |
| `entropy` | float | Shannon entropy in bits (depends on `histogram.bins`) |

These columns can be used directly as inputs to Diversity (`columns.input: ["luminosity", "blur"]`) or Representativeness (`columns.input: ["contrast"]`).
