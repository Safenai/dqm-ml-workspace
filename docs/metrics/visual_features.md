# Visual Features Metric

The Visual Features metric extracts standard image quality indicators from image datasets. It is used to analyze characteristics like brightness, contrast, and sharpness.

## What It Measures

Visual Features extracts image quality metrics to assess your image dataset. Use it to:

- **Check image quality** — Identify blurry or dark images
- **Validate dataset health** — Ensure images meet quality thresholds
- **Preprocessing validation** — Check augmentation results

### Available Features

| Feature | What It Measures | Use Case |
|---------|-----------------|----------|
| **Luminosity** | Brightness level | Detect under/over-exposed images |
| **Contrast** | RMS contrast | Find low-contrast images |
| **Blur** | Laplacian variance | Identify blurry images |
| **Entropy** | Shannon entropy | Measure information content |

### Use Cases

- Filter low-quality images from training data
- Validate image preprocessing pipeline
- Monitor image quality in production
- Check dataset balance across brightness levels

## Processor Information

* **Class**: `VisualFeaturesProcessor`
* **Package**: `dqm-ml-images`
* **Type Name**: `visual_metric`

## Computed Features

* **Luminosity**: Mean gray level of the image.
* **Contrast**: Root Mean Square (RMS) contrast.
* **Blur**: Variance of the Laplacian, used to estimate the level of focus/sharpness.
* **Entropy**: Shannon entropy of the image histogram.

## Configuration Parameters

* `input_columns`: Column name containing image bytes or local file paths.
* `grayscale`: Boolean, whether to convert images to grayscale before processing (default: true).

## Example YAML Configuration

```yaml
metrics_processor:
  image_quality:
    type: visual_metric
    input_columns: ["image_data"]
    grayscale: true
```

## Output

The processor generates the following feature columns in the output:

* `m_luminosity`
* `m_contrast`
* `m_blur_level`
* `m_entropy`
