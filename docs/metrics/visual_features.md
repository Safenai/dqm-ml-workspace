# Visual Features Metric

The Visual Features metric extracts standard image quality indicators from image datasets. It is used to analyze characteristics like brightness, contrast, and sharpness.

## Processor Information

* **Class**: `VisualFeaturesProcessor`
* **Package**: `dqm-ml-images`
* **Type Name**: `visual_metrique`

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
    type: visual_metrique
    input_columns: ["image_data"]
    grayscale: true
```

## Output

The processor generates the following feature columns in the output:

* `m_luminosity`
* `m_contrast`
* `m_blur_level`
* `m_entropy`
