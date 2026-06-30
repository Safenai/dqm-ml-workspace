# Visual Features

[📄 Config](../config/scenario/visual_features.yaml) · [📖 Docs](../../docs/metrics/visual_features.md)

Extracts per-image quality indicators (luminosity, contrast, blur, entropy) from the synthetic 32×32 images. The output columns can be consumed downstream by Diversity or Representativeness processors.

## Data Loader

```yaml
# This loader reads the image dataset (1200 rows), not the tabular one.
# Image bytes columns are auto-detected; no mode declaration needed.
dataloaders:
  loaders:
    - name: source_dataset
      type: parquet
      path: examples/data/samples_with_images.parquet
      batch_size: 500
```

## Feature Extraction

```yaml
# Results are written to a separate parquet file, preserving select
# metadata columns alongside the computed visual features.
features:
  outputs:
    path: examples/outputs/story_visual_features.parquet
    include:
      - sample_id
      - class_id
      - class_name
      - source

  processors:
    - name: visual_features
      type: image_features
      columns:
        input: [image_bytes]
      features: [luminosity, contrast, blur, entropy]
      grayscale: true         # convert colour images to grayscale first
      normalize: true         # scale output values to [0, 1]
      histogram:
        bins: 256             # bins for the entropy computation
      laplacian_kernel: 3x3   # kernel size for the blur detector
```

## Run

```bash
dqm-ml process -p examples/config/scenario/visual_features.yaml
```

## Output

Metadata columns (from `include`) are written to `examples/outputs/story_visual_features.parquet` alongside the per-image feature values. Preview the first 5 rows:

```python
import pandas as pd

df = pd.read_parquet("examples/outputs/story_visual_features.parquet")
print(df.head(5).to_markdown(index=False))
```

The output includes both the metadata columns (from `include`) and the per-image visual feature columns computed by the processor.

|   sample_id |   class_id | class_name   | source   |   image_bytes_luminosity |   image_bytes_contrast |   image_bytes_blur |   image_bytes_entropy |
|------------:|-----------:|:-------------|:---------|-------------------------:|-----------------------:|-------------------:|----------------------:|
|           0 |          2 | giraffe      | reserve  |                 0.351482 |               0.169792 |           0.573141 |               4.9047  |
|           1 |          2 | giraffe      | safari   |                 0.503512 |               0.189889 |           0.763345 |               5.13467 |
|           2 |          1 | lion         | safari   |                 0.508238 |               0.202693 |           0.823988 |               5.16956 |
|           3 |          0 | elephant     | safari   |                 0.534855 |               0.196135 |           0.839806 |               5.11685 |
|           4 |          2 | giraffe      | reserve  |                 0.393763 |               0.192446 |           0.746597 |               4.93627 |

Aggregating by acquisition source reveals systematic differences:

| source   | luminosity (μ) | contrast (μ) | blur (μ) | entropy (μ) |
|:---------|---------------:|-------------:|---------:|------------:|
| zoo      | 0.501          | 0.154        | 0.505    | 4.197       |
| safari   | 0.518          | 0.199        | 0.816    | 5.160       |
| reserve  | 0.389          | 0.180        | 0.666    | 4.902       |

Zoo images have the lowest contrast (0.154) and sharpest detail (blur 0.505), consistent with close enclosure shots in controlled conditions. Safari images show the highest contrast (0.199) and entropy (5.160), reflecting open savanna with overhead sunlight and vehicle-based photography. Reserve images are notably darker (luminosity 0.389) relative to both zoo (0.501) and safari (0.518), with higher blur (0.666) from long-distance tourist photos in wild conditions. These per-source profiles show that acquisition conditions imprint measurable signatures on pixel-level statistics — downstream diversity or representativeness processors can exploit these to detect environment-specific quality issues.
