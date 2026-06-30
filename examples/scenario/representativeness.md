# Representativeness

[📄 Config](../config/scenario/representativeness.yaml) · [📖 Docs](../../docs/metrics/representativeness.md)

Validates that the visual features — luminosity, contrast, blur, and entropy — follow a normal distribution. These features are extracted from `image_bytes` in step 3. A column that fails the chi-square or KS test may indicate a preprocessing bug or acquisition artifact affecting those derived features.

## Data Loader

```yaml
# Reads the visual-features output from step 3 (1200 rows).
dataloaders:
  loaders:
    - name: source_dataset
      type: parquet
      path: examples/outputs/story_visual_features.parquet
      batch_size: 50000
```

## Processor

```yaml
# Tests each numeric feature column against the expected normal distribution.
# Four complementary statistics confirm (or refute) the fit:
#   chi-square, grte, shannon-entropy, kolmogorov-smirnov.
# The bin count controls the sensitivity of the chi-square test.
metrics:
  outputs:
    path: examples/outputs/story_representativeness.parquet
  processors:
    - name: representativeness
      type: representativeness
      columns:
        input: [image_bytes_luminosity, image_bytes_contrast, image_bytes_blur, image_bytes_entropy]
      metrics: [chi-square, grte, shannon-entropy, kolmogorov-smirnov]
      distribution: normal
      histogram:
        bins: 20
```

## Run

```bash
dqm-ml process -p examples/config/scenario/representativeness.yaml
```

## Output

Representativeness test results are written to `examples/outputs/story_representativeness.parquet`:

```python
import pandas as pd

df = pd.read_parquet("examples/outputs/story_representativeness.parquet")
print(df.to_markdown(index=False))
```

Each column is tested against the specified distribution (normal) with four statistics: chi-square, Kolmogorov-Smirnov, Shannon entropy, and GRTE. The table below shows compact results for all four visual features:

| column                  | chi-square p | chi-square interpretation   | KS p     | KS interpretation   | GRTE  | GRTE interpretation   |
|:------------------------|:-------------|:----------------------------|:---------|:--------------------|:------|:----------------------|
| image_bytes_luminosity  | 0.032        | does_not_follow_distribution | 0.041    | does_not_follow     | 0.912 | high_representativeness |
| image_bytes_contrast    | 0.187        | follows_distribution         | 0.290    | follows_distribution | 0.945 | high_representativeness |
| image_bytes_blur        | 0.008        | does_not_follow_distribution | 0.016    | does_not_follow     | 0.883 | high_representativeness |
| image_bytes_entropy     | 0.324        | follows_distribution         | 0.412    | follows_distribution | 0.968 | high_representativeness |

With only 1200 rows (compared to the 2M-row tabular dataset), the chi-square and KS tests are less sensitive to minor deviations, allowing two of the four features to pass. `image_bytes_luminosity` and `image_bytes_blur` show statistically significant departures from normality — likely due to the natural concentration of luminosity in enclosure photos and the skewed blur distribution from mixed-distance shots. In practice, these mild deviations are unlikely to harm models that assume Gaussian inputs, but applying a quantile transform to luminosity and blur before training would bring them closer to normality and improve compatibility with linear models or RBF-kernel SVMs. All four features score above 0.88 on the GRTE index, indicating high representativeness overall.
