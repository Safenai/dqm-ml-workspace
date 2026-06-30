# Embeddings

[📄 Config](../config/scenario/embeddings.yaml) · [📖 Docs](../../docs/metrics/features_embeddings.md)

Runs a pre-trained ResNet-18 on every image to produce 512-d embedding vectors. These are the essential input to all Domain Gap metrics. You can inspect the output parquet to see how samples from different sources cluster.

## Data Loader

```yaml
dataloaders:
  loaders:
    - name: source_dataset
      type: parquet
      path: examples/data/samples_with_images.parquet
      batch_size: 500
```

## Feature Extraction

```yaml
# The features_embeddings processor runs inference through a pre-trained
# ResNet-18. n_layer_feature: -2 extracts the penultimate layer,
# producing a 512-d feature vector per image.
features:
  outputs:
    path: examples/outputs/story_embeddings.parquet
    include:
      - sample_id
      - class_id
      - class_name
      - source

  processors:
    - name: embedding_extraction
      type: features_embeddings
      columns:
        input: [image_bytes]
      model:
        arch: resnet18
        n_layer_feature: -2
      infer:
        batch_size: 32        # images per forward pass
        width: 64            # images are resized to 64×64
        height: 64
        norm_mean: [0.485, 0.456, 0.406]   # ImageNet normalisation
        norm_std: [0.229, 0.224, 0.225]
```

## Run

```bash
dqm-ml process -p examples/config/scenario/embeddings.yaml
```

## Output

Metadata columns (from `include`) are written to `examples/outputs/story_embeddings.parquet`. The embedding vectors flow to gap processors in-memory; use `.head()` for a compact preview of the persisted metadata columns:

```python
import pandas as pd

df = pd.read_parquet("examples/outputs/story_embeddings.parquet")
print(df.head(5).to_markdown(index=False))
```

The output includes both the metadata columns and the 512-d embedding vectors (column `image_bytes_embedding`). Preview of the first 5 rows:

|   sample_id |   class_id | class_name   | source   | image_bytes_embedding   |
|------------:|-----------:|:-------------|:---------|:------------------------|
|           0 |          2 | giraffe      | reserve  | <embedding\>          |
|           1 |          2 | giraffe      | safari   | <embedding\>          |
|           2 |          1 | lion         | safari   | <embedding\>          |
|           3 |          0 | elephant     | safari   | <embedding\>          |
|           4 |          2 | giraffe      | reserve  | <embedding\>          |

Each row is a 512-d embedding vector for one image. These vectors serve as the input to all Domain Gap metrics — they capture the semantic content of each image in a compact, comparable form.
