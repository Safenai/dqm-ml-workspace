### 2.4 Features Processors

Features processors take a **Sample Selection** and compute a **Feature** on each sample. The output is the original data enriched with new columns.

#### Visual Features Processor

Computes image quality features (luminosity, contrast, blur, entropy) on image columns.

```yaml
features:
  processors:
    - name: image_quality
      type: image_features
      columns:
        input: ["image_bytes"]
      features: [luminosity, contrast, blur, entropy]
      batch_size: 64
      grayscale: true
      normalize: true
      laplacian_kernel: 3x3
      clip_percentiles: [2, 98]
      luminosity_weights: bt709    # bt601, bt2020, or custom list
      histogram:
        bins: 256
```

> See [Visual Features](../metrics/visual_features.md) for detailed parameter documentation.

> **💡 Runnable example:** See [../../examples/scenario/visual_features.md](../../examples/scenario/visual_features.md) for this processor in a complete pipeline.

#### Embedding Features Processor

Computes vector embeddings from images using a pretrained model (e.g., ResNet). Embeddings feed into Domain Gap computation.

```yaml
features:
  processors:
    - name: image_embeddings
      type: features_embeddings
      columns:
        input: ["image_bytes"]
      model:
        arch: resnet18
        n_layer_feature: -2
      infer:
        batch_size: 32
        width: 224
        height: 224
        norm_mean: [0.485, 0.456, 0.406]
        norm_std: [0.229, 0.224, 0.225]
```

> See [Features Embeddings](../metrics/features_embeddings.md) for architecture selection and normalization guidance.

> **💡 Runnable example:** See [../../examples/scenario/embeddings.md](../../examples/scenario/embeddings.md) for this processor in a complete pipeline.
