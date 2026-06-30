# Embeddings Features Processor

The Embeddings Features **Processor** extracts high-dimensional vector representations (**Embeddings**) from images using a pre-trained deep neural network. Each image is mapped to a fixed-length float vector — similar images produce similar vectors.

It is a **Feature Processor**: its outputs are embedding columns consumed by the **Domain Gap** **Metric**.

> **See also:** [Concepts](../formal_concepts.md) for definitions of **Feature**, **Processor**, **Embedding**, **Domain Gap**, **Metric**, and related terminology.

## What It Produces

A fixed-length float-vector column — one embedding per image. For ResNet-18 (default), each embedding is 512 floats, representing what the model's final pooling layer "understands" about the image.

### Interpretation

| Aspect | What it means |
|--------|---------------|
| **Same class, similar photos** | Embeddings cluster together — small Euclidean distance |
| **Different classes** | Embeddings are far apart — large Euclidean distance |
| **Domain shift** | If source and target embeddings occupy disjoint regions of vector space, Domain Gap will be high |
| **Pre-trained bias** | The model was trained on ImageNet (1,000 everyday object categories). Embeddings of medical, satellite, or industrial images reflect what the model *can* extract — not necessarily what's *actually* discriminative in your domain |

### Use Cases

**1. Feed Domain Gap (primary use)**

Embeddings are the essential input to all Domain Gap sub-metrics (FID, MMD, CMD, Wasserstein-1D, PAD, etc.). The pipeline topology ensures embeddings are computed before Domain Gap runs.

**2. Compare distributions at different semantic levels (multi-layer mode)**

For the **CMD** metric: extract embeddings from multiple network layers simultaneously. Early layers capture edges and textures (low-level style), late layers capture object parts and scenes (high-level content).

Use this when you suspect domain shift operates at multiple levels — e.g. synthetic vs real images may differ in both texture (early layers) and object layout (late layers).

**3. Choose the right model for your data**

| Data type | Recommended architecture | Rationale |
|-----------|------------------------|-----------|
| General photographs, objects | `resnet18` (default) or `resnet50` | ImageNet pre-training matches well |
| Fine-grained categories (birds, cars) | `resnet50` or `vit_b_16` | Higher capacity captures subtle differences |
| Document scans, X-rays | `resnet18` | Low-level features often suffice; overfitting risk low |
| Aerial / satellite | `resnet18` (test first) | Pre-training domain is far from yours — bigger models may not help |

*Tip:* If your images are very different from everyday photographs (e.g. medical, microscopic, SAR), the pre-trained model's features may not be ideal. Run Domain Gap with multiple architectures and check whether results are consistent. If they disagree, neither may be reliable.

## Processor Information

- **Class**: `ImageEmbeddingProcessor`
- **Package**: `dqm-ml-pytorch`
- **Type Name**: `features_embeddings`

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns.input` | list[string] | `["image_bytes"]` | Column(s) with image data (bytes or file paths) |
| `model.arch` | string | `"resnet18"` | Torchvision model architecture |
| `model.n_layer_feature` | int or list[string] | `-2` | Target layer: `int`=single layer index, `list[str]`=multi-layer mode |
| `model.device` | string | `"auto"` | `"auto"`, `"cpu"`, or `"cuda"` |
| `infer.batch_size` | int | `32` | Images per inference batch |
| `infer.width` | int | `224` | Resize width before model input |
| `infer.height` | int | `224` | Resize height before model input |
| `infer.norm_mean` | list[float] | `[0.485, 0.456, 0.406]` | Per-channel mean (ImageNet stats) |
| `infer.norm_std` | list[float] | `[0.229, 0.224, 0.225]` | Per-channel std (ImageNet stats) |
| `columns.prefix` | string | `None` | Prepended to output column names |
| `columns.suffix` | string | `None` | Appended to output column names |

## Practical Effect of Parameters

### `model.arch`

Controls which pre-trained network extracts features. All models are loaded with ImageNet weights.

| Architecture | Embedding dim | Speed | When to use |
|-------------|---------------|-------|-------------|
| `resnet18` (default) | 512 | Fastest | General purpose — strong baseline |
| `resnet50` | 2048 | ~2x slower | Higher capacity, richer features |
| `vit_b_16` | 768 | ~3x slower (ViT) | Vision Transformer — global context, may capture domain shift CNNs miss |

*Tip:* Start with `resnet18`. If Domain Gap scores are unexpectedly low for datasets you know are different, try `resnet50` or `vit_b_16` — the default model may lack discriminative power for your domain.

**Important:** `features_embeddings` results are **not comparable across architectures**. A Domain Gap of 0.3 with `resnet18` vs 0.5 with `vit_b_16` does not mean the gap grew — the metric scales differ. Always report `model.arch` alongside any Domain Gap number.

### `model.n_layer_feature`

Controls which layer's activations become the embedding.

**Integer mode** (single layer, default `-2`):

| Value | Layer (ResNet-18) | Captures | Use for |
|-------|-------------------|----------|---------|
| `-2` (default) | `avgpool` | Global average of final feature map — balanced semantic summary | All standard Domain Gap metrics |
| `-1` | `fc` input | Just before classifier — most task-specific | When you want maximum semantic abstraction |
| `0` | First conv layer | Edges, corners, color blobs | Rarely useful — too low-level |

Negative indices count from the end (`-1` = last, `-2` = second-to-last). For ResNet-18, the default `-2` extracts the 512-d pooled feature map, which is the standard choice for image similarity.

**List mode** (multi-layer, for **CMD** metric):

Specify exact layer names found in the torchvision model graph:

```yaml
n_layer_feature: ["maxpool", "layer4.1.relu_1"]
```

Each named layer produces two output columns (embeddings + channel counts). The CMD metric then compares distributions across all layers jointly, capturing both low-level and high-level differences.

*When to use multi-layer:* When you suspect domain shift happens at multiple scales. For example, synthetic images may have unrealistic textures (visible in early layers) even if object shapes are correct (visible in late layers). Single-layer CMD or MMD might miss the texture shift.

### `model.device`

| Setting | Behavior |
|---------|----------|
| `"auto"` (default) | Use CUDA GPU if available, else CPU |
| `"cpu"` | Force CPU inference |
| `"cuda"` | Force GPU; fails if CUDA unavailable |

Performance-only parameter — results are identical regardless of device.

### `infer.width` / `infer.height`

Images are resized to `width x height` pixels before feeding the model. Default 224x224 matches ImageNet training resolution.

| Scenario | Effect of changing size |
|----------|------------------------|
| Higher-res input (448x448) | More detail preserved, but model sees unfamiliar scale — feature quality may degrade |
| Lower-res input (112x112) | Faster, but loses detail — may hurt discriminability |
| Non-square (e.g. 224x320) | Distorts aspect ratio — model may respond poorly |

*Recommendation:* Match the original training resolution of the architecture (224 for ResNet/ViT). Only change if your images have consistent native dimensions very different from 224 and you have validated the effect on your task.

### `infer.batch_size`

Controls GPU memory usage and throughput. Does not change results.

- Default `32` is safe for ResNet-18 on most GPUs
- Increase to `64`–`128` if you have ample GPU memory and want faster processing
- Decrease to `8`–`16` if running on CPU or low-memory GPU

### `infer.norm_mean` / `infer.norm_std`

These are the ImageNet channel-wise normalisation statistics. The model was trained to expect input pixels normalized with these exact values.

**Keep the defaults unless you have a specific reason to change them.** Changing these shifts the distribution of every embedding and will change Domain Gap results.

| Setting | When to consider |
|---------|-----------------|
| `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` (default) | Always start here |
| `[0, 0, 0]` / `[1, 1, 1]` | Only if you are using a model trained with raw [0,1] pixel values (uncommon with torchvision defaults) |

## ⚠️ Downstream Coupling

Embedding parameters cascade directly into **Domain Gap** results. Changing any of these will change every Domain Gap score:

- `model.arch` — different embedding space entirely
- `model.n_layer_feature` — different semantic level
- `infer.width` / `infer.height` — different spatial granularity
- `infer.norm_mean` / `infer.norm_std` — different input distribution

**Always document your full embedding configuration alongside any Domain Gap result.** A Domain Gap of 0.3 from dataset A vs B is meaningless without knowing `features_embeddings` was run with `resnet18`, `n_layer_feature: -2`, and 224x224 input.

## YAML Configuration Examples

### Minimal — all defaults

```yaml
features:
  processors:
    - name: image_embedding
      type: features_embeddings
      columns:
        input: [image_bytes]
```

Single output column: `image_bytes_embedding` (512-d for ResNet-18).

### Custom architecture and device

```yaml
features:
  processors:
    - name: image_embedding_resnet50
      type: features_embeddings
      columns:
        input: [image_bytes]
      model:
        arch: resnet50
        device: cuda
```

Output: `image_bytes_embedding` (2048-d).

### Multi-layer for CMD

```yaml
features:
  processors:
    - name: image_embedding_cmd
      type: features_embeddings
      columns:
        input: [image_bytes]
      model:
        n_layer_feature: ["maxpool", "layer4.1.relu_1"]
```

Output columns: `image_bytes_emb_maxpool`, `image_bytes_emb_maxpool_channels`, `image_bytes_emb_layer4_1_relu_1`, `image_bytes_emb_layer4_1_relu_1_channels`.

### Column prefix for multi-dataset pipelines

```yaml
features:
  processors:
    - name: crop_embeddings
      type: features_embeddings
      columns:
        input: [crop_bytes]
        prefix: crop_
```

Output column: `crop_image_bytes_embedding` (note: prefix applies before the input column name).

## Output

For each input column, the processor generates one or more embedding columns:

| Mode | Output column pattern | Type |
|------|-----------------------|------|
| Single-layer | `<prefix><input_col>_embedding<suffix>` | `FixedSizeList(float32, N)` where N = feature dimension |
| Multi-layer | `<prefix><input_col>_emb_<layer_name><suffix>` | `FixedSizeList(float32, layer_dim)` |
| Multi-layer (channels) | `<prefix><input_col>_emb_<layer_name>_channels<suffix>` | `FixedSizeList(float32, num_channels)` |

These columns are consumed by the Domain Gap metric via `columns.input` in the `domain_gap` processor config.
