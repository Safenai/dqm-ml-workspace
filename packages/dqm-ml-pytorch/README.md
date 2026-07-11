# DQM-ML PyTorch

PyTorch-based metrics for DQM-ML V2. Provides advanced domain gap analysis for comparing dataset distributions.

## Installation

```bash
pip install dqm-ml-pytorch
```

> **Note:** `dqm-ml-pytorch` provides **Gap Processors** (Domain Gap) and **Features Processors** (Image Embeddings) — no CLI or job orchestration. Use directly via Python or with `dqm-ml-job` for YAML config execution.

## ImageEmbedding Processor

### Using Python Directly

Extracts image embeddings using a pre-trained model (default: ResNet-50 from torchvision). Outputs fixed-size float32 arrays suitable for domain gap analysis or downstream ML tasks.

```python
import io
import numpy as np
import pandas as pd
from PIL import Image
from dqm_ml_core import ProcessorRunner
from dqm_ml_pytorch import ImageEmbeddingProcessor

# Generate synthetic images inline
images = []
for i in range(4):
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    images.append(buf.getvalue())

df = pd.DataFrame({"image_bytes": images})

# Configure processor
processor = ImageEmbeddingProcessor(
    name="img_embed",
    config={
        "columns": {"input": ["image_bytes"]},
        "model": {
            "arch": "resnet18",
            "n_layer_feature": -2,  # layer index (second to last)
            "device": "cpu"
        },
        "infer": {
            "batch_size": 2,
        }
    }
)

# Run using ProcessorRunner (high-level API)
runner = ProcessorRunner()
result = runner.run(df, [processor])

emb_col = "image_bytes_embedding"  # output column name
print(f"Embeddings shape: {len(result[emb_col])} x {len(result[emb_col][0])}")  # 4 x 512
```

### With dqm-ml-job

For running from a YAML config, install together with `dqm-ml-job`:

```bash
pip install dqm-ml-job dqm-ml-pytorch
```

#### Generate test data

Create `test_images.parquet` with 4 classes × 4 samples (32×32 synthetic PNG bytes) — **minimalist example with synthetic 32×32 images**:

```python
# generate_data.py
import io, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image

np.random.seed(42)
images = []
classes = ["cat", "dog", "bird", "car"] * 4
for c in classes:
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    images.append(buf.getvalue())

table = pa.table({
    "sample_id": np.arange(16, dtype=np.int64),
    "class_name": classes,
    "image_bytes": images,
})
pq.write_table(table, "test_images.parquet")
print("Created test_images.parquet (16 rows, 4 classes)")
```

```bash
python generate_data.py
```

#### Run ImageEmbeddingProcessor

```yaml
dataloaders:
  loaders:
    - name: images
      type: parquet
      path: test_images.parquet   # ← generated file
      batch_size: 32

features:
  outputs:
    path: outputs/embeddings.parquet
    include:
      - sample_id
      - class_name
  processors:
    - name: img_embeddings
      type: features_embeddings
      columns:
        input: ["image_bytes"]
      model:
        arch: "resnet18"
        n_layer_feature: -2
      infer:
        batch_size: 32
        width: 32          # ← matches 32×32 generated images
        height: 32
```

Output column: `image_bytes_embedding` (pa.FixedSizeListArray<float32, 512>)

## Gap Processors

### Using Python Directly

#### From raw images (embeddings + gap in one call)

```python
import io
import numpy as np
import pandas as pd
from PIL import Image
from dqm_ml_core import ProcessorRunner
from dqm_ml_pytorch import ImageEmbeddingProcessor, DomainGapProcessor

# Generate synthetic images
np.random.seed(42)
source_images = []
for _ in range(8):
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    source_images.append(buf.getvalue())

target_images = []
for _ in range(8):
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    target_images.append(buf.getvalue())

source_df = pd.DataFrame({"image_bytes": source_images})
target_df = pd.DataFrame({"image_bytes": target_images})

# Compute embeddings + gap in one call
runner = ProcessorRunner()
result = runner.run_gap(
    source_df, target_df,
    DomainGapProcessor(
        name="gap",
        config={
            "columns": {"input": ["image_bytes_embedding"]},
            "distance": {"metric": "mmd_linear"},
        },
    ),
    features=[
        ImageEmbeddingProcessor(
            name="embedding",
            config={
                "columns": {"input": ["image_bytes"]},
                "model": {"arch": "resnet18", "n_layer_feature": -2, "device": "cpu"},
                "infer": {"batch_size": 4},
            },
        )
    ],
)

print(f"Domain Gap (MMD): {result['mmd_linear'][0].as_py():.4f}")
```

#### From pre-computed embeddings

```python
import numpy as np
import pandas as pd
from dqm_ml_core import ProcessorRunner
from dqm_ml_pytorch import DomainGapProcessor

np.random.seed(42)
source_df = pd.DataFrame({"embedding": list(np.random.randn(100, 128).astype(np.float32))})
target_df = pd.DataFrame({"embedding": list(np.random.randn(100, 128).astype(np.float32))})

runner = ProcessorRunner()
result = runner.run_gap(
    source_df, target_df,
    DomainGapProcessor(
        name="gap",
        config={
            "columns": {"input": ["embedding"]},
            "distance": {"metric": "mmd_linear"},
        },
    ),
)

print(f"Domain Gap (MMD): {result['mmd_linear'][0].as_py():.4f}")
```

### With dqm-ml-job

Uses the same `test_images.parquet` generated above. The `split.by: class_name` creates one selection per class (cat/dog/bird/car), and `pairwise: true` computes FID between all 6 class pairs.

```bash
pip install dqm-ml-job dqm-ml-pytorch
```

```yaml
dataloaders:
  loaders:
    - name: animals
      type: parquet
      path: test_images.parquet   # ← same file
      batch_size: 32
      split:
        by: class_name  # auto-discover: cat, dog, bird, car

features:
  outputs:
    path: outputs/features.parquet
  processors:
    - name: embedding
      type: features_embeddings
      columns:
        input: ["image_bytes"]
      model:
        arch: "resnet18"
        n_layer_feature: -2
      infer:
        batch_size: 32
        width: 32
        height: 32

gap:
  outputs:
    path: outputs/gap.parquet
    pairwise: true  # compare all class pairs
  processors:
    - name: fid_gap
      type: domain_gap
      columns:
        input: ["image_bytes_embedding"]
      distance:
        metric: "fid"
```

Requires: Input parquet file must have `image_bytes` column with PNG/JPEG bytes and a `class_name` column for splitting.

## Gap Metrics

| Metric | Full Name | Best For |
|--------|-----------|----------|
| **FID** | Fréchet Inception Distance | Image embeddings |
| **MMD-Linear** | Maximum Mean Discrepancy (linear kernel) | General-purpose comparison |
| **MMD-RBF** | MMD with RBF kernel | Detecting non-linear distribution shifts |
| **MMD-Poly** | MMD with polynomial kernel | Structured / higher-order differences |
| **Wasserstein** | 1D Earth Mover's Distance | 1D distributions |
| **KLMVN** | KL-Divergence (Multivariate Normal) | Gaussian distributions |
| **PAD** | Proxy A-Distance | Classifier-based divergence |
| **CMD** | Central Moment Discrepancy | Multi-layer feature comparison |

## Adding a Custom Gap Metric

Gap metrics are added directly to the `DomainGapProcessor` class. Here are the steps:

### 1. Add a metric computation method

Add a `_compute_delta_<metric>()` method in `domain_gap.py`. The method receives source and target statistics (each `dict[str, pa.Array]`) and returns `dict[str, pa.Array]` with the metric value.

**Example — Cosine Distance** (mean cosine similarity between source and target embeddings):

```python
@staticmethod
def _cosine_distance(src_emb: np.ndarray, tgt_emb: np.ndarray) -> float:
    norm_src = src_emb / np.maximum(np.linalg.norm(src_emb, axis=1, keepdims=True), 1e-12)
    norm_tgt = tgt_emb / np.maximum(np.linalg.norm(tgt_emb, axis=1, keepdims=True), 1e-12)
    cos_sim = (norm_src @ norm_tgt.T).mean()
    return float(1.0 - cos_sim)

def _compute_delta_cosine(self, source: dict[str, pa.Array], target: dict[str, pa.Array]) -> dict[str, pa.Array]:
    if "__emb__" not in source or "__emb__" not in target:
        return {"metric": pa.array(["cosine_distance"]), "note": pa.array(["missing __emb__"])}
    src = _fixed_to_matrix(source["__emb__"])
    tgt = _fixed_to_matrix(target["__emb__"])
    val = self._cosine_distance(src, tgt)
    return {"cosine_distance": pa.array([val], type=pa.float64())}
```

### 2. Wire into the dispatch chain

Add an `elif` branch in `compute_delta()`:

```python
if metric == "cosine_distance":
    return self._compute_delta_cosine(source, target)
```

### 3. (Optional) Configure summary collection

If your metric needs data beyond what's already collected, add auto-detection in `_configure_summary()`:

```python
auto_store_emb = self.delta_metric in {"mmd_rbf", "mmd_poly", "pad", "cmd", "cosine_distance"}
```

### 4. Update the class docstring

Add the new metric to the class and method docstrings so users know it's available.

### Summary of summary requirements

| Requires | Metrics |
|----------|---------|
| **sum/sum_sq** (mean + variance) | `klmvn_diag`, `mmd_linear`, `fid` |
| **sum_outer** (covariance) | `fid` |
| **hist_counts** (1D histograms) | `wasserstein_1d` |
| **__emb__** (raw embeddings) | `mmd_rbf`, `mmd_poly`, `pad`, `cosine_distance` |

## Output

Returns statistical distance values:

- `fid`
- `mmd_linear`
- `mmd_rbf`
- `mmd_poly`
- `wasserstein_1d`
- `klmvn_diag`
- `pad`
- `cmd`

## Requirements

- `torch`
- `torchvision`
- `scipy`

## Dependencies

DQM-ML is modular. For domain gap and embedding features:

```bash
# Minimal: use as library only
pip install dqm-ml-pytorch

# For YAML config execution
pip install dqm-ml-job dqm-ml-pytorch

# Full stack with all metrics
pip install dqm-ml-job dqm-ml-core dqm-ml-images dqm-ml-pytorch
```

| Interface | Entry Point Group |
|-----------|-------------------|
| **Features (Embeddings)** | `dqm_ml.features` |
| **Gap** | `dqm_ml.gap` |

## See Also

- [Formal and Core Concepts](https://safenai.github.io/dqm-ml-workspace/docs/formal_concepts.md) for definitions of **Domain Gap**, **Embedding**, **Metric**, and related terminology.
- [Domain Gap Documentation](https://safenai.github.io/dqm-ml-workspace/docs/metrics/domain_gap/)
- [Configuration Guide](https://safenai.github.io/dqm-ml-workspace/docs/configuration/)