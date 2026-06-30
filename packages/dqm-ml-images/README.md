# DQM-ML Images

Image feature extraction package for DQM-ML V2. Provides metrics for assessing image dataset quality.

## Installation

```bash
pip install dqm-ml-images
```

> **Note:** `dqm-ml-images` provides metric processors only — no CLI or job orchestration. Use directly via Python or with `dqm-ml-job` for YAML config execution.

## Usage

### Using Python Directly

```python
import numpy as np
from pathlib import Path
from dqm_ml_images import VisualFeaturesProcessor
from PIL import Image

# Load or generate sample images
images = [Image.open("path/to/image1.jpg"), Image.open("path/to/image2.jpg")]

# Create and configure the processor
processor = VisualFeaturesProcessor(
    name="image_quality",
    config={
        "columns": {"input": ["image_bytes"]},
        "grayscale": True
    }
)

# Process images to extract features
batch = {"image_bytes": images}
features = processor.compute_features(batch)
print(f"Luminosity: {features['luminosity']}")
print(f"Contrast: {features['contrast']}")
print(f"Blur: {features['blur']}")
print(f"Entropy: {features['entropy']}")
```

### With dqm-ml-job

For running from a YAML config, install together with `dqm-ml-job`:

```bash
pip install dqm-ml-job dqm-ml-images
```

Then use this config:

```yaml
features:
  processors:
    - name: image_quality
      type: image_features
      columns:
        input: ["image_data"]
      grayscale: true
```

## Features

| Feature | Description |
|---------|-------------|
| **Luminosity** | Mean gray level — measures overall brightness |
| **Contrast** | RMS contrast — measures tonal range |
| **Blur** | Variance of Laplacian — estimates sharpness/focus |
| **Entropy** | Shannon entropy — measures information content |

## Output

The processor adds these columns to your data:

- `luminosity`
- `contrast`
- `blur_level`
- `entropy`

## Requirements

- `opencv-python`
- `pillow`
- `numpy`

## Dependencies

DQM-ML is modular. For visual features:

```bash
# Minimal: use as library only
pip install dqm-ml-images

# For YAML config execution
pip install dqm-ml-job dqm-ml-images

# Full stack with all metrics
pip install dqm-ml-job dqm-ml-core dqm-ml-images dqm-ml-pytorch
```

## See Also

- [Visual Features Documentation](https://safenai.github.io/dqm-ml-workspace/docs/metrics/visual_features/)
- [Configuration Guide](https://safenai.github.io/dqm-ml-workspace/docs/configuration/)