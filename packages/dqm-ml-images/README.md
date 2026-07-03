# DQM-ML Images

Image feature extraction package for DQM-ML V2. Provides metrics for assessing image dataset quality.

## Installation

```bash
pip install dqm-ml-images
```

> **Note:** `dqm-ml-images` provides **Features Processors** only — no CLI or job orchestration. Use directly via Python or with `dqm-ml-job` for YAML config execution.

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

## Adding a Custom Feature

Features are currently added directly to the `VisualFeaturesProcessor` class. There are five locations to update:

### 1. Define the output column name

Add to `DEFAULT_OUTPUTS` in `visual_features.py`:

```python
DEFAULT_OUTPUTS: dict[str, str] = {
    "luminosity": "luminosity",
    "contrast": "contrast",
    "blur": "blur",
    "entropy": "entropy",
    "colorfulness": "colorfulness",  # new
}
```

### 2. Add to validation tuple

Update the tuple in `_validate_output_features()`:

```python
for k in ("luminosity", "contrast", "blur", "entropy", "colorfulness"):
```

### 3. Add to generated features

Update the tuple in `generated_features()`:

```python
for fk in ("luminosity", "contrast", "blur", "entropy", "colorfulness"):
```

### 4. Write a computation helper

Add a static or instance method:

```python
@staticmethod
def _colorfulness(gray: np.ndarray) -> float:
    """Mean saturation as a simple colorfulness proxy."""
    # gray is already grayscale at this point if grayscale=True;
    # for a real colorfulness metric the method would need RGB input.
    return float(np.mean(gray))
```

### 5. Wire into the dispatch loop

Add a branch in `compute_features()`:

```python
for fk in ("luminosity", "contrast", "blur", "entropy", "colorfulness"):
    func = {"luminosity": np.mean, "contrast": np.std, "colorfulness": np.mean}.get(fk)
    if fk == "blur":
        arr = self._compute_scalar_feature(gray_images, self._variance_of_laplacian, True)
    elif fk == "entropy":
        arr = self._compute_scalar_feature(gray_images, self._entropy, True)
    elif fk == "colorfulness":
        arr = self._compute_scalar_feature(gray_images, self._colorfulness, True)
    else:
        arr = self._compute_scalar_feature(gray_images, func, self.normalize)
    result[self._output_column_name(image_column, fk)] = arr
```

### 6. (Optional) Add to the Pydantic default

If you want the feature on by default, add it to `ImageFeaturesProcessorConfig.features` in `processors.py`:

```python
features: list[str] = Field(
    default=["luminosity", "contrast", "blur", "entropy", "colorfulness"],
)
```

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