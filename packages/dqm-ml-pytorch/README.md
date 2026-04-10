# DQM-ML PyTorch

PyTorch-based metrics for DQM-ML V2. Provides advanced domain gap analysis for comparing dataset distributions.

## Installation

```bash
pip install dqm-ml-pytorch
```

> **Note:** `dqm-ml-pytorch` provides metric processors only — no CLI or job orchestration. Use directly via Python or with `dqm-ml-job` for YAML config execution.

## Usage

### Using Python Directly

```python
import numpy as np
from dqm_ml_pytorch import DomainGapProcessor

# Create source and target embeddings (example data)
source_embeddings = np.random.randn(100, 2048).astype(np.float32)
target_embeddings = np.random.randn(100, 2048).astype(np.float32)

# Create and configure the processor
processor = DomainGapProcessor(
    name="domain_drift",
    config={
        "INPUT": {"embedding_col": "embedding"},
        "DELTA": {"metric": "mmd_linear"}
    }
)

# Compute statistics for both datasets
source_stats = processor.compute_batch_metric({"embedding": source_embeddings})
target_stats = processor.compute_batch_metric({"embedding": target_embeddings})

# Compute the domain gap delta
result = processor.compute_delta(source_stats, target_stats)
print(f"Domain Gap (MMD): {result['domain_gap_mmd_linear']}")
```

### With dqm-ml-job

For running from a YAML config, install together with `dqm-ml-job`:

```bash
pip install dqm-ml-job dqm-ml-pytorch
```

Then use this config:

```yaml
metrics_processor:
  domain_drift:
    type: domain_gap
    INPUT:
      embedding_col: "features"
    DELTA:
      metric: "mmd_linear"
```

## Features

| Metric | Full Name | Best For |
|--------|-----------|----------|
| **FID** | Fréchet Inception Distance | Image embeddings |
| **MMD** | Maximum Mean Discrepancy | General kernel-based comparison |
| **Wasserstein** | 1D Earth Mover's Distance | 1D distributions |
| **KLMVN** | KL-Divergence (Multivariate Normal) | Gaussian distributions |

## Output

Returns statistical distance values:
- `domain_gap_fid`
- `domain_gap_mmd_linear`
- `domain_gap_wasserstein_1d`
- `domain_gap_klmvn_diag`

## Requirements

- `torch`
- `torchvision`
- `scipy`

## Dependencies

DQM-ML is modular. For domain gap metrics:

```bash
# Minimal: use as library only
pip install dqm-ml-pytorch

# For YAML config execution
pip install dqm-ml-job dqm-ml-pytorch

# Full stack with all metrics
pip install dqm-ml-job dqm-ml-core dqm-ml-pytorch
```

## See Also

- [Domain Gap Documentation](https://safenai.github.io/dqm-ml-workspace/docs/metrics/domain_gap/)
- [Configuration Guide](https://safenai.github.io/dqm-ml-workspace/docs/configuration/)