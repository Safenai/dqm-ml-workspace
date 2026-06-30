"""Root job configuration model for DQM-ML pipelines.

Defines the top-level JobConfig that composes all pipeline stage configurations
including storage, compute, error handling, data loading, and processor interfaces.
"""

from pydantic import BaseModel, ConfigDict

from dqm_ml_core.models.dataloaders import DataLoadersConfig
from dqm_ml_core.models.global_ import ComputeConfig, ErrorsConfig, StorageConfig
from dqm_ml_core.models.interfaces import FeaturesInterfaceConfig, GapInterfaceConfig, MetricsInterfaceConfig


class JobConfig(BaseModel):
    """Root configuration for a dqm-ml job. Each field maps to a pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig | None = None
    compute: ComputeConfig | None = None
    errors: ErrorsConfig | None = None
    dataloaders: DataLoadersConfig
    features: FeaturesInterfaceConfig | None = None
    metrics: MetricsInterfaceConfig | None = None
    gap: GapInterfaceConfig | None = None
