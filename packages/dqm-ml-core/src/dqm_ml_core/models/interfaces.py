"""Pipeline interface configuration models.

Defines the three main pipeline interfaces (features, metrics, gap) each
composing storage, compute, error handling, outputs, and processor lists.
"""

from pydantic import BaseModel, ConfigDict, Field

from dqm_ml_core.models.global_ import ComputeConfig, ErrorsConfig, StorageConfig
from dqm_ml_core.models.outputs import FeaturesOutputsConfig, GapOutputsConfig, MetricsOutputsConfig
from dqm_ml_core.models.processors import ProcessorConfig


class _InterfaceBase(BaseModel):
    """Base for pipeline-interface configs (storage, compute, error overrides)."""

    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig | None = Field(
        default=None,
        description="Storage override for this interface.",
    )
    compute: ComputeConfig | None = Field(
        default=None,
        description="Compute override for this interface.",
    )
    errors: ErrorsConfig | None = Field(
        default=None,
        description="Error-handling override for this interface.",
    )


class FeaturesInterfaceConfig(_InterfaceBase):
    """Feature-extraction pipeline configuration."""

    outputs: FeaturesOutputsConfig | None = None
    processors: list[ProcessorConfig] = Field(default=[], description="Ordered list of feature processors.")


class MetricsInterfaceConfig(_InterfaceBase):
    """Metric-computation pipeline configuration."""

    outputs: MetricsOutputsConfig | None = None
    processors: list[ProcessorConfig] = Field(default=[], description="Ordered list of metric processors.")


class GapInterfaceConfig(_InterfaceBase):
    """Domain-gap computation pipeline configuration."""

    outputs: GapOutputsConfig | None = None
    processors: list[ProcessorConfig] = Field(default=[], description="Ordered list of domain-gap processors.")
