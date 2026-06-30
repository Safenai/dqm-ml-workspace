"""Output configuration models for pipeline results.

Defines output settings for features, metrics, and domain-gap computations
including destination paths and column filtering.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParquetOutputConfig(BaseModel):
    """Configuration for the Parquet output writer.

    Attributes:
        path_pattern: Output file path format string.
        columns: Columns to include in output.
        exclude: Columns to exclude from output.
        add_dataloader_column: Whether to add a dataloader identifier column.
        dataloader_column_name: Name for the dataloader column.
        storage: Optional storage configuration (bool or dict).
    """

    model_config = ConfigDict(extra="forbid")

    path_pattern: str = Field(description="Output file path format string.")
    columns: list[str] = Field(default_factory=list, description="Columns to include in output.")
    exclude: list[str] | None = Field(default=None, description="Columns to exclude from output.")
    add_dataloader_column: bool = Field(default=False, description="Add a dataloader identifier column.")
    dataloader_column_name: str = Field(default="dataloader", description="Name for the dataloader column.")
    storage: bool | dict[str, Any] | None = Field(
        default=None,
        description="Storage configuration (bool or dict with S3 settings).",
    )


class FeaturesOutputsConfig(BaseModel):
    """Output configuration for computed features."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Destination path for feature output.")
    include: list[str] | None = Field(
        default=None,
        description="Feature columns to include (fnmatch patterns supported).",
    )
    exclude: list[str] | None = Field(
        default=None,
        description="Feature columns to exclude (fnmatch patterns supported).",
    )


class MetricsOutputsConfig(BaseModel):
    """Output configuration for computed metrics."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Destination path for metrics output.")


class GapOutputsConfig(BaseModel):
    """Output configuration for domain-gap results."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Destination path for domain-gap output.")
    pairwise: bool = Field(default=True, description="Include pairwise (delta) results.")
