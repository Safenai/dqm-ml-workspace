"""Data loading and transformation configuration models.

Defines models for filters, path sampling, data splitting, column transformations,
and dataloader configurations for Parquet and CSV sources.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dqm_ml_core.models.global_ import StorageConfig


class FilterConfig(BaseModel):
    """Row-level filter applied during data loading."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name to filter on.")
    values: list[bool] | list[str] | list[int] | list[float] = Field(
        description="Value(s) to keep. Rows where column matches are included.",
    )


class SamplePathConfig(BaseModel):
    """Per-column path prefix configuration."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name containing relative file paths.")
    prefix: str | None = Field(default=None, description="Base directory for resolving relative paths.")


class SplitConfig(BaseModel):
    """How to split data into named groups (e.g. train / test)."""

    model_config = ConfigDict(extra="forbid")

    by: str = Field(description="Column used to determine the split group.")
    values: list[str] | None = Field(
        default=None,
        description="Explicit list of split-group values to materialise. Auto-discovered if None.",
    )
    exclude: list[str] | None = Field(
        default=None,
        description="Split-group values to exclude (fnmatch patterns supported).",
    )


class TransformType(str, Enum):
    """Target data type for column transformations."""

    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BOOL = "bool"
    STR = "str"
    CATEGORICAL = "categorical"


class TransformConfig(BaseModel):
    """Column type-casting transformation."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name to transform.")
    to_type: TransformType = Field(description="Target data type.")
    in_place: bool = Field(default=False, description="Overwrite the original column in place.")


class DataLoaderConfig(BaseModel):
    """Configuration for a single dataloader (Parquet or CSV)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique name for this dataloader.")
    type: Literal["parquet", "csv"] = Field(description="Data file format.")
    path: str = Field(description="Glob pattern or path to data files.")
    id_column: str | None = Field(default=None, description="Column used as row identifier.")
    batch_size: int = Field(default=10000, description="Number of rows per batch.")
    filters: list[FilterConfig] | None = None
    sample_path: list[SamplePathConfig] | None = None
    split: SplitConfig | None = None
    transform: list[TransformConfig] | None = None
    storage: StorageConfig | None = None


class DataLoadersConfig(BaseModel):
    """Collection of dataloaders for a job."""

    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig | None = Field(
        default=None,
        description="Default storage config inherited by all loaders.",
    )
    loaders: list[DataLoaderConfig] = Field(description="List of dataloader configurations.")
