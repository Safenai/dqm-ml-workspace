"""Column selection and transformation models.

Provides models for column renaming, input selection, exclusion patterns,
and prefix/suffix modifiers used across processor configurations.
"""

from pydantic import BaseModel, ConfigDict, Field


class ColumnRename(BaseModel):
    """Mapping from an original column name to a new name."""

    model_config = ConfigDict(extra="forbid")

    from_: str = Field(alias="from", description="Original column name.")
    to: str = Field(description="New column name.")


class ColumnsConfig(BaseModel):
    """Column selection, exclusion, renaming, and prefix/suffix operations."""

    model_config = ConfigDict(extra="forbid")

    input: list[str] = Field(
        default=[],
        description="Columns to read (fnmatch patterns supported). [] reads all.",
    )
    exclude: list[str] | None = Field(
        default=None,
        description="Columns to exclude (fnmatch patterns supported).",
    )
    rename: list[ColumnRename] | None = None
    prefix: str = Field(default="", description="Prefix prepended to each column name.")
    suffix: str = Field(default="", description="Suffix appended to each column name.")
