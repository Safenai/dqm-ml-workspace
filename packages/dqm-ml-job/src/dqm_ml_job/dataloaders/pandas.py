"""Pandas data loader for reading CSV files.

This module contains the PandasDataLoader and PandasDataSelection classes
for loading and iterating over CSV file data using Pandas.
"""

import fnmatch
import logging
from typing import Any

from dqm_ml_core.utils.matching import has_pattern, resolve_include_exclude, resolve_patterns
import pandas as pd
import pyarrow as pa

# COMPATIBILITY : from typing import Any, override # When support of 3.10 and 3.11 will be removed
from typing_extensions import override

from dqm_ml_job.dataloaders.proto import DataSelection

logger = logging.getLogger(__name__)


def _match_wildcard(series: pd.Series, patterns: list[str]) -> pd.Series:
    """Return boolean mask for rows matching any wildcard pattern.

    Args:
        series: Pandas Series to filter.
        patterns: List of patterns with fnmatch wildcards (* and ?).

    Returns:
        Boolean Series indicating matches.
    """
    mask = pd.Series(False, index=series.index)
    for pattern in patterns:
        mask = mask | series.astype(str).apply(lambda x, p=pattern: fnmatch.fnmatch(x, p))
    return mask


_PANDAS_TYPE_MAP: dict[str, Any] = {
    "int32": "int32",
    "int64": "int64",
    "float32": "float32",
    "float64": "float64",
    "bool": "bool",
    "str": "string",
    "categorical": "category",
}


def _apply_pandas_transforms(df: pd.DataFrame, transforms: list[dict[str, Any]]) -> None:
    """Apply column transforms to a pandas DataFrame in-place."""
    for t in transforms:
        col = t["column"]
        if col not in df.columns:
            continue
        pandas_type = _PANDAS_TYPE_MAP[t["to_type"]]
        if t.get("in_place", False):
            df[col] = df[col].astype(pandas_type)
        else:
            new_name = f"{col}_{t['to_type']}"
            df[new_name] = df[col].astype(pandas_type)


class PandasDataSelection(DataSelection):
    """A selection of data from a CSV file loaded via Pandas.

    This class represents data loaded from a CSV file and provides
    an iterator over PyArrow RecordBatches.

    Attributes:
        name: Name identifier for this selection.
        path: Path to the CSV file.
        data: The loaded pandas DataFrame.
        sample_path: List of sample path configs describing column path prefixes.
        transforms: List of transform configs (column cast operations).
    """

    def __init__(
        self,
        name: str,
        path: str,
        sample_path: list[dict[str, Any]] | None = None,
        transforms: list[dict[str, Any]] | None = None,
        filters_dict: dict[str, Any] | None = None,
    ):
        """Initialize a Pandas data selection.

        Args:
            name: Name identifier for this selection.
            path: Path to the CSV file.
            sample_path: List of sample path configs describing column path prefixes.
            transforms: List of transform configs (column cast operations).
            filters_dict: Column-value pairs to filter rows by.
        """
        self.name = name
        self.path = path
        self.sample_path = sample_path or []
        self.transforms = transforms or []
        self.filters_dict = filters_dict or {}
        self.data: pd.DataFrame | None = None

    @override
    def bootstrap(self, columns_list: list[str] | None = None) -> None:
        """Load the CSV file into memory as a pandas DataFrame.

        Args:
            columns_list: Unused, kept for API compatibility.
        """
        from dqm_ml_job.dataloaders.filters import build_filter_condition

        data = pd.read_csv(self.path, sep=",")
        assert isinstance(data, pd.DataFrame)
        self.data = data
        for col, val in self.filters_dict.items():
            condition = build_filter_condition(
                col,
                val,
                wildcard_fn=lambda c, vals: _match_wildcard(data[c], vals),
                isin_fn=lambda c, vals: data[c].isin(vals),
                equal_fn=lambda c, v: data[c] == v,
            )
            self.data = self.data[condition.reindex(self.data.index, fill_value=False)]

    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0

    @override
    def get_nb_batches(self) -> int:
        """Return the estimated number of batches (always 1 for CSV).

        Returns:
            1 if data is loaded, 0 otherwise.
        """
        return 1 if self.data is not None else 0

    @override
    def __iter__(self) -> Any:
        if self.data is not None:
            df = self.data.copy() if self.transforms else self.data
            _apply_pandas_transforms(df, self.transforms)
            yield pa.RecordBatch.from_pandas(df)

    @override
    def __repr__(self) -> str:
        return f"PandasSelection(name='{self.name}', path='{self.path}')"


class PandasDataLoader:
    """Data loader for CSV files using Pandas.

    This loader reads CSV files and provides DataSelections for
    processing by the DQM pipeline.

    Attributes:
        type: The loader type identifier ("csv").
    """

    type: str = "csv"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize the Pandas data loader.

        Args:
            name: Unique name for this loader instance.
            config: Configuration dictionary containing:
                - path: Path to CSV file (required)

        Raises:
            ValueError: If required config keys are missing.
        """
        if config is None:
            config = {}
        self.name = name
        self.path = config["path"]
        # Use SplitConfig model fields instead of hardcoded keys
        from dqm_ml_core.models.dataloaders import SplitConfig

        split = config.get("split")
        self.split = SplitConfig.model_validate(split) if split else None
        self.split_by = self.split.by if self.split else None
        self.split_values = self.split.values if self.split else None
        filters = config.get("filters")
        # transform the list of dict into a dict
        self.filters_dict = {}
        if filters is not None:
            for item in filters:
                column = item["column"]
                self.filters_dict[column] = item["values"]
        self.id_column = config.get("id_column")
        self.sample_path = config.get("sample_path", [])
        self.transforms = config.get("transform", [])

        # Storage filesystem configuration - only for S3 paths, not local paths
        self.filesystem = None
        storage_cfg = config.get("storage")
        if storage_cfg:
            # Use StorageConfig model to validate and access fields
            from dqm_ml_core.models.global_ import StorageConfig

            storage_config = StorageConfig.model_validate(storage_cfg)

            if storage_config.type == "s3":
                from dqm_ml_job.utils.s3 import get_s3_filesystem

                self.filesystem = get_s3_filesystem(storage_config)

    def get_selections(self) -> list[DataSelection]:
        """Create one or more PandasDataSelection instances based on split config.

        If split is configured, returns one selection per split value.
        Otherwise returns a single selection for the entire CSV file.

        Returns:
            A list of DataSelection instances.
        """
        if not self.split_by:
            return [
                PandasDataSelection(
                    name=self.name,
                    path=self.path,
                    sample_path=self.sample_path,
                    transforms=self.transforms,
                    filters_dict=self.filters_dict,
                )
            ]

        # Determine split values
        values = self.split_values
        if values is None:
            # Auto-discover unique values from the CSV
            df = pd.read_csv(self.path, sep=",", usecols=[self.split_by])
            values = [str(v) for v in df[self.split_by].unique() if v is not None]
        else:
            # Expand wildcard patterns in values against available data
            if any(has_pattern(v) for v in values):
                df = pd.read_csv(self.path, sep=",", usecols=[self.split_by])
                available = [str(v) for v in df[self.split_by].unique() if v is not None]
                values = resolve_patterns(values, available)

        # Apply split.exclude (including wildcard patterns)
        if self.split and self.split.exclude:
            values = resolve_include_exclude(None, self.split.exclude, values)

        # Create one selection per value
        selections: list[DataSelection] = []
        for val in values:
            selection_name = f"{self.name}_{val}"
            merged_filters = (self.filters_dict or {}).copy()
            merged_filters[self.split_by] = val
            selections.append(
                PandasDataSelection(
                    name=selection_name,
                    path=self.path,
                    sample_path=self.sample_path,
                    transforms=self.transforms,
                    filters_dict=merged_filters,
                )
            )
        return selections
