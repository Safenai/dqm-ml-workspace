"""Parquet data loader for reading Parquet files.

This module contains the ParquetDataLoader and ParquetDataSelection classes
for loading and iterating over Parquet file data.
"""

import fnmatch
import logging
import os
from typing import Any

from dqm_ml_core.utils.matching import has_pattern, resolve_include_exclude, resolve_patterns
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as fs
import pyarrow.parquet as pq

# COMPATIBILITY : from typing import Any, override # When support of 3.10 and 3.11 will be removed
from typing_extensions import override

from dqm_ml_job.dataloaders.proto import DataSelection

logger = logging.getLogger(__name__)


def _fnmatch_to_regex(pattern: str) -> str:
    """Convert fnmatch pattern to regex pattern.

    Args:
        pattern: fnmatch pattern with * and ? wildcards.

    Returns:
        Regex pattern string.
    """
    return fnmatch.translate(pattern)


def _match_wildcard_arrow(col_expr: Any, patterns: list[str]) -> Any:
    """Return pyarrow expression for wildcard matching.

    Args:
        col_expr: pyarrow field expression.
        patterns: List of fnmatch patterns.

    Returns:
        pyarrow compute expression for OR of all patterns.
    """
    regex_patterns = [_fnmatch_to_regex(p) for p in patterns]
    # Combine with OR
    combined_regex = "|".join(f"({p})" for p in regex_patterns)
    return pc.match_substring_regex(col_expr, combined_regex)


def _resolve_pyarrow_type(type_name: str) -> Any:
    """Map TransformType string to pyarrow DataType."""
    import pyarrow as pa

    mapping = {
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
        "str": pa.utf8(),
        "categorical": pa.dictionary(pa.int32(), pa.utf8()),
    }
    return mapping[type_name]


class ParquetDataSelection(DataSelection):
    """A specific selection of data from a Parquet dataset.

    This class represents a filtered subset of a Parquet dataset
    and provides an iterator over PyArrow RecordBatches.

    Attributes:
        name: Name identifier for this selection.
        path: Path to the Parquet file or directory.
        batch_size: Number of rows per batch.
        threads: Number of threads for parallel reading.
        filters_dict: Optional dictionary of column filters to apply.
        filesystem: Optional PyArrow filesystem for reading.
        sample_path: List of sample path configs describing column path prefixes.
        transforms: List of transform configs (column cast operations).
    """

    def __init__(
        self,
        name: str,
        path: str,
        batch_size: int = 100_000,
        threads: int = 4,
        filters_dict: dict[str, Any] | None = None,
        filesystem: Any | None = None,
        sample_path: list[dict[str, Any]] | None = None,
        transforms: list[dict[str, Any]] | None = None,
    ):
        """Initialize a Parquet data selection.

        Args:
            name: Name identifier for this selection.
            path: Path to the Parquet file or directory.
            batch_size: Number of rows per batch (default: 100000).
            threads: Number of threads for parallel reading (default: 4).
            filters_dict: Optional dictionary of column filters to apply.
            filesystem: Optional PyArrow filesystem for reading.
            sample_path: List of sample path configs describing column path prefixes.
            transforms: List of transform configs (column cast operations).
        """
        self.name = name
        self.path = path
        self.batch_size = batch_size
        self.threads = threads
        self.filters_dict = filters_dict
        self.filesystem = filesystem
        self.sample_path = sample_path or []
        self.transforms = transforms or []
        self.columns_list: list[str] | None = None
        self.dataset: pq.ParquetDataset | None = None
        self.samples_count: int = 0

    def _build_filter_expr(self) -> Any:
        """Build a PyArrow filter expression from the filters dictionary.

        Converts the filters_dict configuration into a combined pyarrow
        compute expression using AND logic across all filter conditions.

        Returns:
            PyArrow compute expression for filtering, or None if no filters.
        """
        if self.filters_dict is None:
            return None
        expr = None
        for col, val in self.filters_dict.items():
            if self.columns_list is None:
                self.columns_list = [col]
            elif col not in self.columns_list:
                self.columns_list.append(col)
            from dqm_ml_job.dataloaders.filters import build_filter_condition

            col_expr = build_filter_condition(
                col,
                val,
                wildcard_fn=lambda c, vals: _match_wildcard_arrow(pc.field(c), vals),
                isin_fn=lambda c, vals: pc.is_in(pc.field(c), pa.array(vals)),
                equal_fn=lambda c, v: pc.equal(pc.field(c), v),
            )
            expr = col_expr if expr is None else (expr & col_expr)
        return expr

    @override
    def bootstrap(self, columns_list: list[str]) -> None:
        """Initialize the parquet dataset and filter expression.

        Args:
            columns_list: Names of columns to load from the parquet file.
                Empty list means read all columns.
        """
        self.columns_list = columns_list or None
        logger.debug(f"[DEBUG] ParquetDataSelection.bootstrap: {self.name} received columns_list = {columns_list}")
        self.filter_expr = self._build_filter_expr()
        logger.debug(f"[DEBUG] ParquetDataSelection.bootstrap: filter_expr = {self.filter_expr}")
        self.dataset = pq.ParquetDataset(self.path, filters=self.filter_expr, filesystem=self.filesystem)
        if len(self.dataset.fragments) > 0:
            self.samples_count = sum(p.count_rows() for p in self.dataset.fragments)
        else:
            self.samples_count = 0

    def __len__(self) -> int:
        return int(self.samples_count)

    @override
    def get_nb_batches(self) -> int:
        """Return the estimated number of batches in this selection.

        Returns:
            Number of batches based on total samples and batch size.
        """
        return int(len(self) / self.batch_size) + (len(self) % self.batch_size > 0)

    @override
    def __iter__(self) -> Any:
        if self.dataset is None:
            return
        logger.debug(f"[DEBUG] ParquetDataSelection.__iter__: {self.name} using columns_list = {self.columns_list}")
        for file in self.dataset.files:
            parquet_file = pq.ParquetFile(file, filesystem=self.filesystem)
            batch_iterator = parquet_file.iter_batches(
                batch_size=self.batch_size,
                columns=self.columns_list,
                use_threads=self.threads,
            )
            for batch in batch_iterator:
                if self.filter_expr is not None:
                    batch = batch.filter(self.filter_expr)
                if len(batch) == 0:
                    continue
                batch = self._apply_transforms(batch)
                logger.debug(
                    "[DEBUG] ParquetDataSelection.__iter__: %s yielded batch with columns = %s",
                    self.name,
                    batch.schema.names,
                )
                yield batch

    def _apply_transforms(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Apply column transforms (cast operations) to the batch.

        For each transform entry:
        - If ``in_place``, overwrite the column in-place.
        - Otherwise, append a new column named ``<column>_<to_type>``.
        """
        for t in self.transforms:
            col_idx = batch.schema.get_field_index(t["column"])
            if col_idx == -1:
                continue
            target_type = _resolve_pyarrow_type(t["to_type"])
            cast_col = batch.column(col_idx).cast(target_type)
            if t.get("in_place", False):
                batch = batch.set_column(col_idx, t["column"], cast_col)
            else:
                new_name = f"{t['column']}_{t['to_type']}"
                batch = batch.append_column(pa.field(new_name, target_type), cast_col)
        return batch

    @override
    def __repr__(self) -> str:
        return f"ParquetSelection(name='{self.name}', path='{self.path}', filters={self.filters_dict})"


class ParquetDataLoader:
    """Data loader for Parquet files that generates one or more DataSelections.

    This loader can read from a single Parquet file or a directory of Parquet
    files, optionally splitting the data by a column value to create multiple
    selections.

    Attributes:
        type: The loader type identifier ("parquet").
        filesystem: Optional PyArrow filesystem for reading.
    """

    type: str = "parquet"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize the Parquet data loader.

        Args:
            name: Unique name for this loader instance.
            config: Configuration dictionary containing:
                - path: Path to Parquet file or directory (required)
                - batch_size: Rows per batch (default: 100000)
                - threads: Number of threads (default: 4)
                - split_by: Column name to split selections by
                - split_values: Specific values to split on
                - filter.: list of filters
                - storage: Storage configuration (bool or dict)

        Raises:
            ValueError: If required config keys are missing.
        """
        if config is None:
            config = {}
        self.name = name
        self.config = config
        self.path = config["path"]
        self.batch_size = config.get("batch_size", 100_000)
        self.threads = config.get("threads", 4)
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
        logger.debug(f"[DEBUG] ParquetDataLoader.__init__: filters_dict = {self.filters_dict}")

        self.id_column = config.get("id_column")
        self.sample_path = config.get("sample_path", [])
        self.transforms = config.get("transform", [])

        # Storage filesystem configuration - only for S3 paths, not local paths
        self.filesystem = None
        storage_config = None
        storage_cfg = config.get("storage")
        if storage_cfg:
            # Use StorageConfig model to validate and access fields
            from dqm_ml_core.models.global_ import StorageConfig

            storage_config = StorageConfig.model_validate(storage_cfg)

            self.storage_config = storage_config

            if storage_config.type == "s3":
                from dqm_ml_job.utils.s3 import get_s3_filesystem

                self.filesystem = get_s3_filesystem(storage_config)

    def get_selections(self) -> list[DataSelection]:
        """Create one or more ParquetDataSelection instances based on configuration.

        Returns:
            A list of DataSelection instances. If split_by is configured,
            returns one selection per unique value. Otherwise, returns a
            single selection for the entire dataset.
        """
        path = self.path
        if self.filesystem is not None and isinstance(self.filesystem, fs.S3FileSystem):
            # Use bucket from StorageConfig if available
            bucket_name = self.storage_config.bucket if self.storage_config else os.getenv("S3_BUCKET_NAME", "")
            # Avoid prepending bucket if it's a local path (starts with dqm_data/) or already has bucket
            if bucket_name and not path.startswith(bucket_name + "/"):
                path = f"{bucket_name}/{path}"

        if not self.split_by:
            # Single selection
            return [
                ParquetDataSelection(
                    name=self.name,
                    path=path,
                    batch_size=self.batch_size,
                    threads=self.threads,
                    filters_dict=self.filters_dict,
                    filesystem=self.filesystem,
                    sample_path=self.sample_path,
                    transforms=self.transforms,
                )
            ]

        # Splitting logic
        values = self.split_values
        if values is None:
            # Automatic discovery if split_values not provided
            logger.info(f"Discovering unique values for split_by='{self.split_by}' in {path}")
            table = pq.read_table(path, columns=[self.split_by], filesystem=self.filesystem)
            values = [str(v) for v in pc.unique(table.column(0)).to_pylist() if v is not None]
        else:
            # Expand wildcard patterns in values against available data
            if any(has_pattern(v) for v in values):
                logger.info(f"Expanding wildcard values for split_by='{self.split_by}' in {path}")
                table = pq.read_table(path, columns=[self.split_by], filesystem=self.filesystem)
                available = [str(v) for v in pc.unique(table.column(0)).to_pylist() if v is not None]
                values = resolve_patterns(values, available)

        # Apply split.exclude (including wildcard patterns)
        if self.split and self.split.exclude:
            values = resolve_include_exclude(None, self.split.exclude, values)

        selections: list[DataSelection] = []
        for val in values:
            selection_name = f"{self.name}_{val}"
            # Merge existing filters with the split filter
            merged_filters = (self.filters_dict or {}).copy()
            # TODO: filters shouldn't be on the same column than split by
            # raise error here ? or add this check in pydantic model ? is it possible ?
            merged_filters[self.split_by] = val

            selections.append(
                ParquetDataSelection(
                    name=selection_name,
                    path=path,
                    batch_size=self.batch_size,
                    threads=self.threads,
                    filters_dict=merged_filters,
                    filesystem=self.filesystem,
                    sample_path=self.sample_path,
                    transforms=self.transforms,
                )
            )
        return selections
