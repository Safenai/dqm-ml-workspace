import logging
from typing import Any, override

import pyarrow.compute as pc
import pyarrow.parquet as pq

from dqm_ml_job.dataloaders.proto import DataSelection

logger = logging.getLogger(__name__)


class ParquetDataSelection(DataSelection):
    """
    A specific selection of data from a Parquet dataset.
    """

    def __init__(
        self,
        name: str,
        path: str,
        batch_size: int = 100_000,
        threads: int = 4,
        filters_dict: dict[str, Any] | None = None,
    ):
        self.name = name
        self.path = path
        self.batch_size = batch_size
        self.threads = threads
        self.filters_dict = filters_dict
        self.columns_list: list[str] | None = None
        self.dataset: pq.ParquetDataset | None = None
        self.samples_count: int = 0

    @override
    def bootstrap(self, columns_list: list[str]) -> None:
        self.columns_list = columns_list
        filter_expr = None
        if self.filters_dict is not None:
            expr = None
            for col, val in self.filters_dict.items():
                if col not in (self.columns_list):
                    self.columns_list.append(col)
                col_expr = pc.equal(pc.field(col), val)
                expr = col_expr if expr is None else pc.and_(expr, col_expr)
            filter_expr = expr
        self.filter_expr = filter_expr
        self.dataset = pq.ParquetDataset(self.path, filters=filter_expr)
        if len(self.dataset.fragments) > 0:
            self.samples_count = sum(p.count_rows() for p in self.dataset.fragments)
        else:
            self.samples_count = 0

    def __len__(self) -> int:
        return int(self.samples_count)

    @override
    def get_nb_batches(self) -> int:
        return int(len(self) / self.batch_size) + (len(self) % self.batch_size > 0)

    @override
    def __iter__(self) -> Any:
        if self.dataset is None:
            return
        for file in self.dataset.files:
            parquet_file = pq.ParquetFile(file)
            batch_iterator = parquet_file.iter_batches(
                batch_size=self.batch_size, columns=self.columns_list, use_threads=self.threads
            )
            for batch in batch_iterator:
                if self.filter_expr is not None:
                    batch = batch.filter(self.filter_expr)
                if len(batch) == 0:
                    continue
                yield batch

    @override
    def __repr__(self) -> str:
        return f"ParquetSelection(name='{self.name}', path='{self.path}', filters={self.filters_dict})"


class ParquetDataLoader:
    """
    Data loader for Parquet files that generates one or more DataSelections.
    """

    type: str = "parquet"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        if not config or "path" not in config:
            raise ValueError(f"Configuration for dataloader '{name}' must contain 'path'")

        self.name = name
        self.config = config
        self.path = config["path"]
        self.batch_size = config.get("batch_size", 100_000)
        self.threads = config.get("threads", 4)
        self.split_by = config.get("split_by")
        self.split_values = config.get("split_values")
        self.filters_dict = config.get("filter", None)

    def get_selections(self) -> list[DataSelection]:
        """
        Create one or more ParquetDataSelection instances based on configuration.
        """
        if not self.split_by:
            # Single selection
            return [
                ParquetDataSelection(
                    name=self.name,
                    path=self.path,
                    batch_size=self.batch_size,
                    threads=self.threads,
                    filters_dict=self.filters_dict,
                )
            ]

        # Splitting logic
        values = self.split_values
        if values is None:
            # Automatic discovery if split_values not provided
            logger.info(f"Discovering unique values for split_by='{self.split_by}' in {self.path}")
            table = pq.read_table(self.path, columns=[self.split_by])
            values = [str(v) for v in pc.unique(table.column(0)).to_pylist() if v is not None]

        selections: list[DataSelection] = []
        for val in values:
            selection_name = f"{self.name}_{val}"
            # Merge existing filters with the split filter
            merged_filters = (self.filters_dict or {}).copy()
            merged_filters[self.split_by] = val

            selections.append(
                ParquetDataSelection(
                    name=selection_name,
                    path=self.path,
                    batch_size=self.batch_size,
                    threads=self.threads,
                    filters_dict=merged_filters,
                )
            )
        return selections
