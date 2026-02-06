import logging
from typing import Any, override

import pandas as pd
import pyarrow as pa

from dqm_ml_pipeline.dataloaders.proto import DataSelection

logger = logging.getLogger(__name__)


class PandasDataSelection(DataSelection):
    """
    A selection of data from a CSV file loaded via Pandas.
    """

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.data: pd.DataFrame | None = None

    @override
    def bootstrap(self, columns_list: list[str] | None = None) -> None:
        # For CSV, we currently load everything
        self.data = pd.read_csv(self.path, sep=",")

    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0

    @override
    def get_nb_batches(self) -> int:
        return 1 if self.data is not None else 0

    @override
    def __iter__(self) -> Any:
        if self.data is not None:
            yield pa.RecordBatch.from_pandas(self.data)

    @override
    def __repr__(self) -> str:
        return f"PandasSelection(name='{self.name}', path='{self.path}')"


class PandasDataLoader:
    """
    Data loader for CSV files using Pandas.
    """

    type: str = "csv"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        if not config or "path" not in config:
            raise ValueError(f"Configuration for dataloader '{name}' must contain 'path'")
        self.name = name
        self.path = config["path"]

    def get_selections(self) -> list[DataSelection]:
        return [PandasDataSelection(name=self.name, path=self.path)]
