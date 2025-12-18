import logging
from typing import Any

import pyarrow as pa

logger = logging.getLogger(__name__)


class DatametricProcessor:
    """
    Main class for processing a dataset pipeline.

    """

    def __init__(self, name: str, config: dict[str, Any] | None):
        """
        Initialize the dataset processor

        Args:
            name: Name of the processor
            config: Configuration dictionary (optional)
        """

        self.name = name
        self.config = config or {}

        # Validate input_columns if present
        if "input_columns" in self.config:
            if not isinstance(self.config["input_columns"], list):
                raise ValueError(
                    f"Metric {name} configuration need 'input_columns', got {type(self.config['input_columns'])}"
                )
            self.input_columns = self.config["input_columns"]
        else:
            self.input_columns = []

        # Validate output_columns if present
        if "output_columns" in self.config:
            if not isinstance(self.config["output_columns"], dict):
                raise ValueError(
                    f"Metric {name} configuration need of 'output_columns', got {type(self.config['output_columns'])}"
                )
            self.outputs_columns = self.config["output_columns"]
        else:
            self.outputs_columns = {}

    def needed_columns(self) -> list[str]:
        """
        Return the list of columns needed to compute the metric.

        Returns:
            A list of column names.
        """
        return getattr(self, "input_columns", [])

    def generated_features(self) -> list[str]:
        """
        Return the list of columns that will be generated from the features add in the parquet file.
        Returns:
            A list of column names.
        """

        outputs = getattr(self, "output_features", {})
        return list(outputs.values())

    def generated_metrics(self) -> list[str]:
        """
        Return the list of columns that will be generated from the metric.
        Returns:
            A list of column names.
        """

        outputs = getattr(self, "output_metrics", {})
        return list(outputs.values())

    def compute_features(self, batch: pa.RecordBatch, prev_features: pa.Array = None) -> dict[str, pa.Array]:
        """
        Compute the features for a given batch.
        By default we return the needed columns for metric computation
        Args:
            batch: The input batch of data.

        Returns:
            A dictionary of computed features.
        """
        features = {}

        for col in self.needed_columns():
            if col not in batch.schema.names:
                logger.warning(f"[{self.name}] column '{col}' not found in batch")
                continue
            features[col] = batch.column(col)

        return features

    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """
        Compute metric on of batch that might bbe aggregated. for dataset level
        If no aggregation is possible at batch level, just return the features.
        Args:
            features: dict or features arrays computed on the batch.

        Returns:
            A dictionary of computed features.
        """
        return {}

    def compute(self, batch_metrics: dict[str, pa.Array]) -> dict[str, Any]:
        """
        Compute the metric on a dataset level aggerating sample features

        Args:
            batch_metrics: The intermediate elements computed for each batch of data.

        Returns:
            A dictionary which contains the metrics outputs.
        """
        return {}

    def compute_delta(self, source: dict[str, pa.Array], target: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """
        Compute the metric between two dataset,

        Args:
            source: source for delta computation
            target: target for delta computation
        Returns:
            A dictionary which contains the metrics outputs.
        """
        return {}
