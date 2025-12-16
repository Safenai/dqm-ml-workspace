import logging
from typing import Any

from pandas import DataFrame
import pyarrow as pa

from dqm_ml_core.api.data_processor import DatametricProcessor

logger = logging.getLogger(__name__)


class MetricRunner:
    """
    Main class for processing metrics through a configurable pipeline
    of data loaders, metrics processors, and output writers.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the pipeline with a given configuration.

        Args:
            config: Dictionary containing:
                - loading config
        Return:
            List of computed metrics
        """

    def run(self, df: DataFrame, metrics_processors: list[DatametricProcessor]) -> dict[str, Any]:
        """
        Execute the dataset processing pipeline.
        """

        metrics_array: dict[str, Any] = {}

        batch = pa.RecordBatch.from_pandas(df)
        batch_features: dict[str, Any] = {}
        batch_metrics: dict[str, Any] = {}

        # Compute features and batch-level metrics
        for metric in metrics_processors:
            logger.debug(f"Processing metric {metric.__class__.__name__} for dataloader")
            batch_features |= metric.compute_features(batch, prev_features=batch_features)
            batch_metrics |= metric.compute_batch_metric(batch_features)

        # Merge batch metrics
        for k, v in batch_metrics.items():
            if k in metrics_array:
                # For histogram metrics, we need to sum, not concatenate
                metrics_array[k] = pa.concat_arrays([metrics_array[k], v])
            else:
                metrics_array[k] = v

        # Compute dataset-level metrics
        dataset_metrics: dict[str, Any] = {}
        for metric in metrics_processors:
            logger.debug(f"Processing metric computation {metric.__class__.__name__} for dataloader")
            dataset_metrics |= metric.compute(batch_metrics=metrics_array)
        return dataset_metrics
