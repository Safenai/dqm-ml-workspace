import logging
from typing import Any

from pandas import DataFrame
import pyarrow as pa

from dqm_ml_core.api.data_processor import DatametricProcessor

logger = logging.getLogger(__name__)


class MetricRunner:
    """
    Orchestrator for executing metric processors on in-memory Pandas DataFrames.

    This class provides a high-level API for users who want to compute metrics
    directly on DataFrames without using the full YAML-driven pipeline.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the runner.

        Args:
            config: Optional configuration for metric default behaviors.
        """
        self.config = config or {}

    def run(self, df: DataFrame, metrics_processors: list[DatametricProcessor]) -> dict[str, Any]:
        """
        Execute the provided metric processors on a DataFrame.

        Args:
            df: The input Pandas DataFrame.
            metrics_processors: List of initialized DatametricProcessor instances.

        Returns:
            A dictionary containing the aggregated dataset-level metrics.
        """
        if df.empty or not metrics_processors:
            logger.warning("Empty DataFrame or no metrics provided to MetricRunner")
            return {}

        metrics_array: dict[str, Any] = {}

        batch = pa.RecordBatch.from_pandas(df)
        batch_features: dict[str, Any] = {}
        batch_metrics: dict[str, Any] = {}

        # Compute features and batch-level metrics
        for metric in metrics_processors:
            logger.debug(f"Processing metric {metric.__class__.__name__}")
            batch_features |= metric.compute_features(batch, prev_features=batch_features)
            batch_metrics |= metric.compute_batch_metric(batch_features)

        # Merge batch metrics (trivial here as there's only one batch)
        for k, v in batch_metrics.items():
            metrics_array[k] = v

        # Compute dataset-level metrics
        dataset_metrics: dict[str, Any] = {}
        for metric in metrics_processors:
            logger.debug(f"Computing final score for {metric.__class__.__name__}")
            dataset_metrics |= metric.compute(batch_metrics=metrics_array)

        return dataset_metrics
