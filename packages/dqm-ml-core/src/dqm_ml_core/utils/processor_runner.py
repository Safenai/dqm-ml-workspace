"""Processor runner utility for executing metrics on DataFrames.

This module contains the ProcessorRunner class that provides a high-level
API for running feature and metric processors directly on Pandas DataFrames.
"""

from __future__ import annotations

import logging
from typing import Any

from pandas import DataFrame
import pyarrow as pa

from dqm_ml_core.api.gap_processor import GapProcessor
from dqm_ml_core.api.metrics_processor import MetricsProcessor
from dqm_ml_core.api.processor import Processor

logger = logging.getLogger(__name__)


class ProcessorRunner:
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

    def _compute_batch_level(self, df: DataFrame, processors: list[Processor]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute features and batch-level metrics from non-Gap processors.

        Returns:
            Tuple of (batch_features, batch_metrics) merged dicts.
        """
        from dqm_ml_core.api.features_processor import FeaturesProcessor

        batch = pa.RecordBatch.from_pandas(df)
        batch_features: dict[str, Any] = {}
        batch_metrics: dict[str, Any] = {}

        for metric in processors:
            logger.debug(f"Processing metric {metric.__class__.__name__}")
            if isinstance(metric, FeaturesProcessor):
                batch_features |= metric.compute_features(batch, prev_features=batch_features)
            elif isinstance(metric, MetricsProcessor):
                batch_features |= metric.select_columns(batch, prev_features=batch_features)

            if isinstance(metric, MetricsProcessor):
                batch_metrics |= metric.compute_batch_metric(batch_features)

        return batch_features, batch_metrics

    def _compute_dataset_level(self, batch_metrics: dict[str, Any], processors: list[Processor]) -> dict[str, Any]:
        """Compute dataset-level metrics from MetricsProcessor instances."""
        dataset_metrics: dict[str, Any] = {}
        for metric in processors:
            if not isinstance(metric, MetricsProcessor):
                continue
            logger.debug(f"Computing final score for {metric.__class__.__name__}")
            dataset_metrics |= metric.compute(batch_metrics=batch_metrics)
        return dataset_metrics

    def run(self, df: DataFrame, metrics_processors: list[Processor]) -> dict[str, Any]:
        """
        Execute the provided metric processors on a DataFrame.

        Args:
            df: The input Pandas DataFrame.
            metrics_processors: List of initialized Processor instances.

        Returns:
            A dictionary containing the aggregated dataset-level metrics.
        """
        if df.empty or not metrics_processors:
            logger.warning("Empty DataFrame or no metrics provided to ProcessorRunner")
            return {}

        active = [p for p in metrics_processors if not isinstance(p, GapProcessor)]
        _, batch_metrics = self._compute_batch_level(df, active)
        return self._compute_dataset_level(batch_metrics, active)
