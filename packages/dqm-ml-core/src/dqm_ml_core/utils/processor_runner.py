"""Processor runner utility for executing metrics on DataFrames.

This module contains the ProcessorRunner class that provides a high-level
API for running feature and metric processors directly on Pandas DataFrames.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
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

    @staticmethod
    def _df_to_record_batch(df: DataFrame) -> pa.RecordBatch:
        """Convert a pandas DataFrame to a pyarrow RecordBatch.

        Handles conversion of object columns containing uniform-size numpy
        arrays (e.g. embeddings) to FixedSizeListArray, which is required
        by processors like DomainGapProcessor.
        """
        arrays: list[pa.Array] = []
        for col_name in df.columns:
            series = df[col_name]
            if series.dtype == object and len(series) > 0:
                first = series.iloc[0]
                if isinstance(first, np.ndarray):
                    dim = first.shape[0]
                    if all(isinstance(v, np.ndarray) and v.shape == (dim,) for v in series):
                        flat = np.vstack(series.values).astype(np.float64)
                        arr = pa.FixedSizeListArray.from_arrays(pa.array(flat.flatten().tolist()), dim)
                        arrays.append(arr)
                        continue
            arrays.append(pa.Array.from_pandas(series))
        table = pa.table(dict(zip(df.columns, arrays, strict=True)))
        return table.to_batches()[0]

    def _compute_batch_level(self, df: DataFrame, processors: list[Processor]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute features and batch-level metrics from non-Gap processors.

        Returns:
            Tuple of (batch_features, batch_metrics) merged dicts.
        """
        from dqm_ml_core.api.features_processor import FeaturesProcessor

        batch = self._df_to_record_batch(df)
        batch_features: dict[str, Any] = {}
        batch_metrics: dict[str, Any] = {}

        for metric in processors:
            logger.debug(f"Processing metric {metric.__class__.__name__}")
            if isinstance(metric, FeaturesProcessor):
                batch_features |= metric.compute_features(batch, prev_features=batch_features)
            elif isinstance(metric, MetricsProcessor):
                # select_columns returns intermediate columns for metric computation,
                # not final features - use only for compute_batch_metric
                intermediate = metric.select_columns(batch, prev_features=batch_features)

            if isinstance(metric, MetricsProcessor):
                batch_metrics |= metric.compute_batch_metric(intermediate)

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

    def run(self, df: DataFrame, processors: list[Processor]) -> dict[str, Any]:
        """
        Execute the provided processors on a DataFrame.

        Args:
            df: The input Pandas DataFrame.
            processors: List of initialized Processor instances.

        Returns:
            A dictionary containing the aggregated dataset-level metrics
            and/or per-sample features. Features from FeaturesProcessor
            instances are included alongside metrics from MetricsProcessor instances.
        """
        if df.empty or not processors:
            logger.warning("Empty DataFrame or no processors provided to ProcessorRunner")
            return {}

        active = [p for p in processors if not isinstance(p, GapProcessor)]
        batch_features, batch_metrics = self._compute_batch_level(df, active)
        dataset_metrics = self._compute_dataset_level(batch_metrics, active)

        # Merge features (from FeaturesProcessor) and metrics (from MetricsProcessor)
        result = {}
        result.update(batch_features)  # Per-sample features
        result.update(dataset_metrics)  # Aggregated metrics
        return result

    def _compute_features(self, df: DataFrame, features: list[Processor]) -> DataFrame:
        """Run FeaturesProcessors on a DataFrame and return a new DataFrame with computed columns.

        Args:
            df: The input DataFrame.
            features: List of FeaturesProcessor instances.

        Returns:
            A new DataFrame with the original columns plus any new feature columns.
        """
        result = self.run(df, features)
        if not result:
            return df
        out = df.copy()
        for col, values in result.items():
            out[col] = values
        return out

    def run_gap(
        self,
        source_df: DataFrame,
        target_df: DataFrame,
        processor: GapProcessor,
        features: list[Processor] | None = None,
        source_selection_name: str = "source",
        target_selection_name: str = "target",
    ) -> dict[str, Any]:
        """
        Execute a GapProcessor on two DataFrames and compute the domain gap.

        This method handles the two-dataset execution pattern required by GapProcessor:
        1. Optionally compute features (e.g. embeddings) on both DataFrames
        2. Process source DataFrame to compute source statistics
        3. Process target DataFrame to compute target statistics
        4. Compute the domain gap delta between source and target

        Args:
            source_df: The source dataset DataFrame.
            target_df: The target dataset DataFrame.
            processor: An initialized GapProcessor instance.
            features: Optional list of FeaturesProcessor instances to run on both
                DataFrames before computing the gap. For example, pass an
                ImageEmbeddingProcessor to compute embeddings from raw images.
            source_selection_name: Name for the source selection (default: "source").
            target_selection_name: Name for the target selection (default: "target").

        Returns:
            A dictionary containing:
            - {metric_name}: The domain gap metric value
            - "selection_source": Name of the source selection
            - "selection_target": Name of the target selection
        """
        if source_df.empty or target_df.empty:
            logger.warning("Empty DataFrame provided to run_gap")
            return {}

        # Compute features if provided (e.g. embeddings from raw images)
        if features:
            source_df = self._compute_features(source_df, features)
            target_df = self._compute_features(target_df, features)

        # Process source dataset
        source_batch = self._df_to_record_batch(source_df)
        source_features = processor.select_features(source_batch, prev_features={})
        source_batch_metrics = processor.compute_batch_metric(source_features)
        source_stats = processor.compute(source_batch_metrics)

        # Process target dataset
        target_batch = self._df_to_record_batch(target_df)
        target_features = processor.select_features(target_batch, prev_features={})
        target_batch_metrics = processor.compute_batch_metric(target_features)
        target_stats = processor.compute(target_batch_metrics)

        # Compute domain gap delta
        result = processor.compute_delta(source_stats, target_stats)

        # Add selection metadata
        result["selection_source"] = pa.array([source_selection_name])
        result["selection_target"] = pa.array([target_selection_name])

        return result
