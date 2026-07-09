"""Gap processor base class.

This module contains the GapProcessor base class that all
domain gap processors must inherit from.
"""

import logging
from typing import Any

import pyarrow as pa

from dqm_ml_core.api.processor import Processor
from dqm_ml_core.utils.matching import resolve_include_exclude

logger = logging.getLogger(__name__)


class GapProcessor(Processor):
    """
    Base class for all domain gap processors.

    Gap processors compute distribution shift between two datasets
    (e.g., MMD, FID, KL divergence). The primary lifecycle methods are
    ``select_features`` (per-batch column selection aware of previous features),
    ``compute_batch_metric`` (batch aggregation), ``compute`` (final dataset-level
    statistics), and ``compute_delta`` (pairwise comparison).
    """

    def select_features(self, batch: pa.RecordBatch, prev_features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """
        Extract relevant columns from a batch, resolving patterns against
        both batch columns and previously computed upstream features.

        Args:
            batch: The input pyarrow RecordBatch.
            prev_features: Features already computed by preceding processors.

        Returns:
            A dictionary mapping column names to pyarrow Arrays.
        """
        available = list(prev_features.keys()) + batch.schema.names
        cols = resolve_include_exclude(
            self.input_columns,
            self.exclude_columns,
            available,
        )

        features: dict[str, pa.Array] = {}
        for col in cols:
            if col in prev_features:
                continue
            if col not in batch.schema.names:
                logger.warning(f"[{self.name}] column '{col}' not found in batch")
                continue
            features[col] = batch.column(col)

        return features

    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """
        Aggregate features into intermediate statistics for the current batch.

        Args:
            features: Dictionary of feature arrays computed on the batch.

        Returns:
            A dictionary of aggregated statistics.
        """
        return {}

    def compute(self, batch_metrics: dict[str, pa.Array]) -> dict[str, Any]:  # NOSONAR
        """
        Perform the final dataset-level aggregation of batch statistics.

        Args:
            batch_metrics: The aggregated intermediate statistics from all batches.

        Returns:
            A dictionary containing the final dataset-level statistics.
        """
        # SonarQube raises a warning because batch_metrics is not used.
        # It is irrelevant because compute is implemented in child classes which use batch_metrics.
        return {}

    def compute_delta(self, source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
        """
        Compare metrics between two different dataselections.

        Args:
            source: Final metrics from the source dataselection.
            target: Final metrics from the target dataselection.

        Returns:
            A dictionary containing distance or difference scores.
        """
        return {}
