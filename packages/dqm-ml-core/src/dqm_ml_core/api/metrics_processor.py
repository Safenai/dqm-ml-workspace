"""Metrics processor base class.

This module contains the MetricsProcessor base class that all
metric computation processors must inherit from.
"""

import logging
from typing import Any

import pyarrow as pa

from dqm_ml_core.api.processor import Processor
from dqm_ml_core.utils.matching import resolve_include_exclude

logger = logging.getLogger(__name__)


class MetricsProcessor(Processor):
    """
    Base class for all metric computation processors.

    Metric processors compute dataset-level scores (e.g., completeness,
    diversity, representativeness). The primary lifecycle methods are
    ``extract_columns`` (per-batch column selection), ``compute_batch_metric``
    (batch aggregation), and ``compute`` (final dataset-level computation).
    """

    def generated_metrics(self) -> list[str]:
        """
        Return the names of the final metrics produced by this processor.

        Returns:
            A list of metric names.
        """
        outputs = getattr(self, "output_metrics", {})
        return list(outputs.values())

    def extract_columns(self, batch: pa.RecordBatch, prev_features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """
        Select relevant columns from a raw batch for metric computation.

        Args:
            batch: The input pyarrow RecordBatch.
            prev_features: Features already computed by preceding processors.

        Returns:
            A dictionary mapping column names to pyarrow Arrays.
        """
        features = {}

        available = batch.schema.names
        cols = resolve_include_exclude(
            self.input_columns,
            self.exclude_columns,
            available,
        )

        for col in cols:
            if col in prev_features:
                continue

            if col not in available:
                if (
                    self.errors_config
                    and self.errors_config.tabular
                    and self.errors_config.tabular.on_missing_column == "fail_fast"
                ):
                    raise KeyError(f"Column '{col}' not found in batch")
                logger.warning(f"[{self.name}] column '{col}' not found in batch")
                continue
            features[col] = batch.column(col)

        return features

    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """
        Aggregate features into intermediate statistics for the current batch.

        This method is critical for scalability. It should return a compact
        representation of the data (e.g., partial sums) that can be
        efficiently combined later.

        Args:
            features: Dictionary of feature arrays computed on the batch.

        Returns:
            A dictionary of aggregated statistics.
        """
        return {}

    def compute(self, batch_metrics: dict[str, pa.Array]) -> dict[str, Any]:
        """
        Perform the final dataset-level metric calculation.

        Args:
            batch_metrics: The aggregated intermediate statistics from all batches.

        Returns:
            A dictionary containing the final metrics.
        """
        return {}
