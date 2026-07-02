"""Completeness metric processor for evaluating data completeness.

This module contains the CompletenessProcessor class that evaluates
the completeness of tabular data by computing non-null value ratios.
"""

import json
import logging
from typing import Any

import numpy as np
import pyarrow as pa

# COMPATIBILITY : from typing import Any, override # When support of 3.10 and 3.11 will be removed
from typing_extensions import override

from dqm_ml_core.api.metrics_processor import MetricsProcessor
from dqm_ml_core.models.processors import CompletenessProcessorConfig

logger = logging.getLogger(__name__)


class CompletenessProcessor(MetricsProcessor):
    """
    Data completeness processor that evaluates the completeness of tabular data.

    This processor calculates completeness scores (ratio of non-null to
    total values) for specified columns and provides overall dataset
    completeness metrics.

    The processor operates at multiple levels:
    - Batch level: Aggregated counts for streaming processing
    - Dataset level: Final completeness scores and statistics
    """

    def __init__(self, name: str = "completeness", config: dict[str, Any] | None = None) -> None:
        """
        Initialize the completeness processor.

        Args:
            name: Name of the processor.
            config: Configuration dictionary containing:
                - input_columns: List of columns to analyze.
                - output_metrics: Mapping of metric names to column names.
                - include_per_column: Include per-column completeness scores.
                - include_overall: Include overall completeness score.
        """
        super().__init__(name, config)

        cfg = CompletenessProcessorConfig.model_validate({**self.config, "name": self.name})

        # Which completeness levels to compute: per-column, overall, and metadata
        self.include_per_column: bool = cfg.include_per_column
        self.include_overall: bool = cfg.include_overall
        self.include_metadata: bool = cfg.include_metadata

        # Output column mappings
        self.output_metrics: dict[str, str] = {}

    @override
    def generated_metrics(self) -> list[str]:
        """
        Return the list of metric columns that will be generated.

        Returns:
            List of output metric column names
        """
        # TODO : manage output metrics names with configuration
        metrics = []

        if self.include_overall:
            overall_key = self.output_metrics.get("overall_completeness", "completeness_overall")
            metrics.append(overall_key)

        if self.include_per_column:
            for col in self.input_columns or []:
                col_key = self.output_metrics.get(f"completeness_{col}", f"completeness_{col}")
                metrics.append(col_key)

        return metrics

    @override
    def extract_columns(
        self, batch: pa.RecordBatch, prev_features: dict[str, pa.Array] | None = None
    ) -> dict[str, pa.Array]:
        """
        Extract the needed columns from the batch for completeness analysis.

        This method simply passes through the columns we need to analyze,
        as completeness calculation is done at batch and dataset levels.

        Args:
            batch: Input batch of data
            prev_features: Previous features (not used in this processor)

        Returns:
            Dictionary containing the columns to analyze
        """
        features = {}

        columns_to_analyze = self.input_columns if self.input_columns else batch.column_names

        for col in columns_to_analyze:
            if col not in batch.schema.names:
                logger.warning(f"[{self.name}] column '{col}' not found in batch")
                continue

            # Simply pass through the column data for batch-level processing
            features[col] = batch.column(col)

        return features

    @override
    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """
        Compute batch-level completeness counts for streaming aggregation.

        This counts total and non-null values per column in this batch,
        which will be aggregated across all batches for final dataset completeness.

        Args:
            features: Dictionary of column arrays from this batch

        Returns:
            Dictionary of batch-level completeness counts
        """
        batch_metrics = {}

        for col, col_array in features.items():
            total_count = len(col_array)

            # Count complete (non-null, non-NaN) samples in this batch.
            # pa.compute.is_valid() returns True for non-null values, but
            # float NaN from numpy is preserved as a valid float in Arrow
            # (not as a null), so we subtract NaN positions for float cols.
            valid_count = pa.compute.sum(pa.compute.is_valid(col_array)).as_py()
            if pa.types.is_floating(col_array.type):
                nan_count = pa.compute.sum(pa.compute.is_nan(col_array)).as_py()
                complete_count = valid_count - nan_count
            else:
                complete_count = valid_count

            # store counts for aggregation across batches
            batch_metrics[f"{col}_total_count"] = pa.array([total_count], type=pa.int64())
            batch_metrics[f"{col}_complete_count"] = pa.array([complete_count], type=pa.int64())

        return batch_metrics

    @staticmethod
    def _extract_columns_from_metrics(batch_metrics: dict[str, pa.Array]) -> list[str]:
        """Extract column names from batch metrics keys ending in _total_count.

        Args:
            batch_metrics: Dictionary of batch-level metrics.

        Returns:
            List of column names.
        """
        columns = []
        for key in batch_metrics:
            if key.endswith("_total_count"):
                columns.append(key.replace("_total_count", ""))
        return columns

    @staticmethod
    def _compute_column_completeness(col: str, batch_metrics: dict[str, pa.Array]) -> float | None:
        """Compute completeness score for a single column.

        Args:
            col: Column name.
            batch_metrics: Dictionary of batch-level metrics.

        Returns:
            Completeness score (0.0-1.0) or None if metrics are missing.
        """
        total_key = f"{col}_total_count"
        complete_key = f"{col}_complete_count"
        if total_key not in batch_metrics or complete_key not in batch_metrics:
            return None

        col_total = int(np.sum(batch_metrics[total_key].to_numpy()))
        col_complete = int(np.sum(batch_metrics[complete_key].to_numpy()))
        return col_complete / col_total if col_total > 0 else 0.0

    def _write_output_metrics(
        self,
        results: dict[str, Any],
        per_column_completeness: dict[str, float],
    ) -> None:
        """Write per-column and overall completeness metrics to results.

        Args:
            results: Output dict to populate.
            per_column_completeness: Dict of column -> completeness score.
        """
        if self.include_per_column:
            for col, score in per_column_completeness.items():
                output_key = self.output_metrics.get(f"completeness_{col}", f"completeness_{col}")
                results[output_key] = score

        if self.include_overall:
            overall = (
                sum(per_column_completeness.values()) / len(per_column_completeness) if per_column_completeness else 0.0
            )
            output_key = self.output_metrics.get("overall_completeness", "completeness_overall")
            results[output_key] = overall

    @override
    def compute(self, batch_metrics: dict[str, pa.Array] | None = None) -> dict[str, Any]:
        """
        Compute final dataset-level completeness metrics.

        This aggregates the batch-level counts to compute final completeness scores
        for each column and overall dataset completeness.

        Args:
            batch_metrics: Dictionary of batch-level metrics to aggregate

        Returns:
            Dictionary of final completeness metrics
        """
        if not batch_metrics:
            return {"_metadata": {"error": "No batch metrics provided"}}

        columns_analyzed = self._extract_columns_from_metrics(batch_metrics)
        if not columns_analyzed:
            logger.warning(f"[{self.name}] No columns found in batch metrics")
            return {"_metadata": {"error": "No columns found in batch metrics"}}

        per_column_completeness: dict[str, float] = {}
        total_samples = 0

        for col in columns_analyzed:
            score = self._compute_column_completeness(col, batch_metrics)
            if score is None:
                logger.warning(f"[{self.name}] Missing batch metrics for column '{col}'")
                continue
            per_column_completeness[col] = score
            total_key = f"{col}_total_count"
            total_samples += int(np.sum(batch_metrics[total_key].to_numpy()))

        results: dict[str, Any] = {}
        self._write_output_metrics(results, per_column_completeness)

        if self.include_metadata:
            metadata = {
                "columns_analyzed": columns_analyzed,
                "total_samples_per_column": total_samples // len(columns_analyzed) if columns_analyzed else 0,
                "per_column_scores": per_column_completeness,
                "overall_score": sum(per_column_completeness.values()) / len(per_column_completeness)
                if per_column_completeness
                else 0.0,
            }
            results["_metadata"] = json.dumps(metadata)

        return results

    @override
    def reset(self) -> None:
        """Reset processor state for new processing run.

        The completeness processor has no persistent state across runs,
        so this is a no-op. Provided for interface compliance.
        """
        # No persistent state to reset for completeness processor
