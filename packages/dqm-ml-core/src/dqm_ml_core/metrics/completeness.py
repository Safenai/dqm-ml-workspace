import json
import logging
from typing import Any, override

import numpy as np
import pyarrow as pa

from dqm_ml_core.api.data_processor import DatametricProcessor

logger = logging.getLogger(__name__)


class CompletenessProcessor(DatametricProcessor):
    """
    Data completeness processor that evaluates the completeness of tabular data.

    This processor calculates completeness scores (ratio of non-null to total values)
    for specified columns and provides overall dataset completeness metrics.

    The processor operates at multiple levels:
    - Batch level: Aggregated counts for streaming processing
    - Dataset level: Final completeness scores and statistics
    """

    def __init__(self, name: str = "completeness", config: dict[str, Any] | None = None) -> None:
        """
        Initialize the completeness processor.

        Args:
            name: Name of the processor
            config: Configuration dictionary containing:
                - input_columns: List of columns to analyze for completeness
                - output_metrics: Dictionary mapping metric names to output column names
                - include_per_column: Whether to include per-column completeness scores
                - include_overall: Whether to include overall completeness score
        """
        super().__init__(name, config)

        cfg = self.config or {}

        # Configuration for what metrics to compute - pour chaque colonne et pour l'ensemble des colonnes
        self.include_per_column: bool = bool(cfg.get("include_per_column", True))
        self.include_overall: bool = bool(cfg.get("include_overall", True))
        self.include_metadata: bool = bool(cfg.get("include_metadata", False))

        # Output column mappings
        self.output_metrics = cfg.get("output_metrics", {})

        # Validation
        if not self.include_per_column and not self.include_overall:
            raise ValueError(f"[{self.name}] At least one of 'include_per_column' or 'include_overall' must be True")

    @override
    def generated_metrics(self) -> list[str]:
        """
        Return the list of metric columns that will be generated.

        Returns:
            List of output metric column names
        """
        # TODO : manage output metrics names
        metrics = []

        if self.include_overall:
            overall_key = self.output_metrics.get("overall_completeness", "completeness_overall")
            metrics.append(overall_key)

        if self.include_per_column:
            for col in self.input_columns:
                col_key = self.output_metrics.get(f"completeness_{col}", f"completeness_{col}")
                metrics.append(col_key)

        return metrics

    @override
    def compute_features(
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
            # Count total samples in this batch
            total_count = len(col_array)

            # Count non-null (complete) samples in this batch
            # pa.compute.is_valid() returns True for non-null values
            is_valid_mask = pa.compute.is_valid(col_array)
            complete_count = pa.compute.sum(is_valid_mask).as_py()

            # store counts for aggregation across batches
            batch_metrics[f"{col}_total_count"] = pa.array([total_count], type=pa.int64())
            batch_metrics[f"{col}_complete_count"] = pa.array([complete_count], type=pa.int64())

        return batch_metrics

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

        results: dict[str, Any] = {}

        # Determine columns from batch metrics
        columns_analyzed = []
        for key in batch_metrics:
            if key.endswith("_total_count"):
                col = key.replace("_total_count", "")
                columns_analyzed.append(col)

        if not columns_analyzed:
            logger.warning(f"[{self.name}] No columns found in batch metrics")
            return {"_metadata": {"error": "No columns found in batch metrics"}}

        per_column_completeness = {}
        total_samples = 0
        total_complete = 0

        # Calculate completeness for each column
        for col in columns_analyzed:
            total_key = f"{col}_total_count"
            complete_key = f"{col}_complete_count"

            if total_key not in batch_metrics or complete_key not in batch_metrics:
                logger.warning(f"[{self.name}] Missing batch metrics for column '{col}'")
                continue

            # agg counts across all batches
            col_total = int(np.sum(batch_metrics[total_key].to_numpy()))
            col_complete = int(np.sum(batch_metrics[complete_key].to_numpy()))

            # Calculate completeness score for this column
            completeness_score = col_complete / col_total if col_total > 0 else 0.0

            per_column_completeness[col] = completeness_score

            # Add to overall totals
            total_samples += col_total
            total_complete += col_complete

        # Generate output metrics based on configuration
        if self.include_per_column:
            for col, score in per_column_completeness.items():
                output_key = self.output_metrics.get(f"completeness_{col}", f"completeness_{col}")
                results[output_key] = score

        if self.include_overall:
            # Calculate overall completeness as average of column completeness scores
            if per_column_completeness:
                overall_completeness = sum(per_column_completeness.values()) / len(per_column_completeness)
            else:
                overall_completeness = 0.0

            output_key = self.output_metrics.get("overall_completeness", "completeness_overall")
            results[output_key] = overall_completeness

        # Add metadata
        metadata = {
            "columns_analyzed": columns_analyzed,
            "total_samples_per_column": total_samples // len(columns_analyzed) if columns_analyzed else 0,
            "per_column_scores": per_column_completeness,
            "overall_score": sum(per_column_completeness.values()) / len(per_column_completeness)
            if per_column_completeness
            else 0.0,
        }

        if self.include_metadata:
            results["_metadata"] = json.dumps(metadata)

        return results

    def reset(self) -> None:
        """Reset processor state for new processing run."""
        # No persistent state to reset for completeness processor
