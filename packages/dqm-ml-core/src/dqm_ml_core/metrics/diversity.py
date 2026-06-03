"""Diversity metric processor for evaluating categorical data diversity.

This module contains the DiversityProcessor class that computes
diversity indices (Simpson, Gini-Simpson, Shannon Entropy, Richness)
for categorical or discretized data columns.
"""

from collections import Counter
import logging
from typing import Any

import numpy as np
import pyarrow as pa
from typing_extensions import override

from dqm_ml_core.api.data_processor import DatametricProcessor

logger = logging.getLogger(__name__)


class DiversityProcessor(DatametricProcessor):
    """Computes diversity indices for categorical data columns.

    Simpson Index (1 - Σn(n-1) / N(N-1)):
        Unbiased estimator — probability two random samples belong to
        different categories. Ranges [0, 1]; high = diverse.

    Gini-Simpson Index (1 - Σp²):
        Gini impurity — probability of incorrect classification.
        Ranges [0, 1]; high = diverse.

    Shannon Entropy (-Σp·log(p)):
        Information content / uncertainty. Lower bound 0; no upper bound
        (grows with category count and evenness).

    Richness:
        Simple count of unique categories.

    The processor uses a streaming architecture:
    - Batch level: Computes value counts per column.
    - Dataset level: Merges all batch counts and computes final indices.
    """

    SUPPORTED_METRICS = {"simpson", "gini", "shannon", "richness"}

    def __init__(self, name: str = "diversity", config: dict[str, Any] | None = None) -> None:
        """Initialize the diversity processor.

        Args:
            name: Name of the processor.
            config: Configuration dictionary containing:
                - input_columns: List of columns to analyze.
                - metrics: List of metrics to compute (default: all supported).
        """
        super().__init__(name, config)

        cfg = config or {}
        self.metrics: list[str] = list(cfg.get("metrics", ["simpson", "gini", "shannon", "richness"]))

        if not self.input_columns:
            raise ValueError(f"[{self.name}] 'input_columns' must be provided")
        if any(m not in self.SUPPORTED_METRICS for m in self.metrics):
            raise ValueError(f"[{self.name}] unsupported metric; supported: {self.SUPPORTED_METRICS}")

    @override
    def generated_metrics(self) -> list[str]:
        """Return the list of metric columns that will be generated.

        Returns:
            List of output metric column names.
        """
        metrics = []
        for col in self.input_columns:
            if "simpson" in self.metrics:
                metrics.append(f"{col}_simpson")
            if "gini" in self.metrics:
                metrics.append(f"{col}_gini")
            if "shannon" in self.metrics:
                metrics.append(f"{col}_shannon")
            if "richness" in self.metrics:
                metrics.append(f"{col}_richness")
        return metrics

    @override
    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Compute per-batch value counts for streaming aggregation.

        Args:
            features: Dictionary of column arrays from this batch.

        Returns:
            Dictionary of batch-level value-count pairs.
        """
        batch_metrics: dict[str, pa.Array] = {}

        for col in self.input_columns:
            if col not in features:
                logger.warning("[%s] column '%s' not found in batch", self.name, col)
                continue

            col_array = features[col]
            vc = pa.compute.value_counts(col_array)
            if len(vc) == 0:
                continue

            values = vc.field("values")
            counts = vc.field("counts")

            values_str = pa.compute.cast(values, pa.string())
            batch_metrics[f"{col}_values"] = values_str
            batch_metrics[f"{col}_counts"] = counts

        return batch_metrics

    @override
    def compute(self, batch_metrics: dict[str, pa.Array] | None = None) -> dict[str, Any]:
        """Compute final dataset-level diversity indices.

        Merges all batch-level value counts and computes diversity
        indices for each column.

        Args:
            batch_metrics: Dictionary of batch-level value-count arrays.

        Returns:
            Dictionary containing final diversity scores.
        """
        if not batch_metrics:
            return {"_metadata": {"error": "No batch metrics provided"}}

        results: dict[str, Any] = {}
        for col in self.input_columns:
            col_res = self._compute_column_diversity(col, batch_metrics)
            results.update(col_res)
        return results

    def _compute_column_diversity(self, col: str, batch_metrics: dict[str, pa.Array]) -> dict[str, Any]:
        """Compute diversity indices for a single column.

        Args:
            col: Column name to compute diversity for.
            batch_metrics: Dictionary of batch-level value-count arrays.

        Returns:
            Dictionary mapping column-specific metric names to values.
        """
        values_key = f"{col}_values"
        counts_key = f"{col}_counts"

        if values_key not in batch_metrics or counts_key not in batch_metrics:
            logger.warning("[%s] no batch metrics for column '%s'", self.name, col)
            return {}

        values_arr = batch_metrics[values_key]
        counts_arr = batch_metrics[counts_key]

        if len(values_arr) == 0:
            return {}

        counter: Counter[str] = Counter()
        total = 0
        for i in range(len(values_arr)):
            val = values_arr[i].as_py()
            cnt = int(counts_arr[i])
            counter[val] += cnt
            total += cnt

        if total == 0:
            return {}

        col_res: dict[str, Any] = {}

        if "richness" in self.metrics:
            col_res["richness"] = len(counter)

        if "shannon" in self.metrics:
            probs = np.array(list(counter.values()), dtype=np.float64) / total
            probs = probs[probs > 0]
            entropy = float(-(probs * np.log(probs)).sum())
            col_res["shannon"] = entropy

        if "gini" in self.metrics:
            probs = np.array(list(counter.values()), dtype=np.float64) / total
            gini = float(1.0 - (probs**2).sum())
            col_res["gini"] = gini

        if "simpson" in self.metrics:
            counts_np = np.array(list(counter.values()), dtype=np.float64)
            simpson = float(1.0 - (counts_np * (counts_np - 1)).sum() / (total * (total - 1))) if total > 1 else 0.0
            col_res["simpson"] = simpson

        return {f"{col}_{metric_name}": value for metric_name, value in col_res.items()}

    def reset(self) -> None:
        """Reset processor state for new processing run."""
