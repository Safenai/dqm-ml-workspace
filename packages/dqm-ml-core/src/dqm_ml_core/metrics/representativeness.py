"""Representativeness metric processor for evaluating distribution fit.

This module contains the RepresentativenessProcessor class that evaluates
how well a dataset represents a target statistical distribution using
various statistical tests.
"""

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats

# COMPATIBILITY : from typing import Any, override # When support of 3.10 and 3.11 will be removed
from typing_extensions import override

from dqm_ml_core.api.data_processor import DatametricProcessor

logger = logging.getLogger(__name__)


class RepresentativenessProcessor(DatametricProcessor):
    """
    Evaluates how well the dataset represents a target statistical distribution.

    This processor performs on samples discretisation statistical tests to compare the observed
    distribution of numerical columns against a theoretical target distribution
    (Normal or Uniform).

    Supported Metrics:
      - Chi-square: Goodness-of-fit test for categorical/binned data.
      - Kolmogorov-Smirnov (KS): Non-parametric test for continuous distributions (approximated via sampling).
      - Shannon Entropy: Measures the information diversity of the binned data.
      - GRTE (Geometric Representativeness Trajectory Error): Measures the exponential gap
        between observed and theoretical entropy.

    The processor uses a streaming architecture:
    - Batch level: Computes partial calculus.
    - Dataset level: Aggregates histograms and performs final statistical tests.
    """

    SUPPORTED_METRICS = {
        "chi-square",
        "grte",
        "shannon-entropy",
        "kolmogorov-smirnov",
    }
    SUPPORTED_DISTS = {"normal", "uniform"}

    # Configuration constants - can be overridden in config
    DEFAULT_ALPHA = 0.05  # Significance level for statistical tests
    DEFAULT_SHANNON_ENTROPY_THRESHOLD = 2.0  # Threshold for high/low diversity interpretation
    DEFAULT_GRTE_THRESHOLD = 0.5  # Threshold for high/low representativeness interpretation
    DEFAULT_KS_SAMPLE_SIZE = 500  # Maximum sample size for KS test
    DEFAULT_KS_MIN_SAMPLE_SIZE = 50  # Minimum sample size for KS test
    DEFAULT_KS_SAMPLE_DIVISOR = 20  # Divisor for calculating sample size per batch
    DEFAULT_EPSILON = 1e-9  # Small value to avoid division by zero
    DEFAULT_INTERPRETATION_THRESHOLDS = {
        "follows_distribution": "follows_distribution",
        "does_not_follow_distribution": "does_not_follow_distribution",
        "high_diversity": "high_diversity",
        "low_diversity": "low_diversity",
        "high_representativeness": "high_representativeness",
        "low_representativeness": "low_representativeness",
    }

    def __init__(
        self,
        name: str = "representativeness",
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the representativeness processor.

        Args:
            name: Name of the processor.
            config: Configuration dictionary containing:
                - input_columns: List of columns to analyze.
                - metrics: List of metrics to compute (default: all supported).
                - bins: Number of bins for histograms (default: 10).
                - distribution: Target distribution ("normal" or "uniform").
                - alpha: Significance level (default: 0.05).
                - distribution_params: Dictionary of params (e.g., mean, std, min, max).
        """
        super().__init__(name, config)
        self.name = name

        cfg = self.config
        self.metrics: list[str] = list(
            cfg.get(
                "metrics",
                ["chi-square", "grte", "kolmogorov-smirnov", "shannon-entropy"],
            )
        )

        self.bins: int = int(cfg.get("bins", 10))
        self.distribution: str = str(cfg.get("distribution", "normal")).lower()

        # Load configurable constants from config or use defaults
        self.alpha: float = float(cfg.get("alpha", self.DEFAULT_ALPHA))
        self.shannon_entropy_threshold: float = float(
            cfg.get(
                "shannon_entropy_threshold",
                self.DEFAULT_SHANNON_ENTROPY_THRESHOLD,
            )
        )
        self.grte_threshold: float = float(cfg.get("grte_threshold", self.DEFAULT_GRTE_THRESHOLD))
        self.ks_sample_size: int = int(cfg.get("ks_sample_size", self.DEFAULT_KS_SAMPLE_SIZE))
        self.ks_min_sample_size: int = int(cfg.get("ks_min_sample_size", self.DEFAULT_KS_MIN_SAMPLE_SIZE))
        self.ks_sample_divisor: int = int(cfg.get("ks_sample_divisor", self.DEFAULT_KS_SAMPLE_DIVISOR))
        self.epsilon: float = float(cfg.get("epsilon", self.DEFAULT_EPSILON))

        # Load interpretation thresholds from config or use defaults
        self.interpretation_thresholds: dict[str, str] = cfg.get(
            "interpretation_thresholds", self.DEFAULT_INTERPRETATION_THRESHOLDS
        )

        # Handle distribution_params properly - it can be None or a dict
        dist_params_raw = cfg.get("distribution_params")

        self.dist_params: dict[str, Any] = {}
        if dist_params_raw is not None:
            self.dist_params = dict(dist_params_raw)

        # Avoid redundant config validation (pipeline already validates)
        if not self.input_columns:
            raise ValueError(f"[{self.name}] 'input_columns' must be provided")
        if any(m not in self.SUPPORTED_METRICS for m in self.metrics):
            raise ValueError(f"[{self.name}] unsupported metric; supported: {self.SUPPORTED_METRICS}")
        if self.distribution not in self.SUPPORTED_DISTS:
            raise ValueError(f"[{self.name}] 'distribution' must be in {self.SUPPORTED_DISTS}")
        if self.bins < 2:
            raise ValueError(f"[{self.name}] 'bins' must be >= 2")
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError(f"[{self.name}] 'alpha' must be between 0 and 1")
        if self.epsilon <= 0:
            raise ValueError(f"[{self.name}] 'epsilon' must be positive")

        self._rng = np.random.default_rng()
        self._bin_edges: dict[str, np.ndarray] = {}
        self._initialized: bool = False

    @override
    def generated_metrics(self) -> list[str]:
        """
        Return the list of metric columns that will be generated.

        Returns:
            List of output metric column names
        """
        # TODO : manage output metrics names with configuration
        # for now we follow a fixed naming convention
        metrics = []
        for col in self.input_columns:
            if "chi-square" in self.metrics:
                metrics.append(f"{col}_chi-square_p_value")
                metrics.append(f"{col}_chi-square_statistic")
                metrics.append(f"{col}_chi-square_interpretation")
            if "kolmogorov-smirnov" in self.metrics:
                metrics.append(f"{col}_kolmogorov-smirnov_p_value")
                metrics.append(f"{col}_kolmogorov-smirnov_statistic")
                metrics.append(f"{col}_kolmogorov-smirnov_interpretation")
            if "shannon-entropy" in self.metrics:
                metrics.append(f"{col}_shannon-entropy_entropy")
                metrics.append(f"{col}_shannon-entropy_interpretation")
            if "grte" in self.metrics:
                metrics.append(f"{col}_grte_grte_value")
                metrics.append(f"{col}_grte_interpretation")

        return metrics

    @staticmethod
    def _convert_column_to_numeric(feature_array: pa.Array) -> pd.Series | None:
        """Convert a PyArrow column array to a numeric pandas Series with NaN handling.

        Args:
            feature_array: PyArrow array from the batch.

        Returns:
            Numeric pandas Series with NaN dropped, or None if conversion fails.
        """
        try:
            np_col = np.asarray(feature_array.to_numpy(zero_copy_only=False))
        except Exception:
            np_col = pd.Series(feature_array.to_pylist()).to_numpy(copy=True)
        numeric_values = pd.to_numeric(pd.Series(np_col), errors="coerce").dropna()
        return numeric_values if not numeric_values.empty else None

    def _compute_batch_ks_sample(self, numeric_values: pd.Series) -> np.ndarray | None:
        """Compute a random KS sample from numeric values if KS or chi-square is enabled.

        Args:
            numeric_values: Numeric pandas Series from a batch.

        Returns:
            Sampled numpy array, or None if no sampling is needed.
        """
        if "kolmogorov-smirnov" not in self.metrics and "chi-square" not in self.metrics:
            return None

        sample_per_batch = min(
            self.ks_sample_size,
            max(self.ks_min_sample_size, len(numeric_values) // self.ks_sample_divisor),
        )
        if len(numeric_values) > sample_per_batch:
            sample_indices = self._rng.choice(len(numeric_values), sample_per_batch, replace=False)
            return np.asarray(numeric_values[sample_indices])
        return np.asarray(numeric_values)

    @override
    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """
        Compute partial histogram statistics per batch for streaming aggregation.

        Args:
            features: Dictionary of column arrays from this batch.

        Returns:
            Dictionary containing:
                - {col}_count: Total valid numeric samples.
                - {col}_hist: Histogram counts.
                - {col}_ks_sample: Random subset of data for KS test.
        """
        batch_metrics = {}

        for col in self.input_columns:
            if col not in features:
                logger.warning(f"[{self.name}] column '{col}' not found in batch")
                continue

            numeric_values = self._convert_column_to_numeric(features[col])
            if numeric_values is None:
                logger.warning(f"[{self.name}] column '{col}' has no valid numeric values in this batch")
                continue

            if not self._initialized or col not in self._bin_edges:
                self._initialize_bin_edges(numeric_values.to_numpy(), col)

            edges = self._bin_edges[col]
            hist_counts = np.histogram(numeric_values, bins=edges)[0].astype(np.int64)

            batch_metrics[f"{col}_count"] = pa.array([len(numeric_values)], type=pa.int64())
            batch_metrics[f"{col}_hist"] = pa.FixedSizeListArray.from_arrays(
                hist_counts, list_size=hist_counts.shape[0]
            )

            ks_sample = self._compute_batch_ks_sample(numeric_values)
            if ks_sample is not None:
                batch_metrics[f"{col}_ks_sample"] = pa.array(ks_sample.tolist(), type=pa.float64())

        if not self._initialized and batch_metrics:
            self._initialized = True

        return batch_metrics

    def _initialize_bin_edges(self, sample_data: np.ndarray, col: str) -> None:
        """
        Initialize bin edges for a column based on sample data and target distribution.

        Args:
            sample_data: Data array used to infer parameters if not provided in config.
            col: Name of the column.
        """
        if self.distribution == "normal":
            mean = float(self.dist_params.get("mean", np.mean(sample_data)))
            std = float(self.dist_params.get("std", np.std(sample_data, ddof=0)))
            std = std if std > 0.0 else self.epsilon
            edges = self._bin_edges_normal(mean, std, self.bins, sample_data)
        else:
            min_val = float(self.dist_params.get("min", np.min(sample_data)))
            max_val = float(self.dist_params.get("max", np.max(sample_data)))
            if max_val <= min_val:
                max_val = min_val + self.epsilon
            edges = self._bin_edges_uniform(min_val, max_val, self.bins, sample_data)

        self._bin_edges[col] = edges

    def _aggregate_column_metrics(
        self, batch_metrics: dict[str, pa.Array], col: str
    ) -> tuple[int, np.ndarray, np.ndarray] | None:
        """Aggregate histogram and count for a single column across all batches.

        Args:
            batch_metrics: Dictionary of batch-level metrics.
            col: Column name.

        Returns:
            Tuple of (total_count, obs_counts, edges) or None if aggregation fails.
        """
        count_key = f"{col}_count"
        hist_key = f"{col}_hist"
        if count_key not in batch_metrics or hist_key not in batch_metrics:
            logger.warning(f"[{self.name}] no batch metrics for column '{col}'")
            return None

        hist_batch_arrays = np.asarray(batch_metrics[hist_key].to_numpy(zero_copy_only=False))
        if hist_batch_arrays.shape[0] == 0:
            logger.warning(f"[{self.name}] no histogram batch for '{col}'")
            return None

        total_count = int(np.sum(batch_metrics[count_key].to_numpy()))
        hist_arrays = hist_batch_arrays[0].copy()
        for batch_hist in hist_batch_arrays[1:]:
            hist_arrays += batch_hist
        obs_counts = hist_arrays.astype(float)

        if total_count <= 0 or obs_counts.sum() <= 0:
            logger.warning(f"[{self.name}] no valid data for column '{col}'")
            return None

        if col not in self._bin_edges:
            logger.warning(f"[{self.name}] no bin edges for column '{col}' - skipping")
            return None

        return total_count, obs_counts, self._bin_edges[col]

    def _compute_expected_counts(
        self,
        col: str,
        batch_metrics: dict[str, pa.Array],
        total_count: int,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Generate expected counts under the configured distribution.

        Args:
            col: Column name.
            batch_metrics: Batch-level metrics dict (used for KS samples to estimate params).
            total_count: Total number of samples.
            edges: Bin edges for histogram.

        Returns:
            Expected frequency counts per bin.
        """
        if self.distribution == "normal":
            mean, std = self._estimate_normal_params(col, batch_metrics)
            expected_values = self._rng.normal(mean, std, total_count)
        else:
            min_val = float(self.dist_params.get("min", edges[0]))
            max_val = float(self.dist_params.get("max", edges[-1]))
            expected_values = self._rng.uniform(min_val, max_val, total_count)
        return np.histogram(expected_values, bins=edges)[0].astype(np.float64)

    def _estimate_normal_params(self, col: str, batch_metrics: dict[str, pa.Array]) -> tuple[float, float]:
        """Estimate normal distribution parameters from config or KS samples.

        Args:
            col: Column name.
            batch_metrics: Batch-level metrics dict.

        Returns:
            Tuple of (mean, std).
        """
        sample_key = f"{col}_ks_sample"
        if sample_key in batch_metrics:
            sample_arrays = batch_metrics[sample_key].to_numpy()
            if sample_arrays.ndim > 1:
                sample_arrays = sample_arrays.flatten()
            mean = float(self.dist_params.get("mean", np.mean(sample_arrays)))
            std = float(self.dist_params.get("std", np.std(sample_arrays, ddof=0)))
        else:
            mean = float(self.dist_params.get("mean", 0.0))
            std = float(self.dist_params.get("std", 1.0))
        std = std if std > 0.0 else self.epsilon
        return mean, std

    def _compute_chi_square_metric(self, obs_counts: np.ndarray, exp_counts: np.ndarray) -> dict[str, Any]:
        """Compute chi-square goodness-of-fit test between observed and expected counts.

        Args:
            obs_counts: Observed frequency counts per bin.
            exp_counts: Expected frequency counts per bin.

        Returns:
            Dict with p_value, statistic, interpretation keys.
        """
        mask = exp_counts > 0
        if mask.sum() < 2:
            return {
                "p_value": float("nan"),
                "statistic": float("nan"),
                "interpretation": "insufficient_bins",
            }

        obs_sum = obs_counts[mask].sum()
        exp_sum = exp_counts[mask].sum()
        if exp_sum <= 0:
            return {
                "p_value": float("nan"),
                "statistic": float("nan"),
                "interpretation": "no_expected_counts",
            }

        exp_counts_normalized = exp_counts[mask] * (obs_sum / exp_sum)
        try:
            chi = stats.chisquare(f_obs=obs_counts[mask], f_exp=exp_counts_normalized)
            return {
                "p_value": float(chi.pvalue),
                "statistic": float(chi.statistic),
                "interpretation": self.interpretation_thresholds.get(
                    "follows_distribution" if chi.pvalue >= self.alpha else "does_not_follow_distribution",
                    "follows_distribution",
                ),
            }
        except ValueError as e:
            return {
                "p_value": float("nan"),
                "statistic": float("nan"),
                "interpretation": f"chi_square_failed: {e!s}",
                "note": "using observed counts only due to statistical constraints",
            }

    def _compute_ks_metric(self, col: str, batch_metrics: dict[str, pa.Array]) -> dict[str, Any]:
        """Compute Kolmogorov-Smirnov test using sampled data.

        Args:
            col: Column name.
            batch_metrics: Batch-level metrics dict containing KS samples.

        Returns:
            Dict with p_value, statistic, interpretation keys.
        """
        sample_key = f"{col}_ks_sample"
        if sample_key not in batch_metrics:
            return {
                "p_value": float("nan"),
                "statistic": float("nan"),
                "interpretation": "no_sample_data_found",
            }

        sample_arrays = batch_metrics[sample_key].to_numpy()
        ks_samples = sample_arrays if sample_arrays.ndim == 1 else sample_arrays.flatten()

        if len(ks_samples) == 0:
            return {
                "p_value": float("nan"),
                "statistic": float("nan"),
                "interpretation": "no_samples_available",
            }

        if self.distribution == "normal":
            mean, std = self._estimate_normal_params(col, batch_metrics)
            ks = stats.kstest(ks_samples, stats.norm.cdf, args=(mean, std))
        else:
            min_val = float(self.dist_params.get("min", np.min(ks_samples)))
            max_val = float(self.dist_params.get("max", np.max(ks_samples)))
            if max_val <= min_val:
                max_val = min_val + self.epsilon
            ks = stats.kstest(ks_samples, stats.uniform.cdf, args=(min_val, max_val - min_val))

        return {
            "p_value": float(ks.pvalue),
            "statistic": float(ks.statistic),
            "interpretation": self.interpretation_thresholds.get(
                "follows_distribution" if ks.pvalue >= self.alpha else "does_not_follow_distribution",
                "follows_distribution",
            ),
            "sample_size": len(ks_samples),
            "note": "approximated_from_random_samples",
        }

    def _compute_shannon_entropy_metric(self, exp_counts: np.ndarray) -> dict[str, Any]:
        """Compute Shannon entropy from expected frequency counts.

        Args:
            exp_counts: Expected frequency counts per bin.

        Returns:
            Dict with entropy and interpretation.
        """
        p_exp = exp_counts / exp_counts.sum()
        h_exp = float(stats.entropy(p_exp))
        is_high = h_exp > self.shannon_entropy_threshold
        return {
            "entropy": h_exp,
            "interpretation": self.interpretation_thresholds.get(
                "high_diversity" if is_high else "low_diversity",
                "high_diversity",
            ),
        }

    def _compute_grte_metric(self, obs_counts: np.ndarray, exp_counts: np.ndarray) -> dict[str, Any]:
        """Compute GRTE (exponential gap between observed and theoretical entropy).

        Args:
            obs_counts: Observed frequency counts per bin.
            exp_counts: Expected frequency counts per bin.

        Returns:
            Dict with grte_value and interpretation.
        """
        p_obs = obs_counts / obs_counts.sum()
        p_exp = exp_counts / exp_counts.sum()
        h_obs = float(stats.entropy(p_obs))
        h_exp = float(stats.entropy(p_exp))
        grte = float(np.exp(-2.0 * abs(h_exp - h_obs)))
        is_high = grte > self.grte_threshold
        return {
            "grte_value": grte,
            "interpretation": self.interpretation_thresholds.get(
                "high_representativeness" if is_high else "low_representativeness",
                "high_representativeness",
            ),
        }

    @staticmethod
    def _build_compute_metadata(
        total_samples: int,
        batch_metrics: dict[str, pa.Array],
        input_columns: list[str],
        distribution: str,
        metrics: list[str],
        bins: int,
    ) -> str:
        """Build metadata JSON string for compute results.

        Args:
            total_samples: Total number of samples processed.
            batch_metrics: Batch-level metrics dict.
            input_columns: Input column names.
            distribution: Target distribution name.
            metrics: List of computed metric names.
            bins: Number of histogram bins.

        Returns:
            JSON-encoded metadata string.
        """
        return json.dumps(
            {
                "bins": bins,
                "distribution": distribution,
                "metrics_computed": metrics,
                "total_samples": total_samples,
                "columns_analyzed": [c for c in input_columns if f"{c}_count" in batch_metrics],
                "ks_sampling_enabled": "kolmogorov-smirnov" in metrics,
                "note": "KS test uses random sampling approximation for scalability",
            }
        )

    @staticmethod
    def _flatten_col_results(col_res: dict[str, Any], col: str, results: dict[str, Any]) -> None:
        """Flatten nested metric dicts into flat output keys.

        Args:
            col_res: Per-column metric results (potentially nested).
            col: Column name.
            results: Output dict to populate.
        """
        for key, value in col_res.items():
            if isinstance(value, dict):
                for prop, content in value.items():
                    results[f"{key}_{col}_{prop}"] = content
            else:
                results[f"{key}_{col}"] = value

    def _compute_column_results(
        self,
        col: str,
        batch_metrics: dict[str, pa.Array],
        results: dict[str, Any],
    ) -> int:
        """Compute all metrics for a single column and write them to results.

        Args:
            col: Column name.
            batch_metrics: Batch-level metrics dict.
            results: Output dict to populate (mutated in place).

        Returns:
            Number of samples processed (0 if column has no valid data).
        """
        agg = self._aggregate_column_metrics(batch_metrics, col)
        if agg is None:
            return 0

        total_count, obs_counts, edges = agg
        exp_counts = self._compute_expected_counts(col, batch_metrics, total_count, edges)

        col_res: dict[str, Any] = {}

        if "chi-square" in self.metrics:
            col_res["chi-square"] = self._compute_chi_square_metric(obs_counts, exp_counts)

        if "kolmogorov-smirnov" in self.metrics:
            col_res["kolmogorov-smirnov"] = self._compute_ks_metric(col, batch_metrics)

        if "shannon-entropy" in self.metrics:
            col_res["shannon-entropy"] = self._compute_shannon_entropy_metric(exp_counts)

        if "grte" in self.metrics:
            col_res["grte"] = self._compute_grte_metric(obs_counts, exp_counts)

        if col_res:
            self._flatten_col_results(col_res, col, results)

        return total_count

    @override
    def compute(self, batch_metrics: dict[str, pa.Array] | None = None) -> dict[str, Any]:
        """
        Compute final dataset-level metrics by aggregating batch histograms.

        Args:
            batch_metrics: Dictionary of batch-level metrics collected during processing.

        Returns:
            Dictionary containing final scores and interpretations.
        """
        if not batch_metrics:
            return {"_metadata": {"error": "No batch metrics provided"}}

        results: dict[str, Any] = {}
        total_samples = 0

        for col in self.input_columns:
            total_samples += self._compute_column_results(col, batch_metrics, results)

        results["_metadata"] = self._build_compute_metadata(
            total_samples,
            batch_metrics,
            self.input_columns,
            self.distribution,
            self.metrics,
            self.bins,
        )
        return results

    def reset(self) -> None:
        """Reset processor state for new processing run."""
        self._bin_edges = {}
        self._initialized = False

    # utils methods for bin edge calculation

    def _bin_edges_normal(self, mean: float, std: float, bins: int, data: np.ndarray) -> np.ndarray:
        """Calculate bin edges using the PPF of a Normal distribution.

        This ensures bins represent equal probability mass under the
        theoretical distribution. The first and last bins are extended
        to -inf and +inf respectively.
        """
        # logic from dqm-ml v1: use stats.norm.ppf with linspace(1/bins, 1, bins)
        bin_edges_list = [stats.norm.ppf(i / bins, mean, std) for i in range(1, bins)]
        return np.array([-np.inf] + bin_edges_list + [np.inf])

    def _bin_edges_uniform(self, param_min: float, param_max: float, bins: int, data: np.ndarray) -> np.ndarray:
        """
        Calculate linearly spaced bin edges for a Uniform distribution.

        The range is determined by the minimum/maximum of both the configured
        parameters and the actual observed data.
        """
        low_edge = min(param_min, float(np.min(data)))
        high_edge = max(param_max, float(np.max(data)))
        # Handle degenerate case where all data is identical
        if high_edge <= low_edge:
            high_edge = low_edge + self.epsilon
        return np.linspace(low_edge, high_edge, bins + 1)
