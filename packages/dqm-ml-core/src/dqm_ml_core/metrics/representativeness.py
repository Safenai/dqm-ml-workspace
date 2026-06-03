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

            feature_array = features[col]
            # convert to numeric, handle mixed types and NaN
            try:
                np_col = np.asarray(feature_array.to_numpy(zero_copy_only=False))
            except Exception:
                # Fallback for mixed-type columns where zero_copy_only=False still fails
                np_col = pd.Series(feature_array.to_pylist()).to_numpy(copy=True)

            numeric_values = pd.to_numeric(pd.Series(np_col), errors="coerce").dropna()

            if numeric_values.empty:
                logger.warning(f"[{self.name}] column '{col}' has no valid numeric values in this batch")
                continue

            if not self._initialized or col not in self._bin_edges:
                self._initialize_bin_edges(numeric_values.to_numpy(), col)

            edges = self._bin_edges[col]

            logger.debug(f"[{self.name}] edges shape: {edges.shape}, values shape: {numeric_values.shape}")

            hist_counts = np.histogram(numeric_values, bins=edges)[0].astype(np.int64)

            logger.debug(f"[{self.name}] hist_counts shape: {hist_counts.shape}, expected: {self.bins}")

            # store as Arrow arrays for aggregation
            batch_metrics[f"{col}_count"] = pa.array([len(numeric_values)], type=pa.int64())
            batch_metrics[f"{col}_hist"] = pa.FixedSizeListArray.from_arrays(
                hist_counts, list_size=hist_counts.shape[0]
            )

            # sampling for KS test approximation
            # NOTE: uses global numpy RNG; results may vary between runs
            if "kolmogorov-smirnov" in self.metrics or "chi-square" in self.metrics:
                # Clamp per-batch KS sample to [ks_min_sample_size, ks_sample_size],
                # taking a fraction (1/ks_sample_divisor) of the batch
                sample_per_batch = min(
                    self.ks_sample_size,
                    max(
                        self.ks_min_sample_size,
                        len(numeric_values) // self.ks_sample_divisor,
                    ),
                )
                if len(numeric_values) > sample_per_batch:
                    sample_indices = self._rng.choice(len(numeric_values), sample_per_batch, replace=False)
                    sample = numeric_values[sample_indices]
                else:
                    sample = numeric_values

                batch_metrics[f"{col}_ks_sample"] = pa.array(sample.tolist(), type=pa.float64())

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
            count_key = f"{col}_count"
            hist_key = f"{col}_hist"

            if count_key not in batch_metrics or hist_key not in batch_metrics:
                logger.warning(f"[{self.name}] no batch metrics for column '{col}'")
                continue

            # TODO : maybe we need a try ? as in batch or not as na already removed
            # N/A handling shall be documented, and logs added
            hist_batch_arrays = np.asarray(batch_metrics[hist_key].to_numpy(zero_copy_only=False))
            if hist_batch_arrays.shape[0] == 0:
                logger.warning(f"[{self.name}] no histograme batch for '{col}'")
                continue

            # aggregate counts and histograms across all batches
            total_count = int(np.sum(batch_metrics[count_key].to_numpy()))

            # Accumulate histogram counts across all batches
            # hist_batch_arrays is an object array (from FixedSizeListArray.to_numpy)
            # where each element is a 1D histogram of shape (bins,)
            hist_arrays = hist_batch_arrays[0].copy()
            for batch_hist in hist_batch_arrays[1:]:
                hist_arrays += batch_hist
            obs_counts = hist_arrays.astype(float)

            if total_count <= 0 or obs_counts.sum() <= 0:
                logger.warning(f"[{self.name}] no valid data for column '{col}'")
                continue

            total_samples += total_count

            #  distribution parameters and bin edges
            if col not in self._bin_edges:
                logger.warning(f"[{self.name}] no bin edges for column '{col}' - skipping")
                continue

            edges = self._bin_edges[col]

            # Theoretical expected counts — aligned with official DQM-ML v1
            if self.distribution == "normal":
                logger.debug("Generate normal distribution")
                # Use the SAME parameters as those used to generate the bin edges

                sample_key = f"{col}_ks_sample"
                if sample_key in batch_metrics:
                    logger.debug("Use sampled mean and std")
                    sample_arrays = batch_metrics[sample_key].to_numpy()
                    if sample_arrays.ndim > 1:
                        sample_arrays = sample_arrays.flatten()
                    mean = float(self.dist_params.get("mean", np.mean(sample_arrays)))
                    std = float(self.dist_params.get("std", np.std(sample_arrays, ddof=0)))
                    std = std if std > 0.0 else self.epsilon
                else:
                    logger.debug("Fallback: use default or configured mean and std")
                    mean = float(self.dist_params.get("mean", 0.0))
                    std = float(self.dist_params.get("std", 1.0))
                logger.debug(f"mean={mean}")
                logger.debug(f"std={std}")
                # Generate random values and count frequencies (as in official DQM-ML v1)
                expected_values = self._rng.normal(mean, std, total_count)
                exp_counts = np.histogram(expected_values, bins=edges)[0].astype(np.float64)
            else:  # uniform
                logger.debug("Generate uniform distribution")
                min_val = float(self.dist_params.get("min", edges[0]))
                max_val = float(self.dist_params.get("max", edges[-1]))
                logger.debug(f"min={min_val}")
                logger.debug(f"max={max_val}")
                # Generate random values and count frequencies (as in official DQM-ML v1)
                expected_values = self._rng.uniform(min_val, max_val, total_count)
                exp_counts = np.histogram(expected_values, bins=edges)[0].astype(np.float64)

            col_res: dict[str, Any] = {}

            # chi-square: here we compute the chi-square with a alpha value of 0.05
            if "chi-square" in self.metrics:
                # Ensure observed and expected counts have the same sum

                # Exclude bins with zero expected count to avoid division issues in chi-square
                mask = exp_counts > 0
                if mask.sum() >= 2:
                    logger.debug("Normalize expected counts to match observed sum")
                    obs_sum = obs_counts[mask].sum()
                    exp_sum = exp_counts[mask].sum()

                    if exp_sum > 0:
                        # Normalize expected counts to match observed total for fair chi-square comparison
                        exp_counts_normalized = exp_counts[mask] * (obs_sum / exp_sum)

                        logger.debug("Expected frequencies:")
                        logger.debug(exp_counts_normalized)
                        logger.debug("Observed frequencies: ")
                        logger.debug(obs_counts[mask])

                        try:
                            chi = stats.chisquare(
                                f_obs=obs_counts[mask],
                                f_exp=exp_counts_normalized,
                            )
                            logger.debug(f"Chi P value: {chi.pvalue}")
                            col_res["chi-square"] = {
                                "p_value": float(chi.pvalue),
                                "statistic": float(chi.statistic),
                                "interpretation": self.interpretation_thresholds.get(
                                    "follows_distribution"
                                    if chi.pvalue >= self.alpha
                                    else "does_not_follow_distribution",
                                    "follows_distribution",
                                ),
                            }
                        except ValueError as e:
                            # Fallback: use only observed counts if chi-square fails
                            col_res["chi-square"] = {
                                "p_value": float("nan"),
                                "statistic": float("nan"),
                                "interpretation": f"chi_square_failed: {e!s}",
                                "note": "using observed counts only due to statistical constraints",
                            }
                    else:
                        col_res["chi-square"] = {
                            "p_value": float("nan"),
                            "statistic": float("nan"),
                            "interpretation": "no_expected_counts",
                        }
                else:
                    col_res["chi-square"] = {
                        "p_value": float("nan"),
                        "statistic": float("nan"),
                        "interpretation": "insufficient_bins",
                    }

            # Kolmogorov-Smirnov test using sampled data
            if "kolmogorov-smirnov" in self.metrics:
                sample_key = f"{col}_ks_sample"
                if sample_key in batch_metrics:
                    sample_arrays = batch_metrics[sample_key].to_numpy()
                    ks_samples = sample_arrays if sample_arrays.ndim == 1 else sample_arrays.flatten()

                    if len(ks_samples) > 0:
                        # Perform KS test on aggregated samples
                        # Recompute from KS samples (may differ from histogram params due to sampling variance)
                        if self.distribution == "normal":
                            mean = float(self.dist_params.get("mean", np.mean(ks_samples)))
                            std = float(self.dist_params.get("std", np.std(ks_samples, ddof=0)))
                            std = std if std > 0.0 else self.epsilon
                            ks = stats.kstest(ks_samples, stats.norm.cdf, args=(mean, std))
                        else:  # uniform
                            min_val = float(self.dist_params.get("min", np.min(ks_samples)))
                            max_val = float(self.dist_params.get("max", np.max(ks_samples)))
                            if max_val <= min_val:
                                max_val = min_val + self.epsilon
                            ks = stats.kstest(
                                ks_samples,
                                stats.uniform.cdf,
                                args=(min_val, max_val - min_val),
                            )

                        col_res["kolmogorov-smirnov"] = {
                            "p_value": float(ks.pvalue),
                            "statistic": float(ks.statistic),
                            "interpretation": self.interpretation_thresholds.get(
                                "follows_distribution" if ks.pvalue >= self.alpha else "does_not_follow_distribution",
                                "follows_distribution",
                            ),
                            "sample_size": len(ks_samples),
                            "note": "approximated_from_random_samples",
                        }
                    else:
                        col_res["kolmogorov-smirnov"] = {
                            "p_value": float("nan"),
                            "statistic": float("nan"),
                            "interpretation": "no_samples_available",
                        }
                else:
                    col_res["kolmogorov-smirnov"] = {
                        "p_value": float("nan"),
                        "statistic": float("nan"),
                        "interpretation": "no_sample_data_found",
                    }

            # Shannon entropy - aligned on dqm-ml v1 (using theoretical frequencies)
            if "shannon-entropy" in self.metrics:
                # Use theoretical frequencies
                p_exp = exp_counts / exp_counts.sum()
                h_exp = float(stats.entropy(p_exp))
                col_res["shannon-entropy"] = {
                    "entropy": h_exp,
                    "interpretation": self.interpretation_thresholds.get(
                        "high_diversity" if h_exp > self.shannon_entropy_threshold else "low_diversity",
                        "high_diversity",
                    ),
                }

            # GRTE (gap between observed and theoretical entropies) - aligned on dqm-ml v1
            if "grte" in self.metrics:
                # Use observed and theoretical frequencies
                p_obs = obs_counts / obs_counts.sum()
                p_exp = exp_counts / exp_counts.sum()
                h_obs = float(stats.entropy(p_obs))
                h_exp = float(stats.entropy(p_exp))
                # GRTE = exp(-2 * |Δentropy|), maps to [0,1] where 1 = perfect match
                grte = float(np.exp(-2.0 * abs(h_exp - h_obs)))
                col_res["grte"] = {
                    "grte_value": grte,
                    "interpretation": self.interpretation_thresholds.get(
                        "high_representativeness" if grte > self.grte_threshold else "low_representativeness",
                        "high_representativeness",
                    ),
                }

            # Flatten nested metric dicts into flat output keys
            # TODO: refactor for efficiency (avoid nested loops)
            if col_res:
                for key, value in col_res.items():
                    if isinstance(value, dict):
                        for prop, content in value.items():
                            results[key + "_" + col + "_" + prop] = content
                    else:
                        results[key + "_" + col] = value
        # TODO make optional export of metadata
        meta_data = {
            "bins": self.bins,
            "distribution": self.distribution,
            "metrics_computed": self.metrics,
            "total_samples": total_samples,
            "columns_analyzed": [c for c in self.input_columns if f"{c}_count" in batch_metrics],
            "ks_sampling_enabled": "kolmogorov-smirnov" in self.metrics,
            "note": "KS test uses random sampling approximation for scalability",
        }

        results["_metadata"] = json.dumps(meta_data)
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
