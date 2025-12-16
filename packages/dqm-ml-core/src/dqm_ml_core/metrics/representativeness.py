import logging
from typing import Any, override

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats

from dqm_ml_core.api.data_processor import DatametricProcessor

logger = logging.getLogger(__name__)


class RepresentativenessProcessor(DatametricProcessor):
    """
    TODO: this metric is compute for juste one colomn : only performs a one-sample test —
          it compares your data to a target distribution (normal or uniform).
    Dataset-level representativeness metrics:
      - Chi-square
      - Kolmogorov-Smirnov
      - Shannon entropy
      - GRTE

    """

    SUPPORTED_METRICS = {"chi-square", "grte", "shannon-entropy", "kolmogorov-smirnov"}
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

    def __init__(self, name: str = "representativeness", config: dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.name = name

        cfg = self.config
        self.metrics: list[str] = list(
            cfg.get("metrics", ["chi-square", "grte", "kolmogorov-smirnov", "shannon-entropy"])
        )
        self.input_columns: list[str] = list(cfg.get("input_columns", []))

        self.bins: int = int(cfg.get("bins", 10))
        self.distribution: str = str(cfg.get("distribution", "normal")).lower()

        # Load configurable constants from config or use defaults
        self.alpha: float = float(cfg.get("alpha", self.DEFAULT_ALPHA))
        self.shannon_entropy_threshold: float = float(
            cfg.get("shannon_entropy_threshold", self.DEFAULT_SHANNON_ENTROPY_THRESHOLD)
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

        # check config: avoid redondancy checks with pipeline (see datasetpipeline )
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

        self._bin_edges: dict[str, np.ndarray] = {}
        self._initialized: bool = False

    @override
    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Compute partial histogram statistics per batch for streaming aggregation."""
        batch_metrics = {}

        for col in self.input_columns:
            if col not in features:
                logger.warning(f"[{self.name}] column '{col}' not found in batch")
                continue

            arr = features[col]
            # convert to numeric, handle mixed types and NaN, we also can add another transformation
            try:
                np_col = np.asarray(arr.to_numpy(zero_copy_only=False))
            except Exception:
                np_col = pd.Series(arr.to_pylist()).to_numpy(copy=True)

            values = pd.to_numeric(pd.Series(np_col), errors="coerce").dropna()

            if values.empty or len(values) == 0:
                logger.warning(f"[{self.name}] column '{col}' empty numeric values")
                continue

            if not self._initialized:
                self._initialize_bin_edges(values, col)

            if col not in self._bin_edges:
                self._initialize_bin_edges(values, col)

            edges = self._bin_edges[col]

            # Ddebeug
            logger.debug(f"[{self.name}] edges shape: {edges.shape}, values shape: {values.shape}")

            hist_counts = np.histogram(values, bins=edges)[0].astype(np.int64)

            # debeug: vérifier l'histogramme
            logger.debug(f"[{self.name}] hist_counts shape: {hist_counts.shape}, expected: {self.bins}")

            # store as Arrow arrays for aggregation
            batch_metrics[f"{col}_count"] = pa.array([len(values)], type=pa.int64())
            # batch_metrics[f"{col}_hist"] = pa.array(hist_counts.tolist(), type=pa.int64())
            batch_metrics[f"{col}_hist"] = pa.FixedSizeListArray.from_arrays(
                hist_counts, list_size=hist_counts.shape[0]
            )

            # TODO  create hist as fixed array
            # fs_arr =

            # sampling for KS test approximation
            # TODO: KS need to have all the data un memory to compute
            if "kolmogorov-smirnov" in self.metrics or "chi-square" in self.metrics:
                sample_per_batch = min(
                    self.ks_sample_size, max(self.ks_min_sample_size, len(values) // self.ks_sample_divisor)
                )
                if len(values) > sample_per_batch:
                    # Random sampling without replacement
                    sample_indices = np.random.choice(len(values), sample_per_batch, replace=False)
                    sample = values[sample_indices]
                else:
                    sample = values

                batch_metrics[f"{col}_ks_sample"] = pa.array(sample.tolist(), type=pa.float64())

        if not self._initialized and batch_metrics:
            self._initialized = True

        return batch_metrics

    def _initialize_bin_edges(self, sample_data: np.ndarray, col: str) -> None:
        """Initialize bin edges for a column based on sample data and distribution."""
        if self.distribution == "normal":
            mean = float(self.dist_params.get("mean", np.mean(sample_data)))
            std = float(self.dist_params.get("std", np.std(sample_data, ddof=0)))
            std = std if std > 0.0 else self.epsilon
            edges = self._bin_edges_normal(mean, std, self.bins, sample_data)
        else:
            mn = float(self.dist_params.get("min", np.min(sample_data)))
            mx = float(self.dist_params.get("max", np.max(sample_data)))
            if mx <= mn:
                mx = mn + self.epsilon
            edges = self._bin_edges_uniform(mn, mx, self.bins, sample_data)

        self._bin_edges[col] = edges

    @override
    def compute(self, batch_metrics: dict[str, pa.Array] | None = None) -> dict[str, Any]:
        """Compute final dataset-level metrics by aggregating batch histograms."""
        if not batch_metrics:
            return {"_metadata": {"error": "No batch metrics provided"}}

        out: dict[str, Any] = {}
        total_samples = 0

        for col in self.input_columns:
            count_key = f"{col}_count"
            hist_key = f"{col}_hist"

            if count_key not in batch_metrics or hist_key not in batch_metrics:
                logger.warning(f"[{self.name}] no batch metrics for column '{col}'")
                continue

            # TODO : maybe we need a try ? as in batch or not as na already removed
            hist_batch_arrays = np.asarray(batch_metrics[hist_key].to_numpy(zero_copy_only=False))
            if hist_batch_arrays.shape[0] == 0:
                logger.warning(f"[{self.name}] no histograme batch for '{col}'")
                continue

            # aggregate counts and histograms across all batches
            total_count = int(np.sum(batch_metrics[count_key].to_numpy()))

            hist_arrays = None

            for batch_hist in hist_batch_arrays:
                hist_arrays = batch_hist if hist_arrays is None else hist_arrays + batch_hist

            # Debug: vérifier les dimensions
            if hist_arrays is None:
                logger.warning(f"[{self.name}] no valid histogram for column '{col}'")
                continue

            logger.debug(f"[{self.name}] hist_arrays shape: {hist_arrays.shape}, expected bins: {self.bins}")

            # sum histogram counts across batches
            if hist_arrays.ndim == 1:
                # Single histogram
                obs_counts = hist_arrays.astype(float)
            else:
                # Multiple histograms from different batches
                # Ensure we're summing along the right axis
                if hist_arrays.shape[1] == self.bins:
                    # Sum along batch dimension (axis=0)
                    obs_counts = np.sum(hist_arrays, axis=0).astype(float)
                else:
                    # Flatten and create a single histogram
                    logger.warning(f"[{self.name}] Unexpected histogram shape {hist_arrays.shape}, flattening")
                    obs_counts = hist_arrays.flatten().astype(float)

            if total_count <= 0 or obs_counts.sum() <= 0:
                logger.warning(f"[{self.name}] no valid data for column '{col}'")
                continue

            total_samples += total_count

            #  distribution parameters and bin edges
            if col not in self._bin_edges:
                logger.warning(f"[{self.name}] no bin edges for column '{col}' - skipping")
                continue

            edges = self._bin_edges[col]

            # theoretical probabilities - Aligné sur DQM-ML officiel
            if self.distribution == "normal":
                # Utilise les MÊMES paramètres que ceux utilisés pour générer les bins

                sample_key = f"{col}_ks_sample"
                if sample_key in batch_metrics:
                    sample_arrays = batch_metrics[sample_key].to_numpy()
                    if sample_arrays.ndim > 1:
                        sample_arrays = sample_arrays.flatten()
                    mean = float(self.dist_params.get("mean", np.mean(sample_arrays)))
                    std = float(self.dist_params.get("std", np.std(sample_arrays, ddof=0)))
                    std = std if std > 0.0 else self.epsilon
                else:
                    # Fallback: use default or configured parameters
                    mean = float(self.dist_params.get("mean", 0.0))
                    std = float(self.dist_params.get("std", 1.0))
                # génère des valeurs aléatoires et compte les fréquences (comme l'officiel)
                expected_values = np.random.normal(mean, std, total_count)
                exp_probs = np.histogram(expected_values, bins=edges)[0].astype(np.float64)
            else:  # uniform
                mn = float(self.dist_params.get("min", edges[0]))
                mx = float(self.dist_params.get("max", edges[-1]))
                # Génère des valeurs aléatoires et compte les fréquences (comme l'officiel)
                expected_values = np.random.uniform(mn, mx, total_count)
                exp_probs = np.histogram(expected_values, bins=edges)[0].astype(np.float64)

            exp_counts = total_count * exp_probs

            col_res: dict[str, Any] = {}

            # chi-square: here we compute the chi-square with a alpha value of 0.05
            if "chi-square" in self.metrics:
                # Ensure observed and expected counts have the same sum

                mask = exp_counts > 0
                if mask.sum() >= 2:
                    # Normalize expected counts to match observed sum
                    obs_sum = obs_counts[mask].sum()
                    exp_sum = exp_counts[mask].sum()

                    if exp_sum > 0:
                        # Scale expected counts to match observed sum
                        exp_counts_normalized = exp_counts[mask] * (obs_sum / exp_sum)

                        try:
                            chi = stats.chisquare(obs_counts[mask], f_exp=exp_counts_normalized)
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
                        if self.distribution == "normal":
                            mean = float(self.dist_params.get("mean", np.mean(ks_samples)))
                            std = float(self.dist_params.get("std", np.std(ks_samples, ddof=0)))
                            std = std if std > 0.0 else self.epsilon
                            ks = stats.kstest(ks_samples, stats.norm.cdf, args=(mean, std))
                        else:  # uniform
                            mn = float(self.dist_params.get("min", np.min(ks_samples)))
                            mx = float(self.dist_params.get("max", np.max(ks_samples)))
                            if mx <= mn:
                                mx = mn + self.epsilon
                            ks = stats.kstest(ks_samples, stats.uniform.cdf, args=(mn, mx - mn))

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

            # Shannon entropy - Aligné sur DQM-ML officiel (utilise fréquences théoriques)
            if "shannon-entropy" in self.metrics:
                # Utilise les fréquences théoriques comme l'officiel
                p_exp = exp_probs / exp_probs.sum()
                h_exp = float(stats.entropy(p_exp))
                col_res["shannon-entropy"] = {
                    "entropy": h_exp,
                    "interpretation": self.interpretation_thresholds.get(
                        "high_diversity" if h_exp > self.shannon_entropy_threshold else "low_diversity",
                        "high_diversity",
                    ),
                }

            # GRTE (gap between observed and theoretical entropies) - Aligné sur DQM-ML officiel
            if "grte" in self.metrics:
                # Utilise les fréquences observées et théoriques comme l'officiel
                p_obs = obs_counts / obs_counts.sum()
                p_exp = exp_probs / exp_probs.sum()
                h_obs = float(stats.entropy(p_obs))
                h_exp = float(stats.entropy(p_exp))
                grte = float(np.exp(-2.0 * abs(h_exp - h_obs)))
                col_res["grte"] = {
                    "grte_value": grte,
                    "interpretation": self.interpretation_thresholds.get(
                        "high_representativeness" if grte > self.grte_threshold else "low_representativeness",
                        "high_representativeness",
                    ),
                }

            # Stupid way to flatten tree of keys
            # TODO : refactor the implementation
            if col_res:
                for key, value in col_res.items():
                    if isinstance(value, dict):
                        for prop, content in value.items():
                            out[key + "_" + col + "_" + prop] = content
                    else:
                        out[key + "_" + col] = value
            # TODO : we generate here the following column names
            # grte_*, ...

        meta_data = {
            "bins": self.bins,
            "distribution": self.distribution,
            "metrics_computed": self.metrics,
            "total_samples": total_samples,
            "columns_analyzed": [c for c in self.input_columns if f"{c}_count" in batch_metrics],
            "ks_sampling_enabled": "kolmogorov-smirnov" in self.metrics,
            "note": "KS test uses random sampling approximation for scalability",
        }

        import json

        out["_metadata"] = json.dumps(meta_data)
        return out

    def reset(self) -> None:
        """Reset processor state for new processing run."""
        self._bin_edges = {}
        self._initialized = False

    # utils methods for bin edge calculation

    def _bin_edges_normal(self, mean: float, std: float, bins: int, data: np.ndarray) -> np.ndarray:
        """Return bin edges for normal distribution - Aligné sur DQM-ML officiel."""
        # Logique officielle : utilise stats.norm.ppf avec linspace(1/bins, 1, bins)
        interval = []
        for i in range(1, bins):
            val = stats.norm.ppf(i / bins, mean, std)
            interval.append(val)
        interval.insert(0, -np.inf)
        interval.append(np.inf)
        return np.array(interval)

    def _bin_edges_uniform(self, mn: float, mx: float, bins: int, data: np.ndarray) -> np.ndarray:
        """Return bin edges for uniform distribution."""
        lo = min(mn, float(np.min(data)))
        hi = max(mx, float(np.max(data)))
        if hi <= lo:
            hi = lo + self.epsilon
        return np.linspace(lo, hi, bins + 1)
