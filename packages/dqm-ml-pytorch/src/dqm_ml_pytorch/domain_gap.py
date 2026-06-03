"""Domain gap processor for measuring distribution distance between datasets.

This module contains the DomainGapProcessor class that computes statistical
distances (KL divergence, MMD variants, FID, Wasserstein, PAD, CMD) between
source and target datasets using image embeddings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa

# COMPATIBILITY : from typing import Any, override # When support of 3.10 and 3.11 will be removed
from typing_extensions import override

from dqm_ml_core import DatametricProcessor

_MISSING_EMB_MSG = "missing __emb__ — set summary.store_embeddings=true"

logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    """Check whether debug data generation is enabled via environment variable.

    Returns:
        True if ``DQM_ML_DEBUG`` is set to a truthy value ("1", "true", "yes").
    """
    return os.environ.get("DQM_ML_DEBUG", "").strip().lower() in {"1", "true", "yes"}


# Known ResNet-18 layer embedding dimensions (C*H*W) mapped to channel counts (C).
# Used in _compute_batch_metric_cmd to spatially pool flattened features
# so CMD moments match the per-channel computation of v1.
_CMD_RESNET18_EMBDIM_CHANNELS: dict[int, int] = {
    200704: 64,  # maxpool / layer1.1.relu_1:    64 * 56 * 56
    100352: 128,  # layer2.1.relu_1:  128 * 28 * 28
    50176: 256,  # layer3.1.relu_1:  256 * 14 * 14
    25088: 512,  # layer4.1.relu_1:  512 * 7  * 7
}


def _fixed_to_matrix(arr: pa.FixedSizeListArray) -> np.ndarray:
    """Convert a FixedSizeListArray to a (N, D) numpy float64 matrix.

    Args:
        arr: Input FixedSizeListArray with entries of equal length.

    Returns:
        A 2D numpy array of shape (N, D).
    """
    vals = np.asarray(arr.values.to_numpy(), dtype=np.float64)
    dim = len(arr[0])
    return vals.reshape(-1, dim)


def _sum_fixed(fixed_list_array: pa.FixedSizeListArray) -> tuple[np.ndarray, int]:
    """Sum all FixedSizeList entries into a single numpy vector.

    Args:
        fixed_list_array: Input FixedSizeListArray.

    Returns:
        Tuple of (sum_vector, list_size).
    """
    vals = np.asarray(fixed_list_array.values.to_numpy(), dtype=np.float64)
    list_size = len(fixed_list_array[0])
    return vals.reshape(-1, list_size).sum(axis=0), list_size


def _sum_scalar(arr: pa.Array) -> int:
    """Sum all values in a pyarrow Array and return as int.

    Args:
        arr: Input pyarrow Array.

    Returns:
        Integer sum of all elements.
    """
    return int(np.asarray(arr.to_numpy()).sum())


def _mmd_rbf(src_emb: np.ndarray, tgt_emb: np.ndarray, gamma: float) -> float:
    """Compute Maximum Mean Discrepancy with an RBF kernel.

    Uses the biased estimator matching the legacy implementation:
        MMD^2 = mean(K_xx) + mean(K_yy) - 2 * mean(K_xy)

    The RBF kernel uses non-squared Euclidean distance (matching legacy):
        K(x, y) = exp(-gamma * ||x - y||)

    Args:
        src_emb: Source embeddings, shape (N, D).
        tgt_emb: Target embeddings, shape (M, D).
        gamma: RBF kernel coefficient.

    Returns:
        Scalar MMD^2 value.
    """
    m, n = len(src_emb), len(tgt_emb)
    if m <= 1 or n <= 1:
        return 0.0

    # pairwise Euclidean distances (non-squared, matching legacy torch.cdist)
    sq_src = np.sum(src_emb**2, axis=1, keepdims=True)
    sq_tgt = np.sum(tgt_emb**2, axis=1, keepdims=True)
    cross = src_emb @ tgt_emb.T
    dist_xy = np.sqrt(np.maximum(0.0, sq_src - 2 * cross + sq_tgt.T))

    k_xy = np.exp(-gamma * dist_xy)

    # Within-source
    sq_src_src = np.sum(src_emb**2, axis=1, keepdims=True)
    cross_xx = src_emb @ src_emb.T
    dist_xx = np.sqrt(np.maximum(0.0, sq_src_src - 2 * cross_xx + sq_src_src.T))
    k_xx = np.exp(-gamma * dist_xx)

    # Within-target
    sq_tgt_tgt = np.sum(tgt_emb**2, axis=1, keepdims=True)
    cross_yy = tgt_emb @ tgt_emb.T
    dist_yy = np.sqrt(np.maximum(0.0, sq_tgt_tgt - 2 * cross_yy + sq_tgt_tgt.T))
    k_yy = np.exp(-gamma * dist_yy)

    mmd2 = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return float(max(mmd2, 0.0))


def _mmd_poly(src_emb: np.ndarray, tgt_emb: np.ndarray, degree: float, gamma: float, coefficient0: float) -> float:
    """Compute Maximum Mean Discrepancy with a polynomial kernel.

    Uses the biased estimator matching the legacy implementation:
        MMD^2 = mean(K_xx) + mean(K_yy) - 2 * mean(K_xy)

    Polynomial kernel: K(a, b) = (gamma * <a, b> + coefficient0)^degree

    Args:
        src_emb: Source embeddings, shape (N, D).
        tgt_emb: Target embeddings, shape (M, D).
        degree: Polynomial degree.
        gamma: Scaling factor for the dot product.
        coefficient0: Bias term.

    Returns:
        Scalar MMD^2 value.
    """
    m, n = len(src_emb), len(tgt_emb)
    if m <= 1 or n <= 1:
        return 0.0

    def _poly_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (gamma * (a @ b.T) + coefficient0) ** degree  # type: ignore[no-any-return]

    k_xx = _poly_kernel(src_emb, src_emb)
    k_yy = _poly_kernel(tgt_emb, tgt_emb)
    k_xy = _poly_kernel(src_emb, tgt_emb)

    mmd2 = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return float(max(mmd2, 0.0))


def _pad_distance(src_emb: np.ndarray, tgt_emb: np.ndarray, evaluator: str) -> float:
    """Compute Proxy A-Distance (PAD) using a linear SVM.

    Trains an SVM to discriminate source vs target, then returns
    2 * (1 - 2 * error) where error is MSE or MAE of the classifier.

    Args:
        src_emb: Source embeddings, shape (N, D).
        tgt_emb: Target embeddings, shape (M, D).
        evaluator: Error metric, "mse" or "mae".

    Returns:
        PAD scalar value.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    from sklearn.svm import SVC

    x_svm = np.vstack([src_emb, tgt_emb])
    y = np.hstack([np.zeros(len(src_emb)), np.ones(len(tgt_emb))])

    svm = SVC(C=1, kernel="linear", probability=True, random_state=42, verbose=0, gamma="auto")
    svm.fit(x_svm, y)
    pred = svm.predict_proba(x_svm)

    y_onehot = np.zeros_like(pred)
    y_onehot[np.arange(len(y)), y.astype(int)] = 1

    error = float(np.mean((pred - y_onehot) ** 2)) if evaluator == "mse" else float(np.mean(np.abs(pred - y_onehot)))

    return 2.0 * (1.0 - 2.0 * error)


class DomainGapProcessor(DatametricProcessor):
    """Computes statistical distances between source and target
    dataselections using image embeddings.

    This processor works in two stages:
    1. Dataset Summary: Aggregates high-dimensional embeddings into
        compact statistics (mean, variance, outer products, histograms).
    2. Delta Computation: Uses these summaries to calculate distance
        metrics between a source and a target dataset.

    Supported Delta Metrics:
      - ``klmvn_diag``: KL divergence assuming a multivariate Normal
        distribution with a diagonal covariance matrix.
      - ``mmd_linear``: Maximum Mean Discrepancy with a linear kernel.
      - ``mmd_rbf``: Maximum Mean Discrepancy with an RBF kernel.
      - ``mmd_poly``: Maximum Mean Discrepancy with a polynomial kernel.
      - ``fid``: Frechet Inception Distance.
      - ``wasserstein_1d``: Average 1D Wasserstein distance across
        embedding dimensions, approximated via histograms.
      - ``pad``: Proxy A-Distance via linear SVM.
      - ``cmd``: Central Moment Discrepancy (multi-layer only).
    """

    def __init__(
        self,
        name: str = "domain_gap",
        config: dict[str, Any] | None = None,
    ):
        """Initialize the domain gap processor.

        Args:
            name: Unique name of the processor instance.
            config: Configuration dictionary containing:
                - input:
                    - embedding_col: Column name containing embeddings (default: "embedding").
                    - embedding_cols: List of column names for multi-layer metrics (CMD).
                - summary:
                    - collect_sum_outer: Whether to compute outer products (needed for FID).
                    - collect_hist_1d: Whether to compute histograms (needed for Wasserstein).
                    - hist_dims: Number of dimensions to histogram.
                    - hist_bins: Number of bins per histogram.
                    - store_embeddings: Whether to store raw embeddings for full-data metrics.
                - delta:
                    - metric: Target metric name.
                    - k: Number of moments (CMD only, default 5).
                    - feature_weights: Per-layer weights (CMD only).
                    - kernel_params: Kernel parameters (MMD-RBF/Poly).
                - method:
                    - evaluator: Error metric for PAD ("mse" or "mae").
        """
        super().__init__(name, config)
        self._checked = False

    # ---------------- API ----------------
    def check_config(self) -> None:
        """Validate and configure the domain gap processor.

        This method parses the configuration dictionary and sets:
        - input: Embedding column(s)
        - summary: Options for collecting summary statistics
        - delta: The target metric and its parameters
        """
        cfg = self.config or {}
        input_cfg = cfg.get("input", {})
        self.embedding_col: str = input_cfg.get("embedding_col", "embedding")

        delta_cfg = cfg.get("delta", {})
        self.delta_metric: str = str(delta_cfg.get("metric", "klmvn_diag")).lower()
        self.is_cmd = self.delta_metric == "cmd"

        summary_cfg = cfg.get("summary", {})

        # Determine auto-settings per metric
        full_data_metrics = {"mmd_rbf", "mmd_poly", "pad", "cmd"}
        auto_store_emb = self.delta_metric in full_data_metrics

        if self.delta_metric == "fid":
            auto_sum_outer = True
            auto_hist_1d = False
        elif self.delta_metric == "wasserstein_1d":
            auto_sum_outer = False
            auto_hist_1d = True
        else:
            auto_sum_outer = False
            auto_hist_1d = False

        self.collect_sum_outer: bool = bool(summary_cfg.get("collect_sum_outer", auto_sum_outer))
        self.collect_hist_1d: bool = bool(summary_cfg.get("collect_hist_1d", auto_hist_1d))
        self.store_embeddings: bool = bool(summary_cfg.get("store_embeddings", auto_store_emb))

        # Wasserstein-1D parameters
        self.hist_dims: int = int(summary_cfg.get("hist_dims", 64))
        self.hist_bins: int = int(summary_cfg.get("hist_bins", 32))
        rng = summary_cfg.get("hist_range", [-3.0, 3.0])
        self.hist_range: tuple[float, float] = (float(rng[0]), float(rng[1]))

        # CMD-specific parameters
        if self.is_cmd:
            self.cmd_k: int = int(delta_cfg.get("k", 5))
            self.cmd_embedding_cols: list[str] = input_cfg.get("embedding_cols", [self.embedding_col])
            self.cmd_feature_weights: list[float] = list(
                delta_cfg.get("feature_weights", [1.0] * len(self.cmd_embedding_cols))
            )
            if len(self.cmd_feature_weights) != len(self.cmd_embedding_cols):
                raise ValueError(
                    f"[{self.name}] feature_weights length ({len(self.cmd_feature_weights)}) "
                    f"must match embedding_cols length ({len(self.cmd_embedding_cols)})"
                )

        # Kernel parameters for MMD-RBF / MMD-Poly
        self.kernel_params: dict[str, Any] = delta_cfg.get("kernel_params", {})

        # PAD parameters
        self.pad_evaluator: str = str(cfg.get("method", {}).get("evaluator", "mse"))

        self._checked = True

    @override
    def needed_columns(self) -> list[str]:
        """Return the list of columns required for domain gap computation.

        Returns:
            List of embedding column names. For CMD returns multi-layer columns.
        """
        if not getattr(self, "_checked", False):
            self.check_config()
        if self.is_cmd:
            return self.cmd_embedding_cols
        return [self.embedding_col]

    def generated_columns(self) -> list[str]:
        """Return the list of columns generated by this processor.

        Returns:
            Empty list as this processor computes deltas rather than features.
        """
        return []

    @override
    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Reduce a batch of embeddings into summary statistics.

        For single-column metrics, computes count, sum, sum_sq, and
        optionally sum_outer, hist_counts, and raw embeddings.

        For CMD, computes raw moments up to order k for each embedding
        column.

        Args:
            features: Dictionary of feature arrays from the batch.

        Returns:
            Dictionary of aggregated statistics per batch.
        """
        if not getattr(self, "_checked", False):
            self.check_config()

        if self.is_cmd:
            return self._compute_batch_metric_cmd(features)

        emb = features.get(self.embedding_col)
        if emb is None or not isinstance(emb, pa.FixedSizeListArray):
            return {}

        num_samples = len(emb)
        embed_dim = len(emb[0])
        flat_values = emb.values
        emb_matrix = np.asarray(flat_values.to_numpy()).reshape(num_samples, embed_dim)

        out: dict[str, pa.Array] = {}
        out["count"] = pa.array([num_samples], type=pa.int64())
        sum_vec = emb_matrix.sum(axis=0).astype(np.float64)
        out["sum"] = pa.FixedSizeListArray.from_arrays(pa.array(sum_vec), embed_dim)
        sum_sq_vec = (emb_matrix * emb_matrix).sum(axis=0).astype(np.float64)
        out["sum_sq"] = pa.FixedSizeListArray.from_arrays(pa.array(sum_sq_vec), embed_dim)

        # optional: sum_outer for FID
        if self.collect_sum_outer:
            sum_outer_product = (emb_matrix.T @ emb_matrix).reshape(-1).astype(np.float64)
            outer_dim = embed_dim * embed_dim
            out["sum_outer"] = pa.FixedSizeListArray.from_arrays(pa.array(sum_outer_product), outer_dim)

        # optional: histograms for Wasserstein-1D
        if self.collect_hist_1d:
            use_dims = min(embed_dim, self.hist_dims)
            low, high = self.hist_range
            hist_list: list[np.ndarray] = []
            for j in range(use_dims):
                hist_1d, _ = np.histogram(emb_matrix[:, j], bins=self.hist_bins, range=(low, high))
                hist_list.append(hist_1d.astype(np.int64))
            hist_all = np.stack(hist_list, axis=0).reshape(-1)
            out["hist_counts"] = pa.FixedSizeListArray.from_arrays(pa.array(hist_all), self.hist_bins * use_dims)

        # optional: raw embeddings for full-data metrics (MMD-RBF, MMD-Poly, PAD)
        if self.store_embeddings:
            out["__emb__"] = emb

        return out

    def _compute_batch_metric_cmd(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Compute per-batch raw moment power sums for CMD.

        For each CMD column, applies sigmoid and accumulates sum(x^j)
        for j=1..k (raw moment sums). These are aggregated across batches
        in _compute_cmd_aggregate and converted to central moments in
        _compute_delta_cmd.

        Args:
            features: Dictionary of feature arrays from the batch.

        Returns:
            Dictionary of power sums and counts per batch.
        """
        if not getattr(self, "_checked", False):
            self.check_config()

        out: dict[str, pa.Array] = {}
        for col in self.cmd_embedding_cols:
            emb = features.get(col)
            if emb is None or not isinstance(emb, pa.FixedSizeListArray):
                continue
            mat = _fixed_to_matrix(emb)
            batch_n = len(mat)
            if batch_n == 0:
                continue

            # Apply sigmoid (matching v1 behavior)
            mat = 1.0 / (1.0 + np.exp(-mat))

            # Per-channel spatial reshaping to match v1's moment computation.
            # v1 treats each individual spatial element as a sample, computing
            # moments over all C x H x W values per channel across all images.
            # Reshape flattened (N, C*H*W) → (N, C, H*W) so we can sum over
            # both N and H*W, matching v1's element-wise treatment.
            channels = self._resolve_cmd_channels(col, mat.shape[1], features)
            hw = mat.shape[1] // channels
            mat = mat.reshape(-1, channels, hw)

            out[f"cmd_{col}_n"] = pa.array([batch_n * hw], type=pa.int64())

            # Raw moment sums: sum(x^j) for j=1..k over all spatial elements
            for j in range(1, self.cmd_k + 1):
                power_sum = np.power(mat, j).sum(axis=(0, 2)).astype(np.float64)
                out[f"cmd_{col}_sum_{j}"] = pa.FixedSizeListArray.from_arrays(pa.array(power_sum), len(power_sum))

        return out

    def _resolve_cmd_channels(self, col: str, flattened_dim: int, features: dict[str, pa.Array]) -> int:
        channels_col = f"{col}_channels"
        channels_arr = features.get(channels_col)
        if channels_arr is not None and len(channels_arr) > 0:
            c = int(channels_arr[0].as_py())
            if flattened_dim % c == 0:
                return c

        _c = _CMD_RESNET18_EMBDIM_CHANNELS.get(flattened_dim)
        if _c is not None and flattened_dim % _c == 0:
            return _c

        raise ValueError(
            f"Cannot determine channels for embedding column '{col}' "
            f"(flattened_dim={flattened_dim}). "
            f"The image_embedding processor did not produce a "
            f"'{channels_col}' metadata column, and the dimension "
            f"is not in the legacy lookup table. "
            f"Provide 'cmd_channels' in the domain_gap delta config "
            f"or ensure the image_embedding processor outputs "
            f"'{channels_col}'."
        )

    @override
    def compute(self, batch_metrics: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Aggregate batch-level summary statistics into global dataselection statistics.

        For summary-based metrics, aggregates count, sum, sum_sq, etc.
        For CMD, aggregates per-batch power sums for later central moment
        computation in compute_delta.
        For store_embeddings, concatenates raw embedding arrays.

        Args:
            batch_metrics: Dictionary containing batch-level statistics.

        Returns:
            Dictionary containing aggregated dataset-level statistics.
        """
        if not batch_metrics:
            return {}

        if self.is_cmd:
            return self._compute_cmd_aggregate(batch_metrics)

        out: dict[str, pa.Array] = {}

        # count
        if "count" not in batch_metrics:
            return {}
        total_n = _sum_scalar(batch_metrics["count"])
        out["count"] = pa.array([total_n], type=pa.int64())

        # sum / sum_sq
        if "sum" in batch_metrics:
            sum_vec, list_size = _sum_fixed(batch_metrics["sum"])
            out["sum"] = pa.FixedSizeListArray.from_arrays(pa.array(sum_vec), list_size)
        if "sum_sq" in batch_metrics:
            sum_sq_vec, list_size2 = _sum_fixed(batch_metrics["sum_sq"])
            out["sum_sq"] = pa.FixedSizeListArray.from_arrays(pa.array(sum_sq_vec), list_size2)

        # optional sum_outer
        if "sum_outer" in batch_metrics:
            so_vals = np.asarray(batch_metrics["sum_outer"].values.to_numpy(), dtype=np.float64)
            outer_dim = len(batch_metrics["sum_outer"][0])
            out["sum_outer"] = pa.FixedSizeListArray.from_arrays(
                pa.array(so_vals.reshape(-1, outer_dim).sum(axis=0)), outer_dim
            )

        # optional hist_counts
        if "hist_counts" in batch_metrics:
            h_vals = np.asarray(batch_metrics["hist_counts"].values.to_numpy(), dtype=np.int64)
            h_len = len(batch_metrics["hist_counts"][0])
            out["hist_counts"] = pa.FixedSizeListArray.from_arrays(
                pa.array(h_vals.reshape(-1, h_len).sum(axis=0)), h_len
            )

        # raw embeddings for full-data metrics
        if self.store_embeddings and "__emb__" in batch_metrics:
            vals = np.asarray(batch_metrics["__emb__"].values.to_numpy(), dtype=np.float64)
            dim = len(batch_metrics["__emb__"][0])
            out["__emb__"] = pa.FixedSizeListArray.from_arrays(pa.array(vals), dim)

        return out

    def _compute_cmd_aggregate(self, batch_metrics: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Aggregate CMD power sums across batches.

        Args:
            batch_metrics: Dictionary containing per-batch power sums.

        Returns:
            Dictionary with aggregated power sums and total count per layer.
        """
        out: dict[str, pa.Array] = {}
        for col in self.cmd_embedding_cols:
            n_key = f"cmd_{col}_n"
            if n_key not in batch_metrics:
                continue

            total_n = _sum_scalar(batch_metrics[n_key])
            if total_n == 0:
                continue
            out[n_key] = pa.array([total_n], type=pa.int64())

            for j in range(1, self.cmd_k + 1):
                sum_key = f"cmd_{col}_sum_{j}"
                if sum_key in batch_metrics:
                    sum_vec, dim = _sum_fixed(batch_metrics[sum_key])
                    out[sum_key] = pa.FixedSizeListArray.from_arrays(pa.array(sum_vec), dim)

        return out

    @override
    def compute_delta(self, source: dict[str, pa.Array], target: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Calculate the domain gap metric between source and target statistics.

        Args:
            source: Dataselection statistics from the source dataset.
            target: Dataselection statistics from the target dataset.

        Returns:
            Dictionary containing the calculated metric value.
        """
        if not getattr(self, "_checked", False):
            self.check_config()

        metric = self.delta_metric

        if self.is_cmd:
            return self._compute_delta_cmd(source, target)

        if metric in {"klmvn_diag", "mmd_linear", "fid"}:
            return self._compute_delta_summary(source, target, metric)

        if metric == "wasserstein_1d":
            return self._compute_delta_wasserstein(source, target)

        if metric == "mmd_rbf":
            return self._compute_delta_mmd_rbf(source, target)

        if metric == "mmd_poly":
            return self._compute_delta_mmd_poly(source, target)

        if metric == "pad":
            return self._compute_delta_pad(source, target)

        return {
            "metric": pa.array([metric]),
            "note": pa.array(["unsupported metric or invalid inputs"]),
        }

    def _compute_delta_summary(
        self, source: dict[str, pa.Array], target: dict[str, pa.Array], metric: str
    ) -> dict[str, pa.Array]:
        """Compute KLMVN, MMD-Linear, or FID from summary statistics.

        Args:
            source: Source dataset statistics.
            target: Target dataset statistics.
            metric: One of "klmvn_diag", "mmd_linear", "fid".

        Returns:
            Dictionary with the metric value.
        """
        need = {"count", "sum"}
        if metric in {"klmvn_diag", "fid"}:
            need |= {"sum_sq"}
        if metric == "fid":
            need |= {"sum_outer"}
        for dataset_stats, name in ((source, "source"), (target, "target")):
            if not need.issubset(dataset_stats.keys()):
                return {
                    "metric": pa.array([metric]),
                    "note": pa.array([f"missing keys in {name}: {sorted(need)}"]),
                }

        n_src = _sum_scalar(source["count"])
        n_tgt = _sum_scalar(target["count"])
        if n_src <= 0 or n_tgt <= 0:
            return {
                "metric": pa.array([metric]),
                "note": pa.array(["empty summaries"]),
            }

        mean_src = _sum_fixed(source["sum"])[0] / n_src
        mean_tgt = _sum_fixed(target["sum"])[0] / n_tgt

        if metric == "mmd_linear":
            diff = mean_src - mean_tgt
            val = float(np.dot(diff, diff))
            return {"mmd_linear": pa.array([val], type=pa.float64())}

        # variance = E[X^2] - E[X]^2; clip to 1e-9 to avoid division by zero
        var_src = np.maximum(_sum_fixed(source["sum_sq"])[0] / n_src - mean_src * mean_src, 1e-9)
        var_tgt = np.maximum(_sum_fixed(target["sum_sq"])[0] / n_tgt - mean_tgt * mean_tgt, 1e-9)

        if metric == "klmvn_diag":
            term_var = np.sum(var_src / var_tgt - 1.0 - np.log(var_src / var_tgt))
            term_mean = np.sum((mean_tgt - mean_src) ** 2 / var_tgt)
            val = 0.5 * (term_var + term_mean)
            return {"klmvn_diag": pa.array([float(val)], type=pa.float64())}

        if metric == "fid":
            from scipy.linalg import sqrtm

            sum_outer_src = _sum_fixed(source["sum_outer"])[0]
            sum_outer_tgt = _sum_fixed(target["sum_outer"])[0]
            embed_dim = int(np.sqrt(sum_outer_src.size))
            cov_src = (sum_outer_src.reshape(embed_dim, embed_dim) / n_src) - np.outer(mean_src, mean_src)
            cov_tgt = (sum_outer_tgt.reshape(embed_dim, embed_dim) / n_tgt) - np.outer(mean_tgt, mean_tgt)

            covmean = sqrtm(cov_src.dot(cov_tgt))
            if np.iscomplexobj(covmean):
                covmean = covmean.real

            diff = mean_src - mean_tgt
            fid = diff.dot(diff) + np.trace(cov_src) + np.trace(cov_tgt) - 2 * np.trace(covmean)
            return {"fid": pa.array([float(abs(fid))], type=pa.float64())}

        return {"metric": pa.array([metric]), "note": pa.array(["unreachable"])}

    def _compute_delta_wasserstein(
        self, source: dict[str, pa.Array], target: dict[str, pa.Array]
    ) -> dict[str, pa.Array]:
        """Compute 1D Wasserstein distance from histogram summaries.

        Args:
            source: Source dataset statistics.
            target: Target dataset statistics.

        Returns:
            Dictionary with wasserstein_1d value.
        """
        if "hist_counts" not in source or "hist_counts" not in target:
            return {
                "metric": pa.array(["wasserstein_1d"]),
                "note": pa.array(["missing hist_counts"]),
            }
        h_src = np.asarray(source["hist_counts"].values.to_numpy(), dtype=np.int64)
        h_tgt = np.asarray(target["hist_counts"].values.to_numpy(), dtype=np.int64)
        use_dims = self.hist_dims
        bins = self.hist_bins
        if h_src.size != h_tgt.size or h_src.size != bins * use_dims:
            return {
                "metric": pa.array(["wasserstein_1d"]),
                "note": pa.array(["hist_counts length mismatch"]),
            }
        width = (self.hist_range[1] - self.hist_range[0]) / bins
        total = 0.0
        used = 0
        for j in range(use_dims):
            h_src_slice = h_src[j * bins : (j + 1) * bins].astype(np.float64)
            h_tgt_slice = h_tgt[j * bins : (j + 1) * bins].astype(np.float64)
            if h_src_slice.sum() == 0 and h_tgt_slice.sum() == 0:
                continue
            prob_src = h_src_slice / max(1.0, h_src_slice.sum())
            prob_tgt = h_tgt_slice / max(1.0, h_tgt_slice.sum())
            cdf_src = np.cumsum(prob_src)
            cdf_tgt = np.cumsum(prob_tgt)
            total += float(np.sum(np.abs(cdf_src - cdf_tgt)) * width)
            used += 1
        val = total / max(1, used)
        return {"wasserstein_1d": pa.array([val], type=pa.float64())}

    def _compute_delta_mmd_rbf(self, source: dict[str, pa.Array], target: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Compute MMD with RBF kernel from stored embeddings.

        Args:
            source: Source dataset statistics including "__emb__".
            target: Target dataset statistics including "__emb__".

        Returns:
            Dictionary with mmd_rbf value.
        """
        if "__emb__" not in source or "__emb__" not in target:
            return {
                "metric": pa.array(["mmd_rbf"]),
                "note": pa.array([_MISSING_EMB_MSG]),
            }
        src = _fixed_to_matrix(source["__emb__"])
        tgt = _fixed_to_matrix(target["__emb__"])
        gamma = float(self.kernel_params.get("gamma", 1.0))
        val = _mmd_rbf(src, tgt, gamma)
        return {"mmd_rbf": pa.array([val], type=pa.float64())}

    def _compute_delta_mmd_poly(self, source: dict[str, pa.Array], target: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Compute MMD with polynomial kernel from stored embeddings.

        Args:
            source: Source dataset statistics including "__emb__".
            target: Target dataset statistics including "__emb__".

        Returns:
            Dictionary with mmd_poly value.
        """
        if "__emb__" not in source or "__emb__" not in target:
            return {
                "metric": pa.array(["mmd_poly"]),
                "note": pa.array([_MISSING_EMB_MSG]),
            }
        src = _fixed_to_matrix(source["__emb__"])
        tgt = _fixed_to_matrix(target["__emb__"])
        degree = float(self.kernel_params.get("degree", 3.0))
        gamma = float(self.kernel_params.get("gamma", 1.0))
        coefficient0 = float(self.kernel_params.get("coefficient0", 1.0))
        val = _mmd_poly(src, tgt, degree, gamma, coefficient0)
        return {"mmd_poly": pa.array([val], type=pa.float64())}

    def _compute_delta_pad(self, source: dict[str, pa.Array], target: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Compute Proxy A-Distance from stored embeddings.

        Args:
            source: Source dataset statistics including "__emb__".
            target: Target dataset statistics including "__emb__".

        Returns:
            Dictionary with pad value.
        """
        if "__emb__" not in source or "__emb__" not in target:
            return {
                "metric": pa.array(["pad"]),
                "note": pa.array([_MISSING_EMB_MSG]),
            }
        src = _fixed_to_matrix(source["__emb__"])
        tgt = _fixed_to_matrix(target["__emb__"])
        val = _pad_distance(src, tgt, self.pad_evaluator)
        return {"pad": pa.array([val], type=pa.float64())}

    def _compute_delta_cmd(self, source: dict[str, pa.Array], target: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Compute Central Moment Discrepancy between source and target.

        Computes per-layer raw moments from power sums, converts to central
        moments, and compares them using Euclidean distance (matching v1's
        RMSELoss). Weighted averaging follows v1's formula:
            layer_loss = (1/k) * sum(rmse(moment) for moment in 0..k-1)
            total_loss = sum(weight * layer_loss for each layer)
            cmd = total_loss / sum(weights)

        Args:
            source: Source dataset statistics including cmd_{col}_n and
                    cmd_{col}_sum_{j} for j=1..k.
            target: Target dataset statistics (same keys as source).

        Returns:
            Dictionary with cmd value.
        """
        total_loss = 0.0
        total_weight = 0.0
        debug_data: dict[str, np.ndarray] | None = {} if _debug_enabled() else None

        for col, weight in zip(self.cmd_embedding_cols, self.cmd_feature_weights, strict=True):
            if weight == 0:
                continue

            layer_result = self._compute_layer_cmd(col, source, target, debug_data)
            if layer_result is None:
                continue

            layer_loss, layer_debug = layer_result
            total_weight += weight
            total_loss += weight * layer_loss

            if debug_data is not None and layer_debug is not None:
                debug_data.update(layer_debug)

        if debug_data is not None:
            tmp_path = str(Path(tempfile.gettempdir()) / f"debug_moments_{os.getpid()}.npz")
            np.savez_compressed(tmp_path, **debug_data)  # type: ignore[arg-type]

        if total_weight == 0:
            return {"metric": pa.array(["cmd"]), "note": pa.array(["no valid layers"])}

        final_loss = total_loss / total_weight
        return {"cmd": pa.array([final_loss], type=pa.float64())}

    def _compute_layer_cmd(
        self,
        col: str,
        source: dict[str, pa.Array],
        target: dict[str, pa.Array],
        debug_data: dict[str, np.ndarray] | None,
    ) -> tuple[float, dict[str, np.ndarray]] | None:
        """Compute CMD loss for a single embedding layer.

        Args:
            col: Layer column name.
            source: Source statistics.
            target: Target statistics.
            debug_data: Optional debug dict to populate.

        Returns:
            Tuple of (layer_loss, debug_entries) or None if layer is invalid.
        """
        from math import comb

        n_src_key = f"cmd_{col}_n"
        n_tgt_key = f"cmd_{col}_n"
        if n_src_key not in source or n_tgt_key not in target:
            return None

        n_src = int(source[n_src_key].to_numpy()[0])
        n_tgt = int(target[n_tgt_key].to_numpy()[0])
        if n_src <= 0 or n_tgt <= 0:
            return None

        all_j = list(range(1, self.cmd_k + 1))
        if not all(f"cmd_{col}_sum_{j}" in source and f"cmd_{col}_sum_{j}" in target for j in all_j):
            return None

        # Raw moments from power sums: E[X^j] = sum(x^j) / n
        src_raw = []
        tgt_raw = []
        for j in all_j:
            src_sum, _ = _sum_fixed(source[f"cmd_{col}_sum_{j}"])
            tgt_sum, _ = _sum_fixed(target[f"cmd_{col}_sum_{j}"])
            src_raw.append(src_sum / n_src)
            tgt_raw.append(tgt_sum / n_tgt)

        # Debug: save raw moments per layer
        layer_debug: dict[str, np.ndarray] = {}
        if debug_data is not None:
            layer_key = col
            for prefix in ["image_embedding_cmd_", "image_embedding_"]:
                if col.startswith(prefix):
                    layer_key = col[len(prefix) :]
                    break
            layer_debug[f"{layer_key}/mean_src"] = src_raw[0]
            layer_debug[f"{layer_key}/mean_tgt"] = tgt_raw[0]
            layer_debug[f"{layer_key}/raw_moment2_src"] = src_raw[1]
            layer_debug[f"{layer_key}/raw_moment2_tgt"] = tgt_raw[1]
            layer_debug[f"{layer_key}/n_src"] = np.array([n_src], dtype=np.int64)
            layer_debug[f"{layer_key}/n_tgt"] = np.array([n_tgt], dtype=np.int64)

        # Convert raw moments to central moments
        mu_src = src_raw[0]
        mu_tgt = tgt_raw[0]
        src_cm = [mu_src]
        tgt_cm = [mu_tgt]

        for order in range(2, self.cmd_k + 1):
            cm_src = np.zeros_like(mu_src)
            cm_tgt = np.zeros_like(mu_tgt)
            for i in range(order + 1):
                coeff = float(comb(order, i))
                if i == 0:
                    raw_src = np.array(1.0)
                    raw_tgt = np.array(1.0)
                else:
                    raw_src = src_raw[i - 1]
                    raw_tgt = tgt_raw[i - 1]
                cm_src += coeff * raw_src * ((-mu_src) ** (order - i))
                cm_tgt += coeff * raw_tgt * ((-mu_tgt) ** (order - i))
            src_cm.append(cm_src)
            tgt_cm.append(cm_tgt)

        # Euclidean distance per moment (matching v1's RMSELoss = sqrt(sum(diff^2)))
        layer_loss = 0.0
        for t in range(self.cmd_k):
            diff = src_cm[t] - tgt_cm[t]
            dist = float(np.sqrt(np.sum(diff**2)))
            layer_loss += dist

        layer_loss /= self.cmd_k
        return (layer_loss, layer_debug)
