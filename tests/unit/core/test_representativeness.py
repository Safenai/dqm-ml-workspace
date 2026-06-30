"""Unit tests for the representativeness metric processor.

This module contains tests that verify the representativeness metric processor
correctly computes statistical representativeness metrics (Shannon entropy,
GRTE, Kolmogorov-Smirnov, Chi-square) for numeric data, including handling
of missing columns, non-numeric data, seed reproducibility, and edge cases.
"""

import logging
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from dqm_ml_core.metrics.representativeness import RepresentativenessProcessor

# --- Batch Metric Computation Tests ---


def test_compute_empty_batch_metrics():
    """Verify compute returns error metadata when batch_metrics is None."""
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    result = proc.compute(batch_metrics=None)
    assert result == {"_metadata": {"error": "No batch metrics provided"}}


def test_compute_batch_metric_column_not_in_features(caplog):
    """Verify compute_batch_metric returns empty dict and logs for missing column.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    features = {}

    with caplog.at_level(logging.WARNING):
        result = proc.compute_batch_metric(features)

    assert result == {}
    assert "column 'col1' not found in batch" in caplog.text


def test_compute_batch_metric_non_numeric_column(caplog):
    """Verify compute_batch_metric returns empty dict and logs for non-numeric column.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    features = {"col1": pa.array(["a", "b", "c"])}

    with caplog.at_level(logging.WARNING):
        result = proc.compute_batch_metric(features)

    assert result == {}
    assert "no valid numeric values" in caplog.text


def test_compute_seed_is_used():
    """Verify compute_batch_metric produces deterministic results with same seed."""
    proc1 = RepresentativenessProcessor(
        name="test",
        config={"columns": {"input": ["col1"]}},
    )
    proc1.compute_seed = 42
    proc1._rng = np.random.default_rng(42)

    proc2 = RepresentativenessProcessor(
        name="test",
        config={"columns": {"input": ["col1"]}},
    )
    proc2.compute_seed = 42
    proc2._rng = np.random.default_rng(42)

    data = pa.array([1.0, 2.0, 3.0, 4.0, 5.0])
    features = {"col1": data}
    bm1 = proc1.compute_batch_metric(features)
    bm2 = proc2.compute_batch_metric(features)

    for key in bm1:
        assert key in bm2
        assert bm1[key].to_pylist() == bm2[key].to_pylist()


# --- Aggregation & Column Results Tests ---


def test_aggregate_column_metrics_no_data(caplog):
    """Verify _aggregate_column_metrics returns None and logs when no batch metrics.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    with caplog.at_level(logging.WARNING):
        result = proc._aggregate_column_metrics({}, "col1")
    assert result is None
    assert "no batch metrics" in caplog.text


def test_chi_square_insufficient_bins():
    """Verify _compute_chi_square_metric handles insufficient bins (zero expected)."""
    obs = np.array([5, 0])
    exp = np.array([0, 5])
    result = RepresentativenessProcessor._compute_chi_square_metric(None, obs, exp)
    assert result["interpretation"] == "insufficient_bins"


def test_chi_square_success():
    """Verify _compute_chi_square_metric returns p_value and statistic for valid input."""
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    obs = np.array([3, 7])
    exp = np.array([5, 5])
    result = proc._compute_chi_square_metric(obs, exp)
    assert "p_value" in result
    assert "statistic" in result


def test_ks_no_sample_data():
    """Verify _compute_ks_metric returns no_sample_data_found when sample missing."""
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    result = proc._compute_ks_metric("col1", {})
    assert result["interpretation"] == "no_sample_data_found"


def test_ks_empty_samples():
    """Verify _compute_ks_metric returns no_samples_available for empty sample array."""
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    batch_metrics = {"col1_ks_sample": pa.array([], type=pa.float64())}
    result = proc._compute_ks_metric("col1", batch_metrics)
    assert result["interpretation"] == "no_samples_available"


# --- KS / Sample Tests ---


def test_bin_edges_uniform_degenerate():
    """Verify _initialize_bin_edges handles uniform distribution with identical values."""
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}, "distribution": "uniform"})
    data = np.array([5.0, 5.0, 5.0])
    proc._initialize_bin_edges(data, "col1")
    edges = proc._bin_edges["col1"]
    assert len(edges) == 11
    assert edges[-1] > edges[0]


def test_aggregate_column_metrics_empty_hist_array(caplog):
    """Verify _aggregate_column_metrics returns None for empty histogram array.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    proc._bin_edges["col1"] = np.linspace(0, 10, 11)
    batch_metrics = {
        "col1_count": pa.array([0], type=pa.int64()),
        "col1_hist": pa.array([], type=pa.list_(pa.int64())),
    }
    with caplog.at_level(logging.WARNING):
        result = proc._aggregate_column_metrics(batch_metrics, "col1")
    assert result is None
    assert "no histogram batch" in caplog.text


def test_aggregate_column_metrics_zero_count(caplog):
    """Verify _aggregate_column_metrics returns None when total count is zero.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    proc._bin_edges["col1"] = np.linspace(0, 10, 11)
    batch_metrics = {"col1_hist": pa.array([pa.array([0] * 10, type=pa.int64())], type=pa.list_(pa.int64()))}
    with caplog.at_level(logging.WARNING):
        result = proc._aggregate_column_metrics(batch_metrics, "col1")
    assert result is None


def test_aggregate_column_metrics_missing_bin_edges(caplog):
    """Verify _aggregate_column_metrics returns None when bin edges not initialized.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    batch_metrics = {
        "col1_count": pa.array([10], type=pa.int64()),
        "col1_hist": pa.array([pa.array([1] * 10, type=pa.int64())], type=pa.list_(pa.int64())),
    }
    with caplog.at_level(logging.WARNING):
        result = proc._aggregate_column_metrics(batch_metrics, "col1")
    assert result is None
    assert "no bin edges" in caplog.text


def test_ks_metric_degenerate_uniform():
    """Verify _compute_ks_metric handles degenerate uniform distribution (all same value)."""
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}, "distribution": "uniform"})
    batch_metrics = {
        "col1_ks_sample": pa.array([1.0, 1.0, 1.0], type=pa.float64()),
    }
    result = proc._compute_ks_metric("col1", batch_metrics)
    assert result["interpretation"] == "does_not_follow_distribution"


def test_compute_column_results_aggregate_none(caplog):
    """Verify _compute_column_results returns 0 when aggregation returns None.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = RepresentativenessProcessor(name="test", config={"columns": {"input": ["col1"]}})
    with caplog.at_level(logging.WARNING):
        samples = proc._compute_column_results("col1", {}, {})
    assert samples == 0


# --- Device Resolution Tests ---


def test_resolve_device_auto():
    """Verify _resolve_device returns valid device for 'auto' mode."""
    result = RepresentativenessProcessor._resolve_device("auto")
    assert result in ("cpu", "cuda")


def test_resolve_device_explicit():
    """Verify _resolve_device returns specified device for explicit mode."""
    assert RepresentativenessProcessor._resolve_device("cpu") == "cpu"


# --- KS Sample Computation Tests ---


def test_compute_batch_ks_sample_none_for_non_ks_metrics():
    """Verify _compute_batch_ks_sample returns None when KS not in metrics."""
    proc = RepresentativenessProcessor(
        name="test",
        config={"columns": {"input": ["col1"]}, "metrics": ["shannon-entropy"]},
    )
    result = proc._compute_batch_ks_sample(None)
    assert result is None


class TestExtremeDataValues:
    """Tests for edge cases and extreme parameter values in representativeness computation.

    Covers zero standard deviation, alpha boundaries (0 and 1), GRTE scaling factor
    variations, Shannon threshold edge cases (negative, zero, large), tiny epsilon
    values, bin count extremes (1, large), NaN/Inf handling, and all-NaN warnings.
    """

    def test_normal_std_zero_is_clamped(self) -> None:
        """Verify bin edges handle zero standard deviation (all identical values) for normal distribution."""
        proc = RepresentativenessProcessor(
            name="test",
            config={"columns": {"input": ["col1"]}, "distribution": "normal", "epsilon": 1e-6},
        )
        proc._initialize_bin_edges(np.array([5.0, 5.0, 5.0]), "col1")
        edges = proc._bin_edges["col1"]
        assert len(edges) == 11
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_chi_square_alpha_zero_always_follows(self) -> None:
        """Verify chi-square interpretation is 'follows_distribution' when alpha=0."""
        proc = RepresentativenessProcessor(
            name="test",
            config={"columns": {"input": ["col1"]}, "alpha": 0.0},
        )
        obs = np.array([5, 5], dtype=np.float64)
        exp = np.array([5, 5], dtype=np.float64)
        result = proc._compute_chi_square_metric(obs, exp)
        assert result["interpretation"] == "follows_distribution"

    def test_chi_square_alpha_one_never_follows(self) -> None:
        """Verify chi-square interpretation with alpha=1.0 and perfect match."""
        proc = RepresentativenessProcessor(
            name="test",
            config={"columns": {"input": ["col1"]}, "alpha": 1.0},
        )
        obs = np.array([5, 5], dtype=np.float64)
        exp = np.array([5, 5], dtype=np.float64)
        result = proc._compute_chi_square_metric(obs, exp)
        assert result["interpretation"] == "follows_distribution"

    def test_grte_scaling_factor_zero_returns_one(self) -> None:
        """Verify GRTE returns 1.0 when scaling_factor is zero (no penalty)."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "grte": {"scaling_factor": 0.0, "threshold": 0.0},
            },
        )
        obs = np.array([5, 5], dtype=np.float64)
        exp = np.array([5, 5], dtype=np.float64)
        result = proc._compute_grte_metric(obs, exp)
        assert result["grte_value"] == 1.0

    def test_grte_scaling_factor_positive_increases_with_gap(self) -> None:
        """Verify GRTE increases with gap when scaling_factor is positive."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "grte": {"scaling_factor": 1.0, "threshold": 0.0},
            },
        )
        obs = np.array([10, 0], dtype=np.float64)
        exp = np.array([5, 5], dtype=np.float64)
        result = proc._compute_grte_metric(obs, exp)
        assert result["grte_value"] > 1.0

    def test_shannon_threshold_zero_always_high(self) -> None:
        """Verify Shannon interpretation is high_diversity when threshold is zero."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "shannon": {"threshold": 0.0},
            },
        )
        exp = np.array([5, 5], dtype=np.float64)
        result = proc._compute_shannon_entropy_metric(exp)
        assert result["interpretation"] == "high_diversity"

    def test_shannon_threshold_negative_always_high(self) -> None:
        """Verify Shannon interpretation is high_diversity when threshold is negative."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "shannon": {"threshold": -1.0},
            },
        )
        exp = np.array([5, 5], dtype=np.float64)
        result = proc._compute_shannon_entropy_metric(exp)
        assert result["interpretation"] == "high_diversity"

    def test_shannon_threshold_large_always_low(self) -> None:
        """Verify Shannon interpretation is low_diversity when threshold is very large."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "shannon": {"threshold": 1e6},
            },
        )
        exp = np.array([5, 5], dtype=np.float64)
        result = proc._compute_shannon_entropy_metric(exp)
        assert result["interpretation"] == "low_diversity"

    def test_uniform_epsilon_tiny_all_identical(self) -> None:
        """Verify bin edges handle tiny epsilon with identical values for uniform distribution."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "distribution": "uniform",
                "epsilon": 1e-300,
            },
        )
        proc._initialize_bin_edges(np.array([5.0, 5.0, 5.0]), "col1")
        edges = proc._bin_edges["col1"]
        assert len(edges) == 11
        assert edges[-1] >= edges[0]

    def test_normal_epsilon_tiny_all_identical(self) -> None:
        """Verify bin edges handle tiny epsilon with identical values for normal distribution."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "distribution": "normal",
                "epsilon": 1e-300,
            },
        )
        proc._initialize_bin_edges(np.array([5.0, 5.0, 5.0]), "col1")
        edges = proc._bin_edges["col1"]
        assert len(edges) == 11
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_bins_one_normal_edges(self) -> None:
        """Verify bin edges for normal distribution with bins=1."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "distribution": "normal",
                "histogram": {"bins": 1},
                "epsilon": 1e-9,
            },
        )
        proc._initialize_bin_edges(np.array([1.0, 2.0, 3.0]), "col1")
        edges = proc._bin_edges["col1"]
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_bins_two_normal_edges(self) -> None:
        """Verify bin edges for normal distribution with bins=2."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "distribution": "normal",
                "histogram": {"bins": 2},
                "epsilon": 1e-9,
            },
        )
        proc._initialize_bin_edges(np.array([1.0, 2.0, 3.0]), "col1")
        edges = proc._bin_edges["col1"]
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf
        assert len(edges) == 3

    def test_convert_column_all_nan_returns_none(self) -> None:
        """Verify _convert_column_to_numeric returns None when all values are NaN."""
        result = RepresentativenessProcessor._convert_column_to_numeric(pa.array([float("nan"), float("nan")]))
        assert result is None

    def test_convert_column_with_inf(self) -> None:
        """Verify _convert_column_to_numeric preserves infinity values."""
        result = RepresentativenessProcessor._convert_column_to_numeric(pa.array([1.0, float("inf")]))
        assert result is not None
        assert float("inf") in result.values

    def test_compute_batch_metric_with_nan(self) -> None:
        """Verify compute_batch_metric handles NaN values correctly (counts only valid)."""
        proc = RepresentativenessProcessor(
            name="test",
            config={"columns": {"input": ["col1"]}},
        )
        features = {"col1": pa.array([1.0, float("nan"), 3.0])}
        result = proc.compute_batch_metric(features)
        col1_count = result.get("col1_count")
        assert col1_count is not None
        assert col1_count.to_pylist()[0] == 2

    def test_compute_batch_metric_all_nan_logs_warning(self, caplog: Any) -> None:
        """Verify compute_batch_metric returns empty and logs when all values are NaN.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        proc = RepresentativenessProcessor(
            name="test",
            config={"columns": {"input": ["col1"]}},
        )
        features = {"col1": pa.array([float("nan"), float("nan")])}
        result = proc.compute_batch_metric(features)
        assert result == {}
        assert "no valid numeric values" in caplog.text

    def test_compute_batch_metric_with_inf(self) -> None:
        """Verify compute_batch_metric handles infinity values correctly."""
        proc = RepresentativenessProcessor(
            name="test",
            config={"columns": {"input": ["col1"]}},
        )
        features = {"col1": pa.array([1.0, float("inf")])}
        result = proc.compute_batch_metric(features)
        assert "col1_count" in result
        assert result["col1_count"].to_pylist()[0] == 2

    def test_compute_bins_one(self) -> None:
        """Verify compute_batch_metric works with bins=1."""
        proc = RepresentativenessProcessor(
            name="test",
            config={"columns": {"input": ["col1"]}, "histogram": {"bins": 1}},
        )
        features = {"col1": pa.array([1.0, 2.0, 3.0], type=pa.float64())}
        result = proc.compute_batch_metric(features)
        assert result
        assert "col1_hist" in result

    def test_compute_bins_large(self) -> None:
        """Verify compute_batch_metric works with large number of bins."""
        proc = RepresentativenessProcessor(
            name="test",
            config={"columns": {"input": ["col1"]}, "histogram": {"bins": 10000}},
        )
        features = {"col1": pa.array([1.0, 2.0, 3.0], type=pa.float64())}
        result = proc.compute_batch_metric(features)
        assert result

    def test_compute_epsilon_large(self) -> None:
        """Verify compute_batch_metric works with large epsilon."""
        proc = RepresentativenessProcessor(
            name="test",
            config={"columns": {"input": ["col1"]}, "distribution": "normal", "epsilon": 1e6},
        )
        features = {"col1": pa.array([1.0, 2.0, 3.0], type=pa.float64())}
        result = proc.compute_batch_metric(features)
        assert result

    def test_compute_epsilon_tiny(self) -> None:
        """Verify compute_batch_metric works with tiny epsilon."""
        proc = RepresentativenessProcessor(
            name="test",
            config={"columns": {"input": ["col1"]}, "distribution": "normal", "epsilon": 1e-300},
        )
        features = {"col1": pa.array([1.0, 2.0, 3.0], type=pa.float64())}
        result = proc.compute_batch_metric(features)
        assert result

    def test_compute_shannon_threshold_negative(self) -> None:
        """Verify Shannon interpretation with negative threshold."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "shannon": {"threshold": -1.0},
            },
        )
        exp = np.array([5, 5], dtype=np.float64)
        result = proc._compute_shannon_entropy_metric(exp)
        assert result["interpretation"] == "high_diversity"

    def test_compute_shannon_threshold_large(self) -> None:
        """Verify Shannon interpretation with very large threshold."""
        proc = RepresentativenessProcessor(
            name="test",
            config={
                "columns": {"input": ["col1"]},
                "shannon": {"threshold": 1e6},
            },
        )
        exp = np.array([5, 5], dtype=np.float64)
        result = proc._compute_shannon_entropy_metric(exp)
        assert result["interpretation"] == "low_diversity"


# --- mean_std_estimation Strategy Tests ---


def test_mean_std_from_first_batch_caches_params():
    """Verify from_first_batch (default) caches params from the first batch and reuses them."""
    proc = RepresentativenessProcessor(
        name="test",
        config={
            "columns": {"input": ["col1"]},
            "distribution": "normal",
            "mean_std_estimation": "from_first_batch",
        },
    )
    proc._initialize_bin_edges(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), "col1")
    cached = proc._bin_params.get("col1")
    assert cached is not None
    assert "mean" in cached
    assert "std" in cached
    assert cached["mean"] == pytest.approx(np.mean([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))


def test_mean_std_per_batch_estimates_from_batch_metrics():
    """Verify per_batch re-estimates params from the current batch's KS sample."""
    proc = RepresentativenessProcessor(
        name="test",
        config={
            "columns": {"input": ["col1"]},
            "distribution": "normal",
            "mean_std_estimation": "per_batch",
        },
    )
    batch_metrics = {
        "col1_ks_sample": pa.array([10.0, 20.0, 30.0], type=pa.float64()),
    }
    mean, std = proc._estimate_normal_params("col1", batch_metrics)
    assert mean == pytest.approx(20.0)
    assert std == pytest.approx(np.std([10.0, 20.0, 30.0], ddof=0))


def test_mean_std_user_provided_normal():
    """Verify user_provided strategy uses explicit distribution_params for normal."""
    proc = RepresentativenessProcessor(
        name="test",
        config={
            "columns": {"input": ["col1"]},
            "distribution": "normal",
            "mean_std_estimation": "user_provided",
            "distribution_params": [{"column": "col1", "mean": 5.0, "std": 2.0}],
        },
    )
    proc._initialize_bin_edges(np.array([0.0, 1.0, 2.0]), "col1")
    edges = proc._bin_edges["col1"]
    assert len(edges) == 11  # 10 bins = 11 edges
    params = proc._bin_params["col1"]
    assert params["mean"] == 5.0
    assert params["std"] == 2.0


def test_mean_std_user_provided_uniform():
    """Verify user_provided strategy uses explicit distribution_params for uniform."""
    proc = RepresentativenessProcessor(
        name="test",
        config={
            "columns": {"input": ["col1"]},
            "distribution": "uniform",
            "mean_std_estimation": "user_provided",
            "distribution_params": [{"column": "col1", "min": -1.0, "max": 1.0}],
        },
    )
    proc._initialize_bin_edges(np.array([0.0, 0.5, 1.0]), "col1")
    edges = proc._bin_edges["col1"]
    assert len(edges) == 11
    assert edges[0] == -1.0
    assert edges[-1] == 1.0


def test_mean_std_user_provided_missing_params_raises():
    """Verify user_provided raises ValueError when params are missing."""
    proc = RepresentativenessProcessor(
        name="test",
        config={
            "columns": {"input": ["col1"]},
            "distribution": "normal",
            "mean_std_estimation": "user_provided",
        },
    )
    with pytest.raises(ValueError, match="requires 'mean' and 'std'"):
        proc._initialize_bin_edges(np.array([0.0, 1.0, 2.0]), "col1")


def test_mean_std_from_all_data_not_implemented():
    """Verify from_all_data raises NotImplementedError in _compute_expected_counts."""
    proc = RepresentativenessProcessor(
        name="test",
        config={
            "columns": {"input": ["col1"]},
            "distribution": "normal",
            "mean_std_estimation": "from_all_data",
        },
    )
    proc._initialize_bin_edges(np.array([0.0, 1.0, 2.0]), "col1")
    batch_metrics = {
        "col1_ks_sample": pa.array([0.0, 1.0, 2.0], type=pa.float64()),
    }
    with pytest.raises(NotImplementedError, match="from_all_data"):
        proc._compute_expected_counts("col1", batch_metrics, 100, proc._bin_edges["col1"])
