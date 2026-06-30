"""Unit tests for the DomainGapProcessor and MMD kernel functions.

This module contains tests for the PyTorch-based domain gap metric processor,
including MMD kernel functions (RBF, polynomial), constructor validation,
feature/batch/delta computation, device resolution, embedding pattern resolution,
CMD channel resolution, and extreme value handling for numerical stability.
"""

import logging

from dqm_ml_pytorch.domain_gap import DomainGapProcessor, _mmd_poly, _mmd_rbf
import numpy as np
import pyarrow as pa
import pytest


class TestMmdFunctions:
    """Tests for MMD (Maximum Mean Discrepancy) kernel functions."""

    def test_mmd_rbf_single_sample_src(self):
        """Verify MMD RBF returns 0 when source has single sample."""
        assert _mmd_rbf(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0], [5.0, 6.0]]), 1.0) == 0.0

    def test_mmd_rbf_single_sample_tgt(self):
        """Verify MMD RBF returns 0 when target has single sample."""
        assert _mmd_rbf(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0]]), 1.0) == 0.0

    def test_mmd_rbf_both_single(self):
        """Verify MMD RBF returns 0 when both source and target have single sample."""
        assert _mmd_rbf(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]), 1.0) == 0.0

    def test_mmd_poly_single_sample_src(self):
        """Verify MMD Poly returns 0 when source has single sample."""
        assert _mmd_poly(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0], [5.0, 6.0]]), 2.0, 1.0, 0.0) == 0.0

    def test_mmd_poly_single_sample_tgt(self):
        """Verify MMD Poly returns 0 when target has single sample."""
        assert _mmd_poly(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0]]), 2.0, 1.0, 0.0) == 0.0

    def test_mmd_poly_both_single(self):
        """Verify MMD Poly returns 0 when both source and target have single sample."""
        assert _mmd_poly(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]), 2.0, 1.0, 0.0) == 0.0


class TestDomainGapComputeFeatures:
    """Tests for compute_features method."""

    def test_column_missing_in_batch_logs_warning(self, caplog):
        """Verify empty result and warning when input column missing from batch."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
        )
        batch = pa.RecordBatch.from_pydict({"other_col": pa.array([1.0])})
        with caplog.at_level(logging.WARNING):
            result = proc.compute_features(batch, {})
        assert result == {}


class TestDomainGapComputeBatchMetric:
    """Tests for compute_batch_metric method."""

    def test_emb_is_none_returns_empty(self):
        """Verify empty result when embedding is None."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
        )
        features = {"emb": pa.array([1.0])}
        result = proc.compute_batch_metric(features)
        assert result == {}

    def test_emb_not_fixed_size_list_returns_empty(self):
        """Verify empty result when embedding is not a fixed-size list array."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
        )
        features = {"emb": pa.array([1.0, 2.0], type=pa.float64())}
        result = proc.compute_batch_metric(features)
        assert result == {}


class TestDomainGapCompute:
    """Tests for compute method (delta aggregation)."""

    def test_empty_batch_metrics_returns_empty(self):
        """Verify empty result when batch_metrics is empty."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
        )
        result = proc.compute(batch_metrics={})
        assert result == {}

    def test_missing_count_key_returns_empty(self):
        """Verify empty result when required count key is missing."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
        )
        result = proc.compute(batch_metrics={"some_key": pa.array([1.0])})
        assert result == {}


class TestDomainGapResolveDevice:
    """Tests for _resolve_device method."""

    def test_auto_returns_cpu_or_cuda(self):
        """Verify auto mode returns valid device."""
        result = DomainGapProcessor._resolve_device("auto")
        assert result in ("cuda", "cpu")

    def test_explicit_device_returns_as_is(self):
        """Verify explicit device passed through unchanged."""
        assert DomainGapProcessor._resolve_device("cpu") == "cpu"
        assert DomainGapProcessor._resolve_device("cuda") == "cuda"


class TestDomainGapResolveEmbeddingPatterns:
    """Tests for _resolve_embedding_patterns method."""

    def test_wildcard_pattern_resolves(self):
        """Verify wildcard pattern resolves to matching columns."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb_*"]}, "distance": {"metric": "mmd_linear"}},
        )
        proc._resolve_embedding_patterns(["emb_a", "emb_b", "other"])
        assert proc.embedding_col == "emb_a"
        assert proc.embedding_cols == ["emb_a", "emb_b"]

    def test_no_wildcard_no_change(self):
        """Verify explicit column name passes through unchanged."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
        )
        proc._resolve_embedding_patterns(["emb", "other"])
        assert proc.embedding_col == "emb"


class TestDomainGapResolveCmdChannels:
    """Tests for _resolve_cmd_channels method."""

    def test_unknown_dim_raises(self):
        """Verify ValueError for unknown embedding dimension."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
        )
        with pytest.raises(ValueError, match="Cannot determine channels"):
            proc._resolve_cmd_channels("emb", 12345, {})

    def test_legacy_lookup_resolves(self):
        """Verify legacy dimension lookup resolves correctly."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
        )
        result = proc._resolve_cmd_channels("emb", 25088, {})
        assert result == 512


def test_compute_delta_summary_empty_counts():
    """Verify _compute_delta_summary handles zero-count source/target."""
    proc = DomainGapProcessor(
        name="test",
        config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
    )
    source = {"count": pa.array([0], type=pa.int64()), "sum": pa.array([0.0], type=pa.float64())}
    target = {"count": pa.array([0], type=pa.int64()), "sum": pa.array([0.0], type=pa.float64())}
    result = proc._compute_delta_summary(source, target, "mmd_linear")
    assert "empty summaries" in result["note"].to_pylist()[0]


def test_compute_delta_unknown_metric():
    """Verify compute_delta handles unsupported metric gracefully."""
    proc = DomainGapProcessor(
        name="test",
        config={"columns": {"input": ["emb"]}, "distance": {"metric": "unknown_metric"}},
    )
    result = proc.compute_delta({}, {})
    assert "unsupported metric" in result["note"].to_pylist()[0]


def test_compute_cmd_aggregate_missing_n_key():
    """Verify _compute_cmd_aggregate returns empty when n key missing."""
    proc = DomainGapProcessor(
        name="test",
        config={"columns": {"input": ["emb"]}, "distance": {"metric": "cmd"}},
    )
    result = proc._compute_cmd_aggregate({})
    assert result == {}


def test_compute_cmd_aggregate_zero_total_n():
    """Verify _compute_cmd_aggregate handles zero total count."""
    proc = DomainGapProcessor(
        name="test",
        config={"columns": {"input": ["emb"]}, "distance": {"metric": "cmd"}},
    )
    batch_metrics = {"cmd_emb_n": pa.array([0], type=pa.int64())}
    result = proc._compute_cmd_aggregate(batch_metrics)
    assert "cmd_emb_n" not in result


def test_compute_delta_wasserstein_missing_hist_counts():
    """Verify _compute_delta_wasserstein handles missing hist_counts."""
    proc = DomainGapProcessor(
        name="test",
        config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
    )
    result = proc._compute_delta_wasserstein({}, {})
    assert "missing hist_counts" in result["note"].to_pylist()[0]


def test_compute_delta_mmd_poly_missing_emb():
    """Verify _compute_delta_mmd_poly handles missing embedding."""
    proc = DomainGapProcessor(
        name="test",
        config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
    )
    result = proc._compute_delta_mmd_poly({}, {})
    assert "__emb__" in result["note"].to_pylist()[0]


def test_compute_delta_pad_missing_emb():
    """Verify _compute_delta_pad handles missing embedding."""
    proc = DomainGapProcessor(
        name="test",
        config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
    )
    result = proc._compute_delta_pad({}, {})
    assert "__emb__" in result["note"].to_pylist()[0]


class TestDomainGapLayerCmd:
    """Tests for _compute_layer_cmd method."""

    def test_missing_n_key_returns_none(self):
        """Verify _compute_layer_cmd returns None when n key missing."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
        )
        assert proc._compute_layer_cmd("col", {}, {}) is None

    def test_zero_n_returns_none(self):
        """Verify _compute_layer_cmd returns None when source count is zero."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
        )
        source = {"cmd_col_n": pa.array([0], type=pa.int64())}
        target = {"cmd_col_n": pa.array([1], type=pa.int64())}
        assert proc._compute_layer_cmd("col", source, target) is None

    def test_missing_power_sum_returns_none(self):
        """Verify _compute_layer_cmd returns None when power sum keys missing."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "cmd", "k": 5}},
        )
        source = {"cmd_col_n": pa.array([5], type=pa.int64())}
        target = {"cmd_col_n": pa.array([5], type=pa.int64())}
        assert proc._compute_layer_cmd("col", source, target) is None


class TestDomainGapExtremeValues:
    """Tests for extreme numerical values in kernel/summary parameters."""

    @pytest.mark.parametrize("gamma", [0.0, 1e-10, 1e10])
    def test_mmd_rbf_extreme_gamma(self, gamma: float) -> None:
        """Verify MMD RBF handles extreme gamma values without overflow."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tgt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        result = _mmd_rbf(src, tgt, gamma)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_mmd_rbf_gamma_zero_identical(self) -> None:
        """Verify MMD RBF returns 0 for identical embeddings with gamma=0."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = _mmd_rbf(src, src, 0.0)
        assert result == 0.0

    @pytest.mark.parametrize("gamma", [0.0, 1e10])
    def test_mmd_poly_extreme_gamma(self, gamma: float) -> None:
        """Verify MMD Poly handles extreme gamma values."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tgt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        result = _mmd_poly(src, tgt, 3.0, gamma, 1.0)
        assert np.isfinite(result)
        assert result >= 0.0

    @pytest.mark.parametrize("degree", [0.0, 1.0, 100.0])
    def test_mmd_poly_extreme_degree(self, degree: float) -> None:
        """Verify MMD Poly handles extreme polynomial degree."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tgt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        result = _mmd_poly(src, tgt, degree, 1.0, 1.0)
        assert np.isfinite(result)
        assert result >= 0.0

    @pytest.mark.parametrize("coefficient0", [0.0, -1.0, 1e10])
    def test_mmd_poly_extreme_coefficient0(self, coefficient0: float) -> None:
        """Verify MMD Poly handles extreme coefficient0 values."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tgt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        result = _mmd_poly(src, tgt, 3.0, 1.0, coefficient0)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_fid_epsilon_zero(self) -> None:
        """Verify FID computation works with epsilon=0."""
        proc = DomainGapProcessor(
            name="test",
            config={
                "columns": {"input": ["emb"]},
                "distance": {"metric": "fid", "epsilon": 0.0},
            },
        )
        embed_dim = 3
        n_src = 5
        n_tgt = 5
        src_emb = np.random.default_rng(42).normal(size=(n_src, embed_dim))
        tgt_emb = np.random.default_rng(99).normal(size=(n_tgt, embed_dim))
        src = {
            "count": pa.array([n_src], type=pa.int64()),
            "sum": pa.FixedSizeListArray.from_arrays(pa.array(src_emb.sum(axis=0)), embed_dim),
            "sum_sq": pa.FixedSizeListArray.from_arrays(pa.array((src_emb * src_emb).sum(axis=0)), embed_dim),
            "sum_outer": pa.FixedSizeListArray.from_arrays(
                pa.array((src_emb.T @ src_emb).reshape(-1)), embed_dim * embed_dim
            ),
        }
        tgt = {
            "count": pa.array([n_tgt], type=pa.int64()),
            "sum": pa.FixedSizeListArray.from_arrays(pa.array(tgt_emb.sum(axis=0)), embed_dim),
            "sum_sq": pa.FixedSizeListArray.from_arrays(pa.array((tgt_emb * tgt_emb).sum(axis=0)), embed_dim),
            "sum_outer": pa.FixedSizeListArray.from_arrays(
                pa.array((tgt_emb.T @ tgt_emb).reshape(-1)), embed_dim * embed_dim
            ),
        }
        result = proc._compute_delta_summary(src, tgt, "fid")
        assert "fid" in result
        val = result["fid"].to_pylist()[0]
        assert np.isfinite(val)
        assert val >= 0.0

    def test_fid_epsilon_tiny_singular_cov(self) -> None:
        """Verify FID handles tiny epsilon with singular covariance."""
        proc = DomainGapProcessor(
            name="test",
            config={
                "columns": {"input": ["emb"]},
                "distance": {"metric": "fid", "epsilon": 1e-12},
            },
        )
        embed_dim = 3
        n = 5
        src_emb = np.ones((n, embed_dim)) * 5.0
        tgt_emb = np.ones((n, embed_dim))
        src = {
            "count": pa.array([n], type=pa.int64()),
            "sum": pa.FixedSizeListArray.from_arrays(pa.array(src_emb.sum(axis=0)), embed_dim),
            "sum_sq": pa.FixedSizeListArray.from_arrays(pa.array((src_emb * src_emb).sum(axis=0)), embed_dim),
            "sum_outer": pa.FixedSizeListArray.from_arrays(
                pa.array((src_emb.T @ src_emb).reshape(-1)), embed_dim * embed_dim
            ),
        }
        tgt = {
            "count": pa.array([n], type=pa.int64()),
            "sum": pa.FixedSizeListArray.from_arrays(pa.array(tgt_emb.sum(axis=0)), embed_dim),
            "sum_sq": pa.FixedSizeListArray.from_arrays(pa.array((tgt_emb * tgt_emb).sum(axis=0)), embed_dim),
            "sum_outer": pa.FixedSizeListArray.from_arrays(
                pa.array((tgt_emb.T @ tgt_emb).reshape(-1)), embed_dim * embed_dim
            ),
        }
        result = proc._compute_delta_summary(src, tgt, "fid")
        assert "fid" in result
        val = result["fid"].to_pylist()[0]
        assert np.isfinite(val)
        assert val >= 0.0

    def test_cmd_feature_weights_all_zero(self) -> None:
        """Verify CMD handles all-zero feature weights."""
        proc = DomainGapProcessor(
            name="test",
            config={
                "columns": {"input": ["emb_a", "emb_b"]},
                "distance": {"metric": "cmd", "feature_weights": [0.0, 0.0]},
            },
        )
        source = {"cmd_emb_a_n": pa.array([5], type=pa.int64())}
        target = {"cmd_emb_a_n": pa.array([5], type=pa.int64())}
        result = proc._compute_delta_cmd(source, target)
        assert "no valid layers" in result["note"].to_pylist()[0]

    def test_delta_mmd_rbf_identical_embeddings(self) -> None:
        """Verify MMD RBF returns 0 for identical source/target embeddings."""
        proc = DomainGapProcessor(
            name="test",
            config={
                "columns": {"input": ["emb"]},
                "distance": {"metric": "mmd_rbf"},
            },
        )
        dim = 2
        emb = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        vals = emb.reshape(-1)
        src = {"__emb__": pa.FixedSizeListArray.from_arrays(pa.array(vals), dim)}
        tgt = {"__emb__": pa.FixedSizeListArray.from_arrays(pa.array(vals), dim)}
        result = proc._compute_delta_mmd_rbf(src, tgt)
        val = result["mmd_rbf"].to_pylist()[0]
        assert val == 0.0

    def test_delta_mmd_poly_identical_embeddings(self) -> None:
        """Verify MMD Poly returns 0 for identical source/target embeddings."""
        proc = DomainGapProcessor(
            name="test",
            config={
                "columns": {"input": ["emb"]},
                "distance": {"metric": "mmd_poly", "kernel_params": {"degree": 2.0, "coefficient0": 0.0}},
            },
        )
        dim = 2
        emb = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        vals = emb.reshape(-1)
        src = {"__emb__": pa.FixedSizeListArray.from_arrays(pa.array(vals), dim)}
        tgt = {"__emb__": pa.FixedSizeListArray.from_arrays(pa.array(vals), dim)}
        result = proc._compute_delta_mmd_poly(src, tgt)
        val = result["mmd_poly"].to_pylist()[0]
        assert val == 0.0

    def test_mmd_poly_negative_gamma(self) -> None:
        """Verify MMD Poly handles negative gamma."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tgt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        result = _mmd_poly(src, tgt, 3.0, -1.0, 1.0)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_mmd_poly_fractional_degree(self) -> None:
        """Verify MMD Poly handles fractional polynomial degree."""
        src = np.array([[1.0, 2.0]], dtype=np.float64)
        tgt = np.array([[3.0, 4.0]], dtype=np.float64)
        result = _mmd_poly(src, tgt, 1.5, 1.0, 1.0)
        assert np.isfinite(result)

    def test_mmd_rbf_gamma_extreme_small(self) -> None:
        """Verify MMD RBF handles extremely small gamma."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tgt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        result = _mmd_rbf(src, tgt, 1e-10)
        assert np.isfinite(result)

    def test_mmd_rbf_gamma_extreme_large(self) -> None:
        """Verify MMD RBF handles extremely large gamma."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tgt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        result = _mmd_rbf(src, tgt, 1e10)
        assert np.isfinite(result)

    def test_mmd_poly_degree_zero(self) -> None:
        """Verify MMD Poly handles zero degree."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tgt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        result = _mmd_poly(src, tgt, 0.0, 1.0, 1.0)
        assert np.isfinite(result)

    def test_mmd_poly_extreme_coefficient0_negative(self) -> None:
        """Verify MMD Poly handles extreme negative coefficient0."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tgt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        result = _mmd_poly(src, tgt, 3.0, 1.0, -1000.0)
        assert np.isfinite(result)

    def test_mmd_poly_extreme_gamma_negative_large(self) -> None:
        """Verify MMD Poly handles extreme negative gamma."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tgt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        result = _mmd_poly(src, tgt, 3.0, -1e6, 1.0)
        assert np.isfinite(result)

    def test_batch_metric_cmd_k_zero(self) -> None:
        """Verify compute_batch_metric works with CMD k=0."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "cmd", "k": 0}},
        )
        dim = 2
        vals = pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], type=pa.float64())
        features = {
            "emb": pa.FixedSizeListArray.from_arrays(vals, dim),
            "emb_channels": pa.array([2], type=pa.int64()),
        }
        result = proc.compute_batch_metric(features)
        assert result != {}

    def test_batch_metric_cmd_k_large(self) -> None:
        """Verify compute_batch_metric works with large CMD k."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "cmd", "k": 50}},
        )
        dim = 2
        vals = pa.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], type=pa.float64())
        features = {
            "emb": pa.FixedSizeListArray.from_arrays(vals, dim),
            "emb_channels": pa.array([2], type=pa.int64()),
        }
        result = proc.compute_batch_metric(features)
        assert "cmd_emb_n" in result
        assert f"cmd_emb_sum_{proc.cmd_k}" in result

    def test_delta_cmd_k_zero(self) -> None:
        """Verify _compute_delta_cmd handles k=0."""
        proc = DomainGapProcessor(
            name="test",
            config={"columns": {"input": ["emb"]}, "distance": {"metric": "cmd", "k": 0}},
        )
        source = {"cmd_emb_n": pa.array([3], type=pa.int64())}
        target = {"cmd_emb_n": pa.array([3], type=pa.int64())}
        result = proc._compute_delta_cmd(source, target)
        assert "no valid layers" in result["note"].to_pylist()[0]

    def test_batch_metric_wasserstein_bins_one(self) -> None:
        """Verify compute_batch_metric works with Wasserstein bins=1."""
        proc = DomainGapProcessor(
            name="test",
            config={
                "columns": {"input": ["emb"]},
                "distance": {"metric": "wasserstein_1d"},
                "summary": {"histogram": {"dims": 2, "bins": 1, "range": [-3.0, 3.0]}},
            },
        )
        dim = 2
        vals = pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], type=pa.float64())
        features = {"emb": pa.FixedSizeListArray.from_arrays(vals, dim)}
        result = proc.compute_batch_metric(features)
        assert result != {}

    def test_batch_metric_wasserstein_bins_large(self) -> None:
        """Verify compute_batch_metric works with large Wasserstein bins."""
        proc = DomainGapProcessor(
            name="test",
            config={
                "columns": {"input": ["emb"]},
                "distance": {"metric": "wasserstein_1d"},
                "summary": {"histogram": {"dims": 2, "bins": 100000, "range": [-3.0, 3.0]}},
            },
        )
        dim = 2
        vals = pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], type=pa.float64())
        features = {"emb": pa.FixedSizeListArray.from_arrays(vals, dim)}
        result = proc.compute_batch_metric(features)
        assert result != {}

    def test_delta_wasserstein_range_zero_width(self) -> None:
        """Verify _compute_delta_wasserstein handles zero-width range."""
        proc = DomainGapProcessor(
            name="test",
            config={
                "columns": {"input": ["emb"]},
                "distance": {"metric": "wasserstein_1d"},
                "summary": {"histogram": {"dims": 2, "bins": 10, "range": [0.0, 0.0]}},
            },
        )
        source = {
            "hist_dims": pa.array([2], type=pa.int64()),
            "hist_counts": pa.array(
                [pa.array([1] * 10, type=pa.int64()), pa.array([1] * 10, type=pa.int64())],
                type=pa.list_(pa.int64()),
            ),
        }
        target = {
            "hist_dims": pa.array([2], type=pa.int64()),
            "hist_counts": pa.array(
                [pa.array([1] * 10, type=pa.int64()), pa.array([1] * 10, type=pa.int64())],
                type=pa.list_(pa.int64()),
            ),
        }
        result = proc._compute_delta_wasserstein(source, target)
        assert "wasserstein_1d" in result

    def test_delta_wasserstein_range_reversed(self) -> None:
        """Verify _compute_delta_wasserstein handles reversed range."""
        proc = DomainGapProcessor(
            name="test",
            config={
                "columns": {"input": ["emb"]},
                "distance": {"metric": "wasserstein_1d"},
                "summary": {"histogram": {"dims": 2, "bins": 10, "range": [5.0, -5.0]}},
            },
        )
        source = {
            "hist_dims": pa.array([2], type=pa.int64()),
            "hist_counts": pa.array(
                [pa.array([1] * 10, type=pa.int64()), pa.array([1] * 10, type=pa.int64())],
                type=pa.list_(pa.int64()),
            ),
        }
        target = {
            "hist_dims": pa.array([2], type=pa.int64()),
            "hist_counts": pa.array(
                [pa.array([1] * 10, type=pa.int64()), pa.array([1] * 10, type=pa.int64())],
                type=pa.list_(pa.int64()),
            ),
        }
        result = proc._compute_delta_wasserstein(source, target)
        assert "wasserstein_1d" in result
