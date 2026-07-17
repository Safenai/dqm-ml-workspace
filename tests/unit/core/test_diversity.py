"""Unit tests for the diversity metric processor.

This module contains tests that verify the diversity metric processor
correctly computes diversity indices (richness, Shannon, Gini, Simpson)
for categorical data, including handling of missing features, empty values,
and edge cases with extreme distributions.
"""

import logging

from dqm_ml_core.metrics.diversity import DiversityProcessor
import pyarrow as pa
import pytest


def test_compute_empty_batch_metrics():
    """Verify compute returns error metadata when batch_metrics is None."""
    proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
    result = proc.compute(batch_metrics=None)
    assert result == {"_metadata": {"error": "No batch metrics provided"}}


def test_compute_batch_metric_missing_feature(caplog):
    """Verify compute_batch_metric returns empty dict and logs warning for missing feature.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
    features = {}  # col1 not in features

    with caplog.at_level(logging.WARNING):
        batch_metrics = proc.compute_batch_metric(features)

    assert batch_metrics == {}
    assert "column 'col1' not found in batch" in caplog.text


def test_compute_batch_metric_empty_values():
    """Verify compute_batch_metric returns empty dict for empty feature array."""
    proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
    features = {"col1": pa.array([], type=pa.int64())}
    batch_metrics = proc.compute_batch_metric(features)

    assert batch_metrics == {}


def test_compute_column_diversity_no_batch_metrics(caplog):
    """Verify _compute_column_diversity returns empty dict and logs when no batch metrics.

    Args:
        caplog: Pytest fixture to capture log output.
    """
    proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
    result = proc._compute_column_diversity("col1", {})

    assert result == {}
    assert "no batch metrics" in caplog.text


def test_compute_column_diversity_empty_values():
    """Verify _compute_column_diversity returns empty dict for empty values/counts arrays."""
    proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
    batch_metrics = {
        "col1_values": pa.array([], type=pa.string()),
        "col1_counts": pa.array([], type=pa.int64()),
    }
    result = proc._compute_column_diversity("col1", batch_metrics)
    assert result == {}


def test_compute_column_diversity_total_zero():
    """Verify _compute_column_diversity returns empty dict when total count is zero."""
    proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
    batch_metrics = {
        "col1_values": pa.array(["a"]),
        "col1_counts": pa.array([0]),
    }
    result = proc._compute_column_diversity("col1", batch_metrics)
    assert result == {}


def test_compute_success():
    """Verify compute returns all four diversity indices for valid batch metrics."""
    proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
    batch_metrics = {
        "col1_values": pa.array(["a", "b"]),
        "col1_counts": pa.array([3, 1]),
    }
    result = proc.compute(batch_metrics=batch_metrics)

    assert "col1_richness" in result
    assert result["col1_richness"] == 2
    assert "col1_shannon" in result
    assert "col1_gini" in result
    assert "col1_simpson" in result


class TestExtremeDataValues:
    """Tests for edge cases and extreme data distributions in diversity computation.

    Covers single category, single sample, extremely skewed distributions,
    and all-unique cases for maximum diversity verification.
    """

    def test_single_category(self) -> None:
        """Verify diversity indices for single category (zero diversity)."""
        proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
        batch_metrics = {
            "col1_values": pa.array(["a"]),
            "col1_counts": pa.array([10]),
        }
        result = proc.compute(batch_metrics=batch_metrics)
        assert result["col1_richness"] == 1
        assert result["col1_shannon"] == pytest.approx(0.0)
        assert result["col1_gini"] == pytest.approx(0.0)
        assert result["col1_simpson"] == pytest.approx(0.0)

    def test_single_sample_total(self) -> None:
        """Verify diversity indices when only one sample exists."""
        proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
        batch_metrics = {
            "col1_values": pa.array(["a"]),
            "col1_counts": pa.array([1]),
        }
        result = proc.compute(batch_metrics=batch_metrics)
        assert result["col1_richness"] == 1
        assert result["col1_simpson"] == pytest.approx(0.0)

    def test_extremely_skewed(self) -> None:
        """Verify diversity indices for extremely skewed distribution (999:1)."""
        proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
        batch_metrics = {
            "col1_values": pa.array(["a", "b"]),
            "col1_counts": pa.array([999, 1]),
        }
        result = proc.compute(batch_metrics=batch_metrics)
        assert result["col1_richness"] == 2
        assert result["col1_simpson"] < 0.01
        assert result["col1_gini"] < 0.01

    def test_all_unique(self) -> None:
        """Verify diversity indices when all values are unique (maximum diversity)."""
        proc = DiversityProcessor(name="test", config={"columns": {"input": ["col1"]}})
        batch_metrics = {
            "col1_values": pa.array(["a", "b", "c", "d", "e"]),
            "col1_counts": pa.array([1, 1, 1, 1, 1]),
        }
        result = proc.compute(batch_metrics=batch_metrics)
        assert result["col1_richness"] == 5
        assert result["col1_simpson"] == pytest.approx(1.0)
        assert result["col1_gini"] == pytest.approx(0.8)
