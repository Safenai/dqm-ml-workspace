"""Unit tests for the DatasetJob pipeline orchestration.

This module contains tests for the DatasetJob class internal methods including
interface routing, topological sorting, type conversion, memory parsing,
dataloader column injection, dependency graph building, debug logging,
processor column analysis, path prefix injection, and feature flushing.
"""

import logging
from unittest.mock import MagicMock

from dqm_ml_core.api.features_processor import FeaturesProcessor
from dqm_ml_core.api.gap_processor import GapProcessor
from dqm_ml_core.api.metrics_processor import MetricsProcessor
from dqm_ml_core.api.processor import Processor
from dqm_ml_job.job import DatasetJob
import numpy as np
import pyarrow as pa
import pytest


class TestGetInterfaceForProcessor:
    """Tests for _get_interface_for_processor method."""

    def test_known_features(self):
        """Verify feature processors map to 'features' interface."""
        mock_proc = MagicMock(spec=FeaturesProcessor)
        assert DatasetJob._get_interface_for_processor(mock_proc) == "features"

    def test_known_metrics(self):
        """Verify metric processors map to 'metrics' interface."""
        mock_proc = MagicMock(spec=MetricsProcessor)
        assert DatasetJob._get_interface_for_processor(mock_proc) == "metrics"

    def test_known_gap(self):
        """Verify gap processors map to 'gap' interface."""
        mock_proc = MagicMock(spec=GapProcessor)
        assert DatasetJob._get_interface_for_processor(mock_proc) == "gap"

    def test_unknown(self):
        """Verify unknown processor returns None."""
        mock_proc = MagicMock(spec=Processor)
        assert DatasetJob._get_interface_for_processor(mock_proc) is None


class TestTopologicalSort:
    """Tests for _topological_sort method."""

    def test_single_processor(self):
        """Verify single processor returns as-is."""
        mock = MagicMock()
        ordered = DatasetJob._topological_sort([mock], [set()])
        assert ordered == [mock]

    def test_no_dependencies(self):
        """Verify processors with no dependencies maintain order."""
        m1, m2 = MagicMock(), MagicMock()
        ordered = DatasetJob._topological_sort([m1, m2], [set(), set()])
        assert ordered == [m1, m2]

    def test_circular_uses_min_remaining(self):
        """Verify circular dependency falls back to min remaining strategy."""
        m1, m2 = MagicMock(), MagicMock()
        dep_on = [{1}, {0}]  # circular
        ordered = DatasetJob._topological_sort([m1, m2], dep_on)
        assert len(ordered) == 2


class TestToPaArray:
    """Tests for _to_pa_array static method."""

    def test_int(self):
        """Verify integer converts to float64 array."""
        result = DatasetJob._to_pa_array(42, "metric")
        assert result.type == pa.float64()
        assert result.to_pylist() == [42.0]

    def test_float(self):
        """Verify float converts to array."""
        result = DatasetJob._to_pa_array(3.14, "metric")
        assert result.to_pylist() == [3.14]

    def test_str(self):
        """Verify string converts to array."""
        result = DatasetJob._to_pa_array("hello", "metric")
        assert result.to_pylist() == ["hello"]

    def test_ndarray(self):
        """Verify numpy array converts to list array."""
        result = DatasetJob._to_pa_array(np.array([1, 2, 3]), "metric")
        assert result.to_pylist() == [[1, 2, 3]]

    def test_pa_array(self):
        """Verify PyArrow array passes through unchanged."""
        arr = pa.array([1.0, 2.0])
        result = DatasetJob._to_pa_array(arr, "metric")
        assert result == arr

    def test_unsupported_type(self):
        """Verify unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported delta metric type"):
            DatasetJob._to_pa_array({"a": 1}, "metric")


class TestParseMemoryString:
    """Tests for _parse_memory_string method."""

    def test_gb(self):
        """Verify GB suffix parses correctly."""
        job = DatasetJob(dataloaders={})
        assert job._parse_memory_string("2GB") == 2 * 1024 * 1024 * 1024

    def test_mb(self):
        """Verify MB suffix parses correctly."""
        job = DatasetJob(dataloaders={})
        assert job._parse_memory_string("500MB") == 500 * 1024 * 1024

    def test_kb(self):
        """Verify KB suffix parses correctly."""
        job = DatasetJob(dataloaders={})
        assert job._parse_memory_string("128KB") == 128 * 1024

    def test_bytes(self):
        """Verify B suffix parses correctly."""
        job = DatasetJob(dataloaders={})
        assert job._parse_memory_string("1024B") == 1024

    def test_raw_bytes(self):
        """Verify raw number parses as bytes."""
        job = DatasetJob(dataloaders={})
        assert job._parse_memory_string("4096") == 4096

    def test_lowercase(self):
        """Verify case-insensitive parsing."""
        job = DatasetJob(dataloaders={})
        assert job._parse_memory_string("1gb") == 1024 * 1024 * 1024


class TestInjectDataloaderColumn:
    """Tests for _inject_dataloader_column method."""

    def test_no_features_output(self):
        """Verify no column added when features_output is None."""
        job = DatasetJob(dataloaders={})
        features = {"col1": pa.array([1, 2])}
        job._inject_dataloader_column("selection", features)
        assert "dataloader" not in features

    def test_add_dataloader_column_disabled(self):
        """Verify no column added when add_dataloader_column is False."""
        writer = MagicMock()
        writer.add_dataloader_column = False
        job = DatasetJob(dataloaders={}, features_output=writer)
        features = {"col1": pa.array([1, 2])}
        job._inject_dataloader_column("selection", features)
        assert "dataloader" not in features

    def test_add_dataloader_column_empty_features(self):
        """Verify no column added when features dict is empty."""
        writer = MagicMock()
        writer.add_dataloader_column = True
        job = DatasetJob(dataloaders={}, features_output=writer)
        features = {}
        job._inject_dataloader_column("selection", features)
        assert features == {}

    def test_add_dataloader_column_success(self):
        """Verify dataloader column added with correct values."""
        writer = MagicMock()
        writer.add_dataloader_column = True
        writer.dataloader_column_name = "dataloader"
        job = DatasetJob(dataloaders={}, features_output=writer)
        features = {"col1": pa.array([1, 2])}
        job._inject_dataloader_column("selection", features)
        assert "dataloader" in features
        assert features["dataloader"].to_pylist() == ["selection", "selection"]


class TestBuildDependencyGraph:
    """Tests for _build_dependency_graph method."""

    def test_no_processors(self):
        """Verify empty processor list returns empty dependency graph."""
        dep_on = DatasetJob._build_dependency_graph([])
        assert dep_on == []

    def test_no_generated_columns(self):
        """Verify processor with no generated features has no dependencies."""
        mock_proc = MagicMock()
        mock_proc.generated_features.return_value = []
        dep_on = DatasetJob._build_dependency_graph([mock_proc])
        assert dep_on == [set()]


class TestRegisterGeneratedColumns:
    """Tests for _register_generated_columns method."""

    def test_no_processors(self):
        """Verify empty processor list returns empty dict."""
        result = DatasetJob._register_generated_columns([])
        assert result == {}

    def test_with_generated_features(self):
        """Verify generated features mapped to processor index."""
        mock_proc = MagicMock()
        mock_proc.generated_features.return_value = ["feat1", "feat2"]
        result = DatasetJob._register_generated_columns([mock_proc])
        assert result["feat1"] == {0}
        assert result["feat2"] == {0}


class TestDebugLogging:
    """Tests for debug logging in metric computation methods."""

    def test_compute_selection_metrics_debug_logging(self, caplog) -> None:
        """Verify debug logging includes metric names and available metrics.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        job = DatasetJob(dataloaders={})
        mock_metric = MagicMock()
        mock_metric.__class__.__name__ = "MockMetric"
        mock_metric.compute.return_value = {"mock": pa.array([1.0])}
        with caplog.at_level(logging.DEBUG):
            job._compute_selection_metrics("sel1", {}, [mock_metric])
        assert "Metric computation" in caplog.text
        assert "Available metrics" in caplog.text

    def test_process_batch_debug_logging(self, caplog) -> None:
        """Verify debug logging for batch processing shows metrics and features.

        Args:
            caplog: Pytest fixture to capture log output.
        """
        mock_metric = MagicMock(spec=MetricsProcessor)
        mock_metric.name = "test_metric"
        mock_metric.extract_columns.return_value = {"feat": pa.array([1.0])}
        mock_metric.compute_batch_metric.return_value = {"bm": pa.array([1.0])}
        batch = pa.RecordBatch.from_pydict({"col": pa.array([1.0])})
        with caplog.at_level(logging.DEBUG):
            DatasetJob._process_batch(batch, [mock_metric], [mock_metric], [])
        assert "Available batch_metrics" in caplog.text
        assert "features" in caplog.text


class TestAnalyzeProcessorColumns:
    """Tests for _analyze_processor_columns method."""

    def test_wildcard_columns_resets_needed_input(self):
        """Verify wildcard column resets needed_input_columns to empty."""
        mock_metric = MagicMock(spec=MetricsProcessor)
        mock_metric.needed_columns.return_value = ["*"]
        mock_metric.generated_metrics.return_value = []
        job = DatasetJob(
            dataloaders={},
            metrics_processors={"mock": mock_metric},
        )
        job._analyze_processor_columns()
        assert job.needed_input_columns == []


class TestMaybeFlushFeatures:
    """Tests for _maybe_flush_features method."""

    def test_flush_when_threshold_exceeded(self):
        """Verify flush occurs when feature array size exceeds threshold."""
        writer = MagicMock()
        writer.add_dataloader_column = False
        job = DatasetJob(dataloaders={}, features_output=writer)
        features_accumulator = {"col1": [pa.array([1.0]), pa.array([2.0])]}
        result = job._maybe_flush_features(
            "sel1",
            features_accumulator,
            feature_array_size=100,
            part_index=0,
            memory_threshold=50,
        )
        assert result == 1
        assert features_accumulator == {}

    def test_no_flush_when_threshold_not_exceeded(self):
        """Verify no flush when feature array size is below threshold."""
        job = DatasetJob(dataloaders={})
        features_accumulator = {"col1": [pa.array([1.0])]}
        result = job._maybe_flush_features(
            "sel1",
            features_accumulator,
            feature_array_size=10,
            part_index=0,
            memory_threshold=100,
        )
        assert result == 0
        assert "col1" in features_accumulator
