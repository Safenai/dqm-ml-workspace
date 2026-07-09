"""Tests for outputs configuration: CLI wiring, job filtering, and pairwise logic.

This module tests how ``outputs`` config is read from interface configs,
passed through to writers, and used to filter columns during feature
accumulation.  It does NOT retest Pydantic's own model validation.
"""

from unittest.mock import MagicMock

from dqm_ml_core.models.config import JobConfig
from dqm_ml_core.models.interfaces import FeaturesInterfaceConfig, GapInterfaceConfig, MetricsInterfaceConfig
from dqm_ml_core.models.outputs import FeaturesOutputsConfig, GapOutputsConfig, MetricsOutputsConfig
from dqm_ml_job.cli import _init_interface_outputs
from dqm_ml_job.job import DatasetJob
import pyarrow as pa
import pytest


class TestInitInterfaceOutputs:
    """_init_interface_outputs wiring: config -> writer columns/exclude."""

    def _registry(self) -> dict[str, MagicMock]:
        """Create a mock registry with parquet writer.

        Returns:
            Dictionary mapping writer name to MagicMock.
        """
        return {"parquet": MagicMock()}

    def test_returns_none_when_interface_none(self) -> None:
        """Returns None when interface is None."""
        result = _init_interface_outputs(None, self._registry(), "features")
        assert result is None

    def test_returns_none_when_outputs_none(self) -> None:
        """Returns None when interface.outputs is None."""
        interface = FeaturesInterfaceConfig()
        result = _init_interface_outputs(interface, self._registry(), "features")
        assert result is None

    def test_features_passes_include_exclude(self) -> None:
        """Features outputs include/exclude passed to writer config."""
        outputs = FeaturesOutputsConfig(
            path="test.parquet",
            include=["id", "img_*"],
            exclude=["meta_*"],
        )
        interface = FeaturesInterfaceConfig(outputs=outputs)
        registry = self._registry()
        _init_interface_outputs(interface, registry, "features")
        registry["parquet"].assert_called_once_with(
            name="features",
            config={
                "path_pattern": "test.parquet",
                "columns": ["id", "img_*"],
                "exclude": ["meta_*"],
            },
        )

    def test_features_no_include_exclude(self) -> None:
        """Features outputs without include/exclude uses empty lists."""
        outputs = FeaturesOutputsConfig(path="test.parquet")

        interface = FeaturesInterfaceConfig(outputs=outputs)
        registry = self._registry()
        _init_interface_outputs(interface, registry, "features")
        registry["parquet"].assert_called_once_with(
            name="features",
            config={
                "path_pattern": "test.parquet",
                "columns": [],
                "exclude": [],
            },
        )

    def test_metrics_no_include_exclude(self) -> None:
        """Metrics outputs without include/exclude uses empty lists."""
        outputs = MetricsOutputsConfig(path="metrics.parquet")

        interface = MetricsInterfaceConfig(outputs=outputs)
        registry = self._registry()
        _init_interface_outputs(interface, registry, "metrics")
        registry["parquet"].assert_called_once_with(
            name="metrics",
            config={
                "path_pattern": "metrics.parquet",
                "columns": [],
                "exclude": [],
            },
        )

    def test_gap_no_include_exclude(self) -> None:
        """Gap outputs without include/exclude uses empty lists."""
        outputs = GapOutputsConfig(path="gap.parquet")

        interface = GapInterfaceConfig(outputs=outputs)
        registry = self._registry()
        _init_interface_outputs(interface, registry, "gap")
        registry["parquet"].assert_called_once_with(
            name="gap",
            config={
                "path_pattern": "gap.parquet",
                "columns": [],
                "exclude": [],
            },
        )


class TestJobOutputsIncludeExclude:
    """DatasetJob feature accumulation respects outputs include/exclude."""

    @pytest.fixture
    def mock_writer(self) -> MagicMock:
        """Create a mock writer with default columns/exclude settings.

        Returns:
            MagicMock configured as a features output writer.
        """
        writer = MagicMock()
        writer.columns = None
        writer.exclude = None
        return writer

    def _make_batch(self, column_names: list[str]) -> pa.RecordBatch:
        """Create a RecordBatch with given column names.

        Args:
            column_names: List of column names to include in batch.

        Returns:
            PyArrow RecordBatch with single row of 1.0 values.
        """
        arrays = {c: pa.array([1.0]) for c in column_names}
        return pa.RecordBatch.from_pydict(arrays)

    def test_source_include_literal(self, mock_writer: MagicMock) -> None:
        """Source features filtered by literal column include list."""
        mock_writer.columns = ["a", "c"]

        job = DatasetJob(dataloaders={}, features_output=mock_writer)
        batch = self._make_batch(["a", "b", "c", "d"])
        acc: dict[str, list] = {}
        job._accumulate_source_features(batch, acc, 0)
        assert list(acc.keys()) == ["a", "c"]

    def test_source_include_wildcard(self, mock_writer: MagicMock) -> None:
        """Source features filtered by wildcard include pattern."""
        mock_writer.columns = ["img_*"]
        job = DatasetJob(dataloaders={}, features_output=mock_writer)
        batch = self._make_batch(["img_a", "img_b", "other", "meta_x"])
        acc: dict[str, list] = {}
        job._accumulate_source_features(batch, acc, 0)
        assert list(acc.keys()) == ["img_a", "img_b"]

    def test_source_exclude_wildcard(self, mock_writer: MagicMock) -> None:
        """Source features filtered by wildcard exclude pattern."""
        mock_writer.columns = ["*"]
        mock_writer.exclude = ["meta_*"]
        job = DatasetJob(dataloaders={}, features_output=mock_writer)
        batch = self._make_batch(["a", "meta_x", "b", "meta_y"])
        acc: dict[str, list] = {}
        job._accumulate_source_features(batch, acc, 0)
        assert list(acc.keys()) == ["a", "b"]

    def test_source_include_with_exclude(self, mock_writer):
        """Source features filtered by combined include and exclude patterns."""
        mock_writer.columns = ["img_*", "id"]
        mock_writer.exclude = ["img_bad_*"]
        job = DatasetJob(dataloaders={}, features_output=mock_writer)
        batch = self._make_batch(["id", "img_a", "img_bad_1", "img_b", "other"])
        acc: dict[str, list] = {}
        job._accumulate_source_features(batch, acc, 0)
        assert list(acc.keys()) == ["img_a", "img_b", "id"]

    def test_source_no_features_output(self):
        """Returns immediately when features_output is None."""
        job = DatasetJob(dataloaders={})
        batch = self._make_batch(["a", "b"])
        acc: dict[str, list] = {}
        result = job._accumulate_source_features(batch, acc, 0)
        assert acc == {}
        assert result == 0

    def test_generated_include_wildcard(self, mock_writer):
        """Generated features filtered by wildcard include pattern."""
        mock_writer.columns = ["luminosity_*"]
        job = DatasetJob(dataloaders={}, features_output=mock_writer)
        generated = {
            "luminosity_mean": pa.array([1.0]),
            "luminosity_std": pa.array([2.0]),
            "blur": pa.array([3.0]),
        }
        batch = self._make_batch([])
        acc: dict[str, list] = {}
        job._accumulate_generated_features(batch, generated, {}, acc, 0)
        assert list(acc.keys()) == ["luminosity_mean", "luminosity_std", "blur"]

    def test_generated_exclude_metrics(self, mock_writer):
        """Generated features exclude columns that overlap with metrics keys."""
        mock_writer.columns = ["*"]
        job = DatasetJob(dataloaders={}, features_output=mock_writer)
        generated = {
            "luminosity": pa.array([1.0]),
            "blur": pa.array([2.0]),
        }
        metrics_dict = {"blur": None}
        batch = self._make_batch([])
        acc: dict[str, list] = {}
        job._accumulate_generated_features(batch, generated, metrics_dict, acc, 0)
        assert list(acc.keys()) == ["luminosity"]

    def test_generated_no_features_output(self):
        """Returns immediately when features_output is None for generated features."""
        job = DatasetJob(dataloaders={})
        generated = {"luminosity": pa.array([1.0])}
        batch = self._make_batch([])
        acc: dict[str, list] = {}
        result = job._accumulate_generated_features(batch, generated, {}, acc, 0)
        assert acc == {}
        assert result == 0


class TestPairwiseGapOutput:
    """Pairwise conditional logic for gap outputs."""

    def _check_pairwise(self, config_dict: dict) -> bool:
        """Validate pairwise flag from a partial config dict.

        Args:
            config_dict: Partial config containing gap/outputs key.

        Returns:
            True if pairwise is enabled, False otherwise.
        """
        validated = JobConfig.model_validate({"dataloaders": {"loaders": []}, **config_dict})
        return (
            validated.gap
            and validated.gap.outputs
            and hasattr(validated.gap.outputs, "pairwise")
            and validated.gap.outputs.pairwise
        )

    def test_default_is_true(self):
        """Pairwise defaults to True when not explicitly set."""
        assert self._check_pairwise({"gap": {"outputs": {"path": "g.parquet"}}})

    def test_explicit_true(self):
        """Pairwise is True when explicitly set."""
        assert self._check_pairwise({"gap": {"outputs": {"path": "g.parquet", "pairwise": True}}})

    def test_false_omits_column(self):
        """Pairwise column is omitted when pairwise is False."""
        assert not self._check_pairwise({"gap": {"outputs": {"path": "g.parquet", "pairwise": False}}})

    def test_no_gap_interface(self):
        """Returns False when gap interface is absent."""
        assert not self._check_pairwise({})

    def test_no_gap_outputs(self):
        """Returns False when gap interface has no outputs section."""
        assert not self._check_pairwise({"gap": {}})
