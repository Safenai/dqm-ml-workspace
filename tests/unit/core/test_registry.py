"""Unit tests for the plugin registry utilities.

This module contains tests that verify the plugin loading and registry
functionality for metrics, dataloaders, and output writers.
"""

from importlib.metadata import EntryPoint
from unittest.mock import MagicMock, patch

from dqm_ml_core import DatametricProcessor
from dqm_ml_core.utils.registry import PluginLoadedRegistry, load_registered_plugins


class TestLoadRegisteredPlugins:
    """Tests for the load_registered_plugins function.

    Covers error handling for old Python versions, filtering of non-subclass entry points,
    and logging of ignored plugins.
    """

    def test_old_python_version(self):
        """Verify load_registered_plugins returns empty dict on TypeError (old Python)."""
        with patch("dqm_ml_core.utils.registry.entry_points", side_effect=TypeError("old version")):
            result = load_registered_plugins("dqm_ml.metrics", DatametricProcessor)
        assert result == {}

    def test_entry_point_not_subclass_ignored(self, caplog):
        """Verify entry points not subclassing the base class are ignored and logged.

        Args:
            caplog: Pytest fixture to capture log output.
        """

        class FakeBase:
            pass

        class NotSubclass:
            pass

        mock_entry = MagicMock(spec=EntryPoint)
        mock_entry.name = "bad_metric"
        mock_entry.load.return_value = NotSubclass

        with patch("dqm_ml_core.utils.registry.entry_points", return_value=[mock_entry]):
            result = load_registered_plugins("dqm_ml.metrics", FakeBase)

        assert result == {}
        assert "not derived from" in caplog.text


class TestPluginLoadedRegistry:
    """Tests for the PluginLoadedRegistry lazy-loading singleton.

    Covers lazy initialization of metrics, dataloaders, and output writer registries,
    and verifies built-in processors/loaders/writers are registered on first access.
    """

    def teardown_method(self):
        """Reset the singleton registry after each test to ensure isolation."""
        PluginLoadedRegistry._metrics_registry = None

    def test_get_metrics_registry_lazy_load(self):
        """Verify metrics registry loads built-in processors on first access."""
        registry = PluginLoadedRegistry.get_metrics_registry()
        assert "completeness" in registry
        assert "diversity" in registry
        assert "representativeness" in registry

    def test_get_dataloaders_registry_lazy_load(self):
        """Verify dataloaders registry loads built-in loaders on first access."""
        registry = PluginLoadedRegistry.get_dataloaders_registry()
        assert "csv" in registry
        assert "parquet" in registry

    def test_get_outputwriter_registry_lazy_load(self):
        """Verify output writer registry loads built-in writers on first access."""
        registry = PluginLoadedRegistry.get_outputwriter_registry()
        assert "parquet" in registry
