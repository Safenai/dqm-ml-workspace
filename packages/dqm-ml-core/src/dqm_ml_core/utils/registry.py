"""Plugin registry for dynamically loading DQM components.

This module contains the PluginLoadedRegistry class and load_registered_plugins
function for discovering and loading metric processors, data loaders,
and output writers via Python entry points.
"""

from __future__ import annotations

from importlib.metadata import EntryPoints, entry_points
import logging
import sys
from typing import Any

from dqm_ml_core.api.processor import Processor

logger = logging.getLogger(__name__)


# TODO once a base class for all registry created, dict shall have dict[str, base_class]
def load_registered_plugins(plugin_group: str, base_class: Any, base_name: str = "default") -> dict[str, Any]:
    """Discover and load plugins registered via Python entry points.

    Args:
        plugin_group: The entry point group name (e.g., 'dqm_ml.metrics').
        base_class: Optional base class to verify plugin type safety.
        base_name: Name of the base class to ignore during discovery.

    Returns:
        A dictionary mapping plugin names to their loaded classes.
    """
    try:
        # python 3.10+
        plugin_entry_points: EntryPoints = entry_points(group=plugin_group)
    except TypeError:
        # Old version for older python version
        logger.warning(f"Old python version not supported: {sys.version_info}")
        return {}

    registry = {}
    for v in plugin_entry_points:
        # Filter base class registry (not callable)
        if v.name != base_name:
            obj = v.load()
            if base_class is None or issubclass(obj, base_class):
                logger.debug(f"Referencing {plugin_group} - {v.name} class {obj} from {base_class}")
                registry[v.name] = obj
            else:
                logger.error(f"Entry point {plugin_group} - {v.name} class {obj} not derived from {base_class} ignored")

    # return a dict to class builder registry
    return registry


class PluginLoadedRegistry:
    """
    Singleton registry that provides lazy access to all registered DQM components.

    Components include:
    - Metrics (Processor)
    - DataLoaders
    - OutputWriters
    """

    _metrics_registry: dict[str, type[Processor]] | None = None
    _features_registry: dict[str, type[Processor]] | None = None
    _gap_registry: dict[str, type[Processor]] | None = None
    _dataloaders_registry: dict[str, Any] | None = None
    _outputwriter_registry: dict[str, Any] | None = None

    @classmethod
    def get_metrics_registry(cls) -> dict[str, type[Processor]]:
        """Return the registry of available metric processors.

        Returns:
            A dictionary mapping metric processor names to their classes.
        """
        if not cls._metrics_registry:
            from dqm_ml_core.api.metrics_processor import MetricsProcessor

            cls._metrics_registry = load_registered_plugins("dqm_ml.metrics", MetricsProcessor)

        return cls._metrics_registry

    @classmethod
    def get_features_registry(cls) -> dict[str, type[Processor]]:
        """Return the registry of available feature extraction processors.

        Returns:
            A dictionary mapping feature processor names to their classes.
        """
        if not cls._features_registry:
            from dqm_ml_core.api.features_processor import FeaturesProcessor

            cls._features_registry = load_registered_plugins("dqm_ml.features", FeaturesProcessor)

        return cls._features_registry

    @classmethod
    def get_gap_registry(cls) -> dict[str, type[Processor]]:
        """Return the registry of available gap processors.

        Returns:
            A dictionary mapping gap processor names to their classes.
        """
        if not cls._gap_registry:
            from dqm_ml_core.api.gap_processor import GapProcessor

            cls._gap_registry = load_registered_plugins("dqm_ml.gap", GapProcessor)

        return cls._gap_registry

    @classmethod
    def get_dataloaders_registry(cls) -> dict[str, Any]:
        """Return the registry of available data loaders.

        Returns:
            A dictionary mapping data loader names to their classes.
        """
        if not cls._dataloaders_registry:
            cls._dataloaders_registry = load_registered_plugins("dqm_ml.dataloaders", None)  # TODO add base class
        return cls._dataloaders_registry

    @classmethod
    def get_outputwriter_registry(cls) -> dict[str, Any]:
        """Return the registry of available output writers.

        Returns:
            A dictionary mapping output writer names to their classes.
        """
        if not cls._outputwriter_registry:
            cls._outputwriter_registry = load_registered_plugins("dqm_ml.outputwriter", None)  # TODO add base class

        return cls._outputwriter_registry
