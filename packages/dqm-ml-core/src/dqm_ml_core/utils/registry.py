from importlib.metadata import EntryPoints, entry_points
import logging
import sys
from typing import Any

from dqm_ml_core import DatametricProcessor

logger = logging.getLogger(__name__)


# TODO once a base class for all registry created, dict shall have dict[str, base_class]
def load_registered_plugins(plugin_group: str, base_class: Any, base_name: str = "default") -> dict[str, Any]:
    try:
        # python 3.10+
        plugin_entry_points: EntryPoints = entry_points(group=plugin_group)
    except TypeError:
        # Old version for older python version

        logger.warning(f"Old python version not supported: {sys.version_info}")

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
    Class to provide access to registered object for metrics, dataloader, or output writter
    """

    _metrics_registry: dict[str, type[DatametricProcessor]] | None = None
    _dataloaders_registry: dict[str, Any] | None = None
    _outputwiter_registry: dict[str, Any] | None = None

    @classmethod
    def get_metrics_registry(cls) -> dict[str, type[DatametricProcessor]]:
        if not cls._metrics_registry:
            cls._metrics_registry = load_registered_plugins("dqm_ml.metrics", DatametricProcessor)

        return cls._metrics_registry

    @classmethod
    def get_dataloaders_registry(cls) -> dict[str, Any]:
        if not cls._dataloaders_registry:
            cls._dataloaders_registry = load_registered_plugins("dqm_ml.dataloaders", None)  # TODO add base class
        return cls._dataloaders_registry

    @classmethod
    def get_outputwiter_registry(cls) -> dict[str, Any]:
        if not cls._outputwiter_registry:
            cls._outputwiter_registry = load_registered_plugins("dqm_ml.outputwiter", None)  # TODO add base class

        return cls._outputwiter_registry
