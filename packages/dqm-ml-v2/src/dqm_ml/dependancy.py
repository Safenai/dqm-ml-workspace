from contextlib import contextmanager
import logging
from typing import Any

from dqm_ml_core.registry import PluginLoadedRegistry

from dqm_ml._version_ import version
from dqm_ml_core._version_ import version as core_version

logger = logging.getLogger(__name__)


@contextmanager
def optional_dependencies(error: str = "ignore"):
    assert error in {"raise", "warn", "ignore"}
    try:
        yield None
    except ImportError as e:
        if error == "raise":
            raise e
        if error == "warn":
            msg = f'Missing optional dependency "{e.name}". Use pip to install.'
            print(f"Warning: {msg}")


def display_version(arg_list: list[str] | None = None):
    if core_version != version:
        logger.warning(f"Different dqm-ml-core version {core_version} fro dqm-ml version {version}")
    print(f"DQML version : {version}")


def display_list_of(arg_list: list[str] | None = None):
    # TODO : we display all but we can filter / use extra parameters
    print("Available data metrics_registry")
    for key, value in PluginLoadedRegistry.get_metrics_registry().items():
        print(f"- {key} - {value}")

    print("Available data loaders")
    for key, value in PluginLoadedRegistry.get_dataloaders_registry().items():
        print(f"- {key} - {value}")

    print("Available outputs writter")
    for key, value in PluginLoadedRegistry.get_outputwiter_registry().items():
        print(f"- {key} - {value}")


def get_available_command() -> dict[str, Any]:
    command_list = {"version": display_version, "list": display_list_of}
    optional_dep_mode = "warn"

    # We import available command for dqml cli
    with optional_dependencies(optional_dep_mode):
        from dqm_ml_pipeline._version_ import version as pipeline_version
        from dqm_ml_pipeline.cli import execute

        logger.debug(f"Different dqm-ml-pipeline version {pipeline_version}")
        command_list["process"] = execute

    return command_list
