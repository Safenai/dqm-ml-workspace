"""Dependency management utilities for DQM-ML v2.

This module provides functions for handling optional dependencies,
displaying version information, and discovering available commands.
"""

from collections.abc import Generator
from contextlib import contextmanager
import logging
from typing import Any

from dqm_ml_core import PluginLoadedRegistry
from dqm_ml_core._version_ import version as core_version

logger = logging.getLogger(__name__)


@contextmanager
def optional_dependencies(error: str = "ignore") -> Generator[None, None, None]:
    """Context manager for handling optional dependencies.

    Args:
        error: How to handle missing optional dependencies:
            - "ignore": Silently continue (default)
            - "warn": Print a warning message
            - "raise": Raise the ImportError

    Yields:
        None - this is a context manager that doesn't provide any value.
    """
    assert error in {"raise", "warn", "ignore"}
    try:
        yield None
    except ImportError as e:
        if error == "raise":
            raise e
        if error == "warn":
            msg = f'Missing optional dependency "{e.name}". Use pip to install.'
            print(f"Warning: {msg}")


def display_version(arg_list: list[str] | None = None) -> None:
    """Print the DQM-ML core version to stdout.

    Args:
        arg_list: Unused, provided for CLI compatibility.
    """
    print(f"DQM-ML version : {core_version}")


def display_list_of(arg_list: list[str] | None = None) -> None:
    """Print all registered plugins (metrics, dataloaders, output writers).

    Args:
        arg_list: Unused, provided for CLI compatibility.
    """
    # TODO : we display all but we can filter / use extra parameters
    print("Available data metrics_registry")
    for key, value in PluginLoadedRegistry.get_metrics_registry().items():
        print(f"- {key} - {value}")

    print("Available data features_registry")
    for key, value in PluginLoadedRegistry.get_features_registry().items():
        print(f"- {key} - {value}")

    print("Available data gap_registry")
    for key, value in PluginLoadedRegistry.get_gap_registry().items():
        print(f"- {key} - {value}")

    print("Available data loaders")
    for key, value in PluginLoadedRegistry.get_dataloaders_registry().items():
        print(f"- {key} - {value}")

    print("Available outputs writers")
    for key, value in PluginLoadedRegistry.get_outputwriter_registry().items():
        print(f"- {key} - {value}")


def get_available_command() -> dict[str, Any]:
    """Build a dictionary of available CLI commands.

    Returns:
        Dict mapping command names to their handler functions.
    """
    command_list = {"version": display_version, "list": display_list_of}
    optional_dep_mode = "warn"

    # We import available command for dqml cli
    with optional_dependencies(optional_dep_mode):
        from dqm_ml_job._version_ import version as pipeline_version
        from dqm_ml_job.cli import execute

        logger.debug(f"Different dqm-ml-job version {pipeline_version}")
        command_list["process"] = execute

    return command_list
