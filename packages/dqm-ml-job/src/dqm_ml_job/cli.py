"""Command-line interface for DQM job execution.

This module provides CLI functions for parsing arguments and running
data quality assessment jobs from YAML configuration files.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from dqm_ml_core import PluginLoadedRegistry
from dqm_ml_job.job import DatasetJob
from dqm_ml_job.outputwriter import OutputWriter

logger = logging.getLogger(__name__)


def parse_args(arg_list: list[str] | None) -> Any:
    """
    Parse command line arguments for the DQM job.

    Args:
        arg_list: List of arguments (default: sys.argv[1:]).

    Returns:
        The parsed Namespace object.
    """
    parser = argparse.ArgumentParser(
        prog="dqm-ml",
        description="DQM-ML Job client",
        epilog="for more informations see README",
    )

    parser.add_argument(
        "-p",
        "--process-config",
        type=str,
        nargs="+",
        required=True,
        help="configuration files to execute",
    )

    parser.add_argument(
        "--save-config",
        type=str,
        help="Path to save the resolved configuration",
    )

    # TODO add parameters to pass directly files / directory as inputs for loaders
    args = parser.parse_args(arg_list)

    return args


# TODO get parameters, logs, ...
def execute(arg_list: list[str] | None = None) -> None:
    """
    Main CLI entry point for executing DQM jobs from YAML configurations.
    Args:
        arg_list: List of command line arguments (default: sys.argv[1:]).
    """
    args = parse_args(arg_list)
    config: dict[str, Any] = {}

    for config_file in args.process_config:
        logger.debug("Executing job from config file: %s", config_file)

        with Path(config_file).open() as stream:
            try:
                config_content = yaml.safe_load(stream)
                config.update(config_content)
            except yaml.YAMLError as exc:
                logger.error("Fail to part job configuration: %s", config_file)
                print(exc)
                return

    # if we succeed to load all config files, run the job

    # Optionally save the resolved configuration
    if args.save_config:
        logger.debug("Saving resolved configuration to: %s", args.save_config)
        with Path(args.save_config).open("w") as stream:
            yaml.safe_dump(config, stream)

    if "config" in config:
        run(config["config"])
    elif "pipeline_config" in config:
        logger.warning("'pipeline_config' is deprecated, please use 'config' instead.")
        run(config["pipeline_config"])
    else:
        logger.error("No 'config' found in configuration.")


def _init_components(config_dict: dict[str, Any], registry: dict[str, Any], component_name: str) -> dict[str, Any]:
    """
    Initialize job components (loaders, metrics, writers) from their respective registries.

    Args:
        config_dict: Dictionary of component configurations from YAML.
        registry: The registry containing the component classes.
        component_name: The name of the component type (for error messages).

    Returns:
        A dictionary of initialized component instances.
    """
    components = {}
    for key, comp_config in config_dict.items():
        if "type" not in comp_config:
            raise ValueError(f"Configuration for {component_name} '{key}' must contain 'type'")
        comp_type = comp_config["type"]
        if comp_type not in registry:
            raise ValueError(f"{component_name.capitalize()} '{key}' has invalid type '{comp_type}'")
        components[key] = registry[comp_type](name=key, config=comp_config)
    return components


def _validate_config_key(config: dict[str, Any], key: str, label: str) -> None:
    """Validate that a config key exists and is a dictionary.

    Args:
        config: Configuration dictionary.
        key: Key name to validate.
        label: Human-readable label for error messages.

    Raises:
        ValueError: If the key is missing or not a dict.
    """
    if key not in config or not isinstance(config[key], dict):
        raise ValueError(f"'{key}' must be provided as a dictionary")
    if label == "metrics_processor" and "compute_delta" in config:
        logger.warning("compute_delta' is deprecated and will be removed in future versions.")


def _init_output_writers(
    outputs_config: dict[str, Any], outputs_registry: dict[str, Any]
) -> tuple[OutputWriter | None, OutputWriter | None, OutputWriter | None]:
    """Initialize output writers from configuration.

    Args:
        outputs_config: Outputs configuration dict.
        outputs_registry: Registry of available writer types.

    Returns:
        Tuple of (metrics_output, features_output, delta_output).
    """
    metrics_output: OutputWriter | None = None
    features_output: OutputWriter | None = None
    delta_output: OutputWriter | None = None

    for key, output_config in outputs_config.items():
        if output_config["type"] not in outputs_registry:
            raise ValueError(f"Output '{key}' must have a valid 'type' in {list(outputs_registry.keys())}")
        writer = outputs_registry[output_config["type"]](name=key, config=output_config)
        if key == "metrics":
            metrics_output = writer
        elif key == "delta_metrics":
            delta_output = writer
        elif key == "features":
            features_output = writer
        else:
            raise ValueError(
                f"Unsupported output key '{key}'. Only 'features', delta_metrics' and 'metrics' are allowed."
            )

    return metrics_output, features_output, delta_output


def run(config: dict[str, Any]) -> None:
    """
    Execute a job from a validated configuration dictionary.

    The config must contain:
    - dataloaders: Map of configurations for data sources.
    - metrics_processor: Map of configurations for quality metrics.
    - outputs: Map of configurations for results storage.
    """
    if not config:
        raise ValueError("Job requires a configuration dictionary.")

    dataloaders_registry = PluginLoadedRegistry.get_dataloaders_registry()
    metrics_registry = PluginLoadedRegistry.get_metrics_registry()
    outputs_registry = PluginLoadedRegistry.get_outputwriter_registry()

    _validate_config_key(config, "dataloaders", "dataloaders")
    dataloaders = _init_components(config["dataloaders"], dataloaders_registry, "dataloader")

    _validate_config_key(config, "metrics_processor", "metrics_processor")
    metrics = _init_components(config["metrics_processor"], metrics_registry, "metric")

    _validate_config_key(config, "outputs", "outputs")
    metrics_output, features_output, delta_output = _init_output_writers(config["outputs"], outputs_registry)

    job = DatasetJob(
        dataloaders=dataloaders,
        metrics=metrics,
        features_output=features_output,
        progress_bar=config.get("progress_bar", True),
    )

    dataselection_metrics_list, delta_metrics_table = job.run()

    if metrics_output:
        metrics_output.write_metrics_dict(dataselection_metrics_list)
        if hasattr(metrics_output, "flush"):
            metrics_output.flush()

    if delta_output and delta_metrics_table:
        delta_data = {col: delta_metrics_table.column(col) for col in delta_metrics_table.column_names}
        delta_output.write_table("delta", delta_data)
        if hasattr(delta_output, "flush"):
            delta_output.flush()


if __name__ == "__main__":
    execute()
