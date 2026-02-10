import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from dqm_ml_core import PluginLoadedRegistry
from dqm_ml_core.api.data_processor import DatametricProcessor
from dqm_ml_job.dataloaders import DataLoader
from dqm_ml_job.outputwriter import OutputWriter
from dqm_ml_job.pipeline import DatasetPipeline

logger = logging.getLogger(__name__)


def parse_args(arg_list: list[str] | None) -> Any:
    """
    Parse command line arguments for the DQM pipeline.

    Args:
        arg_list: List of arguments (default: sys.argv[1:]).

    Returns:
        The parsed Namespace object.
    """
    parser = argparse.ArgumentParser(
        prog="dqm-ml", description="DQM-ML Job client", epilog="for more informations see README"
    )

    parser.add_argument("-p", "--pipeline", type=str, nargs="+", required=True, help="pipeline files to execute")
    parser.add_argument("--save-config", type=str, help="Path to save the resolved configuration")

    # TODO add parameters to pass directly files / directory as inputs for loaders
    args = parser.parse_args(arg_list)

    return args


# TODO get parameters, logs, ...
def execute(arg_list: list[str] | None = None) -> None:
    """
    Main CLI entry point for executing DQM pipelines from YAML configurations.
    """
    args = parse_args(arg_list)
    config: dict[str, Any] = {}

    for config_file in args.pipeline:
        logger.debug("Executing pipeline from config file: %s", config_file)
        with Path(config_file).open() as stream:
            try:
                config_content = yaml.safe_load(stream)
                config.update(config_content)
            except yaml.YAMLError as exc:
                logger.error("Fail to part pipeline configuration: %s", config_file)
                print(exc)
                return

    # if we succeed to load all config files, run the pipeline
    # Optionally save the resolved configuration
    if args.save_config:
        logger.debug("Saving resolved configuration to: %s", args.save_config)
        with Path(args.save_config).open("w") as stream:
            yaml.safe_dump(config, stream)

    run(config["pipeline_config"])


def _init_components(config_dict: dict[str, Any], registry: dict[str, Any], component_name: str) -> dict[str, Any]:
    """
    Initialize pipeline components (loaders, metrics, writers) from their respective registries.

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


def run(config: dict[str, Any]) -> None:
    """
    Execute a pipeline from a validated configuration dictionary.

    The config must contain:
    - dataloaders: Map of configurations for data sources.
    - metrics_processor: Map of configurations for quality metrics.
    - outputs: Map of configurations for results storage.
    """
    dataloaders_registry = PluginLoadedRegistry.get_dataloaders_registry()
    metrics_registry = PluginLoadedRegistry.get_metrics_registry()
    outputs_registry = PluginLoadedRegistry.get_outputwriter_registry()

    if not config:
        raise ValueError("Pipeline requires a configuration dictionary.")

    # Load data loaders
    if "dataloaders" not in config or not isinstance(config["dataloaders"], dict):
        raise ValueError("'dataloaders' must be provided as a dictionary")

    dataloaders: dict[str, DataLoader] = _init_components(config["dataloaders"], dataloaders_registry, "dataloader")

    # Load metrics
    if "metrics_processor" not in config or not isinstance(config["metrics_processor"], dict):
        raise ValueError("'metrics_processor' must be provided as a dictionary")

    metrics: dict[str, DatametricProcessor] = _init_components(config["metrics_processor"], metrics_registry, "metric")

    if "compute_delta" in config:
        logger.warning("compute_delta' is deprecated and will be removed in future versions.")

    # Load output writers
    if "outputs" not in config or not isinstance(config["outputs"], dict):
        raise ValueError("'outputs' must be provided as a dictionary")

    metrics_output: OutputWriter | None = None
    features_output: OutputWriter | None = None
    delta_output: OutputWriter | None = None

    for key, output_config in config["outputs"].items():
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
            raise ValueError(f"Unsupported output key '{key}'. Only 'features' and 'metrics' are allowed.")

    progress_bar = config.get("progress_bar", True)

    pipeline = DatasetPipeline(
        dataloaders=dataloaders, metrics=metrics, features_output=features_output, progress_bar=progress_bar
    )

    dataselection_metrics_list, delta_metrics_table = pipeline.run()

    # If we have computed metrics to store
    if metrics_output:
        metrics_output.write_metrics_dict(dataselection_metrics_list)

    # If we have to compute delta metrics
    if delta_output and delta_metrics_table:
        delta_output.write_table("delta", delta_metrics_table)


if __name__ == "__main__":
    execute()
