"""Command-line interface for DQM job execution.

This module provides CLI functions for parsing arguments and running
data quality assessment jobs from YAML configuration files.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, cast

from dqm_ml_core import PluginLoadedRegistry
from dqm_ml_core.models.config import JobConfig
from dqm_ml_core.models.global_ import ComputeConfig, ErrorsConfig
from dqm_ml_core.models.interfaces import FeaturesInterfaceConfig, GapInterfaceConfig, MetricsInterfaceConfig
import pyarrow as pa
import yaml

from dqm_ml_job.job import DatasetJob
from dqm_ml_job.outputwriter import OutputWriter

logger = logging.getLogger(__name__)


def _merge_errors(global_errors: ErrorsConfig | None, interface_errors: ErrorsConfig | None) -> ErrorsConfig:
    """Merge global and interface-specific errors, with interface taking precedence."""
    if interface_errors is None:
        return global_errors or ErrorsConfig()

    # Start with global defaults
    merged = global_errors or ErrorsConfig()

    # Override with interface-specific values (only where interface is not None)
    if interface_errors.default is not None:
        merged.default = interface_errors.default
    if interface_errors.images is not None:
        merged.images = interface_errors.images
    if interface_errors.tabular is not None:
        merged.tabular = interface_errors.tabular
    if interface_errors.max_failure_rate is not None:
        merged.max_failure_rate = interface_errors.max_failure_rate

    return merged


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

        config_path = Path(config_file).resolve()
        if not config_path.is_file():
            logger.error("Config file does not exist: %s", config_file)
            return

        with config_path.open() as stream:
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
        save_path = Path(args.save_config).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as stream:
            yaml.safe_dump(config, stream)

    run(config)


def _init_components_from_list(
    processor_list: list[dict[str, Any]], registry: dict[str, Any], component_name: str
) -> dict[str, Any]:
    """Initialize components from a list of processor configs (new format).

    Each item in the list must have a 'name' and 'type' field.

    Args:
        processor_list: List of component configuration dicts.
        registry: The registry containing the component classes.
        component_name: The name of the component type (for error messages).

    Returns:
        A dictionary mapping component names to initialized instances.
    """
    components = {}
    for comp_config in processor_list:
        proc_name = comp_config.get("name")
        if not proc_name:
            raise ValueError(f"Configuration for {component_name} must contain 'name'")
        if "type" not in comp_config:
            raise ValueError(f"Configuration for {component_name} '{proc_name}' must contain 'type'")
        comp_type = comp_config["type"]
        if comp_type not in registry:
            raise ValueError(f"{component_name.capitalize()} '{proc_name}' has invalid type '{comp_type}'")
        components[proc_name] = registry[comp_type](name=proc_name, config=comp_config)
    return components


def _init_output_writer(
    name: str,
    path: str,
    columns: list[str] | None,
    outputs_registry: dict[str, Any],
    exclude: list[str] | None = None,
    storage: dict[str, Any] | None = None,
) -> OutputWriter | None:
    """Initialize a single output writer from interface outputs config.

    Args:
        name: Name for the writer instance.
        path: Output path from the interface outputs config.
        columns: Optional list of columns to include.
        outputs_registry: Registry of available writer types.
        exclude: Optional list of columns to exclude.
        storage: Storage config dict to inject into the writer (optional).

    Returns:
        An initialized OutputWriter instance, or None if no writer type is available.
    """
    writer_type = "parquet"
    if writer_type not in outputs_registry:
        logger.warning("Output writer type '%s' not found in registry", writer_type)
        return None
    writer_config: dict[str, Any] = {"path_pattern": path, "columns": columns or [], "exclude": exclude or []}
    if storage:
        writer_config["storage"] = storage
    return cast(OutputWriter, outputs_registry[writer_type](name=name, config=writer_config))


def _init_interface_outputs(
    interface_config: FeaturesInterfaceConfig | MetricsInterfaceConfig | GapInterfaceConfig | None,
    outputs_registry: dict[str, Any],
    kind: str,
    storage: dict[str, Any] | None = None,
) -> OutputWriter | None:
    """Initialize the output writer for a given interface.

    Args:
        interface_config: The validated interface configuration.
        outputs_registry: Registry of available writer types.
        kind: The kind of interface ('features', 'metrics', or 'gap').
        storage: Storage config dict to inject into the writer (optional).

    Returns:
        An initialized OutputWriter instance, or None.
    """
    if interface_config is None or interface_config.outputs is None:
        return None
    path = interface_config.outputs.path
    columns: list[str] | None = None
    if hasattr(interface_config.outputs, "include") and interface_config.outputs.include:
        columns = interface_config.outputs.include
    exclude: list[str] | None = None
    if hasattr(interface_config.outputs, "exclude") and interface_config.outputs.exclude:
        exclude = interface_config.outputs.exclude
    return _init_output_writer(kind, path, columns, outputs_registry, exclude, storage)


def _resolve_compute_config(validated: JobConfig) -> ComputeConfig:
    """Resolve the compute config, providing defaults if not specified."""
    if validated.compute:
        return validated.compute
    return ComputeConfig(
        seed=42,
        log_level="warning",
        max_memory=None,
        device="auto",
        progress_bar=True,
        threads=4,
    )


def _init_processors_from_interface(
    interface: Any,
    registry: dict[str, Any],
    storage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Initialize processors from an optional interface config.

    Args:
        interface: The validated interface config (or None).
        registry: The component registry.
        storage: Storage config dict to inject into each processor (optional).

    Returns:
        A dict of name-to-processor instances (empty if interface is None).
    """
    if interface is None:
        return {}
    proc_dicts = [p.model_dump() for p in interface.processors]
    if storage:
        for proc_dict in proc_dicts:
            proc_dict["storage"] = storage
    return _init_components_from_list(proc_dicts, registry, "processor")


def _safe_flush(writer: Any) -> None:
    """Flush a writer if it has a flush method."""
    if writer and hasattr(writer, "flush"):
        writer.flush()


def _build_errors_by_interface(validated: JobConfig) -> dict[str, ErrorsConfig]:
    """Build per-interface error configs by merging global and interface-specific errors.

    Returns:
        Dict mapping interface name to merged ErrorsConfig.
    """
    errors_by_interface: dict[str, ErrorsConfig] = {}
    for name in ("features", "metrics", "gap"):
        interface = getattr(validated, name, None)
        interface_errors = interface.errors if interface else None
        errors_by_interface[name] = _merge_errors(validated.errors, interface_errors)
    return errors_by_interface


def _enrich_delta_with_pairwise(
    validated: JobConfig,
    delta_data: dict[str, Any],
    delta_metrics_table: pa.Table,
) -> None:
    """Add a ``source_target`` column to delta data if pairwise output is configured."""
    if not (validated.gap and validated.gap.outputs and getattr(validated.gap.outputs, "pairwise", False)):
        return
    source_target_values = [
        f"{delta_metrics_table.column('selection_source')[i].as_py()}"
        f"_{delta_metrics_table.column('selection_target')[i].as_py()}"
        for i in range(delta_metrics_table.num_rows)
    ]
    delta_data["source_target"] = pa.array(source_target_values)


def run(config: dict[str, Any]) -> None:
    """
    Execute a job from a validated configuration dictionary.

    The config is validated against JobConfig and must follow the v2 structure:
    - dataloaders: Contains loaders list and optional storage.
    - features: Optional interface with outputs and processors list.
    - metrics: Optional interface with outputs and processors list.
    - gap: Optional interface with outputs and processors list.
    """
    if not config:
        raise ValueError("Job requires a configuration dictionary.")

    validated = JobConfig.model_validate(config)

    dataloaders_registry = PluginLoadedRegistry.get_dataloaders_registry()
    features_registry = PluginLoadedRegistry.get_features_registry()
    metrics_registry = PluginLoadedRegistry.get_metrics_registry()
    gap_registry = PluginLoadedRegistry.get_gap_registry()
    outputs_registry = PluginLoadedRegistry.get_outputwriter_registry()

    # Initialize dataloaders from list format
    dataloader_dicts = [loader.model_dump() for loader in validated.dataloaders.loaders]
    compute = _resolve_compute_config(validated)
    dl_storage = validated.dataloaders.storage.model_dump() if validated.dataloaders.storage else None
    for dl in dataloader_dicts:
        dl["threads"] = compute.threads
        if dl_storage and not dl.get("storage"):
            dl["storage"] = dl_storage
    dataloaders = _init_components_from_list(dataloader_dicts, dataloaders_registry, "dataloader")

    # Resolve storage config: interface override takes precedence over job-level
    def _resolve_storage(interface: Any) -> dict[str, Any] | None:
        if interface and interface.storage:
            result: dict[str, Any] = interface.storage.model_dump()
            return result
        if validated.storage:
            result = validated.storage.model_dump()
            return result
        return None

    # Initialize processors from all interfaces
    features_processors = _init_processors_from_interface(
        validated.features,
        features_registry,
        _resolve_storage(validated.features),
    )
    metrics_processors = _init_processors_from_interface(
        validated.metrics,
        metrics_registry,
        _resolve_storage(validated.metrics),
    )
    gap_processors = _init_processors_from_interface(
        validated.gap,
        gap_registry,
        _resolve_storage(validated.gap),
    )

    # Initialize output writers from interfaces
    features_output = _init_interface_outputs(
        validated.features,
        outputs_registry,
        "features",
        _resolve_storage(validated.features),
    )
    metrics_output = _init_interface_outputs(
        validated.metrics,
        outputs_registry,
        "metrics",
        _resolve_storage(validated.metrics),
    )
    delta_output = _init_interface_outputs(
        validated.gap,
        outputs_registry,
        "delta",
        _resolve_storage(validated.gap),
    )

    # Configure logging based on compute.log_level
    if compute.log_level:
        log_level = compute.log_level.upper()
        level = getattr(logging, log_level)
        logging.basicConfig(level=level)

    job = DatasetJob(
        dataloaders=dataloaders,
        features_processors=features_processors,
        metrics_processors=metrics_processors,
        gap_processors=gap_processors,
        features_output=features_output,
        progress_bar=compute.progress_bar,
        threads=compute.threads,
        errors_by_interface=_build_errors_by_interface(validated),
        compute_seed=compute.seed,
        compute_device=compute.device,
        compute_max_memory=compute.max_memory,
    )

    dataselection_metrics_list, delta_metrics_table = job.run()

    if metrics_output:
        metrics_output.write_metrics_dict(dataselection_metrics_list)
        _safe_flush(metrics_output)

    if delta_output and delta_metrics_table:
        delta_data = {col: delta_metrics_table.column(col) for col in delta_metrics_table.column_names}
        _enrich_delta_with_pairwise(validated, delta_data, delta_metrics_table)
        delta_output.write_table("delta", delta_data)
        _safe_flush(delta_output)


if __name__ == "__main__":
    execute()
