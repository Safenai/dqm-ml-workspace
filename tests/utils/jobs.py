"""Job utility functions for DQM-ML tests.

This module provides helper functions for generating test job configurations.
"""

from pathlib import Path
from typing import Any

import ruamel.yaml

DOMAIN_GAP_INFER_PARAMS = {
    "fid": {"batch_size": 32, "width": 299, "height": 299},
    "klmvn_diag": {"batch_size": 10, "width": 20, "height": 20},
    "mmd_linear": {"batch_size": 10, "width": 224, "height": 224},
    "wasserstein_1d": {"batch_size": 18, "height": 299, "width": 299},
}

OUTPUT_DATA = "outputs/data"

BATCH_SIZES = {
    "representativeness": 50000,
    "domain_gap": 50,
    "completeness": 100,
    "visual_features": 100,
}


def _get_config_name(processor_name: str, test_name: str, metric_name: str | None) -> str:
    """Generate configuration name based on processor and test parameters."""
    if processor_name == "domain_gap":
        return f"{processor_name}_{test_name}" if test_name else f"{processor_name}_{metric_name}"
    if processor_name in ("completeness", "visual_features"):
        return test_name
    return f"{processor_name}_{test_name}"


def _load_yaml_template(test_path: str, processor_name: str) -> tuple[Any, int, Any]:
    """Load and parse YAML template file."""
    template_path = Path(test_path) / f"integration/fixtures/config/templates/{processor_name}.yaml"
    with Path(template_path).open() as file:
        return ruamel.yaml.util.load_yaml_guess_indent(file)  # type: ignore[no-any-return]


def _configure_dataloaders(
    inner_config: dict, processor_name: str, test_name: str, parquet_path: Path, parquet_source_path: Path | None
) -> None:
    """Configure dataloaders section of the config."""
    if processor_name == "domain_gap":
        inner_config["dataloaders"]["source_dataset"]["path"] = str(parquet_source_path)
        inner_config["dataloaders"]["target_dataset"]["path"] = str(parquet_path)
    else:
        inner_config["dataloaders"]["source_dataset"]["path"] = str(parquet_path)

    if "batch" in test_name:
        batch_size = BATCH_SIZES.get(processor_name)
        if batch_size:
            inner_config["dataloaders"]["source_dataset"]["batch_size"] = batch_size
            if processor_name == "domain_gap":
                inner_config["dataloaders"]["target_dataset"]["batch_size"] = batch_size


def _configure_domain_gap(inner_config: dict, processor_name: str, metric_name: str) -> None:
    """Configure domain gap specific settings."""
    if processor_name != "domain_gap":
        return
    inner_config["metrics_processor"][processor_name]["DELTA"]["metric"] = metric_name
    for param in ("batch_size", "height", "width"):
        inner_config["metrics_processor"]["image_embedding"]["infer"][param] = DOMAIN_GAP_INFER_PARAMS[metric_name][
            param
        ]


def _configure_metrics_processor(inner_config: dict, processor_name: str, test_name: str) -> None:
    """Configure metrics processor section."""
    if processor_name == "representativeness":
        inner_config["metrics_processor"]["representativeness"]["distribution"] = (
            "uniform" if "uniform" in test_name else "normal"
        )
    elif processor_name == "visual_features" and "path" in test_name:
        inner_config["metrics_processor"]["visual_features"]["input_columns"] = ["image_path"]
    elif processor_name == "domain_gap" and "bytes" in test_name:
        inner_config["metrics_processor"]["image_embedding"]["DATA"]["image_column"] = "image_bytes"
        inner_config["metrics_processor"]["image_embedding"]["DATA"]["mode"] = "bytes"


def _configure_output(inner_config: dict, output_category: str, config_name: str, output_path: Path) -> None:
    """Configure output section of the config."""
    inner_config["outputs"][output_category]["path_pattern"] = (
        f"{output_path!s}/metrics_{config_name}_" + "{}-{}.parquet"
    )


def generate_job(
    test_path: str,
    processor_name: str,
    output_category: str,
    parquets_path: Path,
    test_list: list[dict[str, str]],
    metric_name: str | None = None,
    parquet_source_path: Path | None = None,
) -> None:
    """Generate test job configuration files from templates.

    Args:
        test_path: Path to the tests directory.
        processor_name: Name of the processor (e.g., 'completeness', 'representativeness').
        output_category: Output category (e.g., 'metrics', 'delta_metrics', 'features').
        parquets_path: Path to parquet files directory.
        test_list: List of test configurations to generate.
        metric_name: Optional metric name for domain gap tests.
        parquet_source_path: Optional source parquet path for domain gap tests.
    """
    configs_path = Path(test_path) / "integration/fixtures/config/generated"
    output_path = Path(test_path) / OUTPUT_DATA
    Path(configs_path).mkdir(exist_ok=True, parents=True)

    for test in test_list:
        parquet_path = parquets_path / test["parquet"]
        test_name = test["test_name"]
        config_name = _get_config_name(processor_name, test_name, metric_name)
        config_path = Path(f"{configs_path}/{config_name}.yaml")

        full_config, ind, bsi = _load_yaml_template(test_path, processor_name)
        inner_config = full_config["config"]

        _configure_dataloaders(inner_config, processor_name, test_name, parquet_path, parquet_source_path)
        if metric_name:
            _configure_domain_gap(inner_config, processor_name, metric_name)
        _configure_metrics_processor(inner_config, processor_name, test_name)
        _configure_output(inner_config, output_category, config_name, output_path)

        yaml_config = ruamel.yaml.YAML()
        yaml_config.indent(mapping=ind, sequence=ind, offset=bsi)
        with Path(config_path).open("w") as fp:
            yaml_config.dump(full_config, fp)
