"""Job utility functions for DQM-ML tests.

This module provides helper functions for generating test job configurations.
"""

from pathlib import Path

import ruamel.yaml


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
    output_path = Path(test_path) / "outputs/data"

    domain_gap_infer_params = {
        "fid": {"batch_size": 32, "width": 299, "height": 299},
        "klmvn_diag": {"batch_size": 10, "width": 20, "height": 20},
        "mmd_linear": {"batch_size": 10, "width": 224, "height": 224},
        "wasserstein_1d": {"batch_size": 18, "height": 299, "width": 299},
    }

    Path(configs_path).mkdir(exist_ok=True, parents=True)

    for test in test_list:
        parquet_path = parquets_path / test["parquet"]
        test_name = test["test_name"]

        if processor_name == "domain_gap":
            config_name = f"{processor_name}_{test_name}" if test_name != "" else f"{processor_name}_{metric_name}"
        elif processor_name in ["completeness", "visual_features"]:
            config_name = test_name
        else:
            config_name = f"{processor_name}_{test_name}"

        config_path = Path(f"{configs_path}/{config_name}.yaml")

        template_path = Path(test_path) / f"integration/fixtures/config/templates/{processor_name}.yaml"
        with Path(template_path).open() as file:
            full_config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(file)

        inner_config = full_config["config"]
        if processor_name == "domain_gap":
            inner_config["dataloaders"]["source_dataset"]["path"] = str(parquet_source_path)
            inner_config["dataloaders"]["target_dataset"]["path"] = str(parquet_path)
            inner_config["metrics_processor"][processor_name]["DELTA"]["metric"] = metric_name
            for param in ["batch_size", "height", "width"]:
                inner_config["metrics_processor"]["image_embedding"]["infer"][param] = domain_gap_infer_params[
                    metric_name
                ][param]
        else:
            inner_config["dataloaders"]["source_dataset"]["path"] = str(parquet_path)

        if "batch" in test_name:
            if processor_name == "representativeness":
                inner_config["dataloaders"]["source_dataset"]["batch_size"] = 50000
            if processor_name == "domain_gap":
                inner_config["dataloaders"]["source_dataset"]["batch_size"] = 50
                inner_config["dataloaders"]["target_dataset"]["batch_size"] = 50
            if processor_name == "completeness":
                inner_config["dataloaders"]["source_dataset"]["batch_size"] = 100
            if processor_name == "visual_features":
                inner_config["dataloaders"]["source_dataset"]["batch_size"] = 100

        if processor_name == "representativeness":
            if "uniform" in test_name:
                inner_config["metrics_processor"]["representativeness"]["distribution"] = "uniform"
            else:
                inner_config["metrics_processor"]["representativeness"]["distribution"] = "normal"

        if processor_name == "visual_features" and "path" in test_name:
            inner_config["metrics_processor"]["visual_features"]["input_columns"] = ["image_path"]

        if processor_name == "domain_gap" and "bytes" in test_name:
            inner_config["metrics_processor"]["image_embedding"]["DATA"]["image_column"] = "image_bytes"
            inner_config["metrics_processor"]["image_embedding"]["DATA"]["mode"] = "bytes"

        if processor_name == "domain_gap":
            inner_config["outputs"][output_category]["path_pattern"] = (
                f"{output_path!s}/metrics_{config_name}_" + "{}-{}.parquet"
            )
        else:
            inner_config["outputs"][output_category]["path_pattern"] = (
                f"{output_path!s}/metrics_{config_name}_" + "{}-{}.parquet"
            )

        yaml_config = ruamel.yaml.YAML()
        yaml_config.indent(mapping=ind, sequence=ind, offset=bsi)
        with Path(config_path).open("w") as fp:
            yaml_config.dump(full_config, fp)
