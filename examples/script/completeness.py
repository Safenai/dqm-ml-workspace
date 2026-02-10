import os
from typing import Any

import yaml

from dqm_ml_job.cli import run as exec_qml_job


def compute_metric() -> None:
    """Example script to compute a metric using a YAML configuration."""

    # Load configuration file or create a dictionary structure with the same keys
    cur_file_path = os.path.abspath(__file__)
    config_path = os.path.join(os.path.dirname(cur_file_path), "../config/completeness.yaml")

    config: dict[str, Any] = {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

        # Execute the job with the loaded configuration, output are directlu saved to disk
        exec_qml_job(config["pipeline_config"])

        # A more granular API will be provided in future releases to access intermediate results


if __name__ == "__main__":
    compute_metric()
