import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from dqm_ml_pipeline.pipeline import DatasetPipeline

logger = logging.getLogger(__name__)


def parse_args(arg_list: list[str] | None) -> Any:
    parser = argparse.ArgumentParser(
        prog="dqm-ml", description="DQM-ML Pipeline client", epilog="for more informations see README"
    )

    parser.add_argument("-p", "--pipeline", type=str, default="pipeline.yaml", help="pipeline file to execute")

    args = parser.parse_args(arg_list)

    return args


# TODO get parameters, logs, ...
def execute(arg_list: list[str] | None = None) -> None:
    args = parse_args(arg_list)

    with Path(args.pipeline).open() as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    run(config=config)


def run(config: dict[str, Any]) -> None:
    # TODO : get parameters from config
    pipeline = DatasetPipeline(config=config["pipeline_config"])

    pipeline.run()


if __name__ == "__main__":
    execute()
