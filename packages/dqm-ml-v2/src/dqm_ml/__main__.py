import argparse
import logging
from typing import Any

from dqm_ml.dependancy import get_available_command

logger = logging.getLogger(__name__)


def parse_args(arg_list: list[str] | None, command_list: str) -> Any:
    parser = argparse.ArgumentParser(
        prog="dqm-ml", description="DQM-ML Pipeline client", epilog="for more informations see README"
    )

    parser.add_argument("command", choices=command_list, help="Available command for your dqm-ml installation")

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")

    cli_args, remaining = parser.parse_known_args(arg_list)

    return cli_args, remaining


# TODO get parameters, logs, ...
def execute(arg_list: list[str] | None = None) -> None:
    # Exemple of other optional dependencies command
    # with optional_dependencies(optional_dep_mode):
    #    import dqm_ml_dummy_command.cli
    #    command_list["dqm_ml_dummy_command"] = dqm_ml_dummy_command.cli
    command_list = get_available_command()

    args, remaining = parse_args(arg_list, command_list)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        # CustomFormatter.init_log(format="%(name)s - %(message)s (%(filename)s:%(lineno)d)", level=logging.DEBUG)  # noqa: E501
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
        # CustomFormatter.init_log(format="%(message)s", level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
        # CustomFormatter.init_log(format="%(message)s", level=logging.INFO)

    logger.debug(f"Execution dqm-ml with {arg_list}")

    if args.command in command_list and command_list[args.command] is not None:
        command_list[args.command](remaining)
    else:
        raise ValueError(f"Unkow comand {args.command}")


if __name__ == "__main__":
    execute()
