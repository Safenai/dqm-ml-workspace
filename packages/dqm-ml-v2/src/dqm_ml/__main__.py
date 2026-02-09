import argparse
from collections.abc import Iterable
import logging
from typing import Any, override

from dqm_ml.cli_tools import CustomFormatter
from dqm_ml.dependency import get_available_command

logger = logging.getLogger(__name__)


class _HelpAction(argparse._HelpAction):
    @override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Iterable[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if namespace.command:
            # print help for the specific command
            command_list = get_available_command()
            if namespace.command in command_list and command_list[namespace.command] is not None:
                command_list[namespace.command](["-h"])
            else:
                raise ValueError(f"Unknown command {namespace.command}")
        else:
            parser.print_help()
            parser.exit()


def parse_args(arg_list: list[str] | None, command_list: Iterable[str]) -> Any:
    parser = argparse.ArgumentParser(
        prog="dqm-ml-v2",
        description="DQM-ML Pipeline client",
        epilog="for more informations see README",
        add_help=False,
    )

    parser.add_argument("-h", "--help", action=_HelpAction, help="help for help if you need some help")

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
        CustomFormatter.init_log(format="%(name)s - %(message)s (%(filename)s:%(lineno)d)", level=logging.DEBUG)  # noqa: E501
    elif args.quiet:
        CustomFormatter.init_log(format="%(message)s", level=logging.ERROR)
    else:
        CustomFormatter.init_log(format="%(message)s", level=logging.INFO)

    logger.debug(f"Execution dqm-ml with {arg_list}")

    if args.command in command_list and command_list[args.command] is not None:
        command_list[args.command](remaining)
    else:
        raise ValueError(f"Unkow comand {args.command}")


if __name__ == "__main__":
    execute()
