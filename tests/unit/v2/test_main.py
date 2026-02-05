import argparse
import logging
from unittest.mock import MagicMock, patch

from dqm_ml.__main__ import _HelpAction, execute, parse_args
import pytest


def test_help_action_no_command():
    parser = MagicMock(spec=argparse.ArgumentParser)
    # MagicMock's exit usually doesn't raise unless we tell it to
    parser.exit.side_effect = SystemExit
    namespace = argparse.Namespace(command=None)
    action = _HelpAction(option_strings=["-h", "--help"], dest="help")

    with pytest.raises(SystemExit):
        action(parser, namespace, None)

    parser.print_help.assert_called_once()
    parser.exit.assert_called_once()


def test_help_action_with_valid_command():
    parser = MagicMock(spec=argparse.ArgumentParser)
    namespace = argparse.Namespace(command="list")
    action = _HelpAction(option_strings=["-h", "--help"], dest="help")

    mock_commands = {"list": MagicMock()}
    with patch("dqm_ml.__main__.get_available_command", return_value=mock_commands):
        action(parser, namespace, None)

    mock_commands["list"].assert_called_once_with(["-h"])


def test_help_action_with_invalid_command():
    parser = MagicMock(spec=argparse.ArgumentParser)
    namespace = argparse.Namespace(command="nonexistent")
    action = _HelpAction(option_strings=["-h", "--help"], dest="help")

    mock_commands = {"list": MagicMock()}
    with patch("dqm_ml.__main__.get_available_command", return_value=mock_commands):
        with pytest.raises(ValueError, match="Unkow comand nonexistent"):
            action(parser, namespace, None)


def test_parse_args_invalid_choice():
    command_list = ["list", "version"]
    with patch("sys.stderr", new=MagicMock()):
        with pytest.raises(SystemExit):
            parse_args(["invalid"], command_list)


@patch("dqm_ml.__main__.get_available_command")
@patch("dqm_ml.cli_tools.CustomFormatter.init_log")
def test_execute_verbose(mock_init_log, mock_get_commands):
    mock_cmd = MagicMock()
    mock_get_commands.return_value = {"list": mock_cmd}

    execute(["list", "-v"])

    mock_init_log.assert_called()
    # Check if DEBUG level was used
    _, kwargs = mock_init_log.call_args
    assert kwargs.get("level") == logging.DEBUG


@patch("dqm_ml.__main__.get_available_command")
def test_execute_unknown_command(mock_get_commands):
    # This shouldn't happen usually because parse_args validates choices,
    # but the execute function has a secondary check.
    mock_get_commands.return_value = {"list": MagicMock()}

    # We need to bypass parse_args or mock it to return an invalid command
    with patch("dqm_ml.__main__.parse_args") as mock_parse:
        mock_parse.return_value = (argparse.Namespace(command="unknown", verbose=False, quiet=False), [])
        with pytest.raises(ValueError, match="Unkow comand unknown"):
            execute(["unknown"])
