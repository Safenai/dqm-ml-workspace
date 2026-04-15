"""Unit tests for the DQM job CLI.

This module contains unit tests that verify the dqm-ml-job CLI
correctly parses arguments and executes jobs.
"""

from pathlib import Path
import shlex
import subprocess

import pytest

from dqm_ml_job.cli import execute, parse_args

test_cases = [
    ("-p examples/config/completeness.yaml", ""),  # no args
    ("-p examples/config/representativness.yaml", ""),  # no args
]


@pytest.mark.parametrize(("command", "expected_output"), test_cases)
def test_main(capsys: pytest.CaptureFixture[str], command: str, expected_output: str) -> None:
    """Test that the CLI execute function runs without errors."""
    # Create output dir if it doesn't exist
    path = Path("tests/outputs/data")
    Path.mkdir(path, exist_ok=True, parents=True)

    execute(shlex.split(command))
    output = capsys.readouterr().out.rstrip()
    assert output == expected_output


@pytest.mark.parametrize(("command", "expected_output"), test_cases)
def test_app(command: str, expected_output: str) -> None:
    """Test that the CLI can be invoked as a subprocess."""
    full_command = ["python", "hello.py"] + shlex.split(command)
    result = subprocess.run(full_command, capture_output=True, text=True)
    output = result.stdout.rstrip()
    assert output == expected_output


@pytest.mark.parametrize(
    ("prompt", "process_config"),
    [
        # short params
        ("-p dummy.yaml", ["dummy.yaml"]),
        # long params TODO
    ],
)
def test_parse_args(prompt: str, process_config: str) -> None:
    """Test that parse_args correctly extracts configuration file paths."""
    args = parse_args(shlex.split(prompt))

    # or split them up, either works
    assert args.process_config == process_config
