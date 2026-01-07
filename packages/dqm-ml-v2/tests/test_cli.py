import shlex
import subprocess

from dqm_ml.__main__ import execute, parse_args
from dqm_ml._version_ import version
import pytest

test_cases = [
    ("version", f"DQML version : {version}"),  # no args
    (
        "list",
        "Available data metrics_registry\n"
        "- completeness - <class 'dqm_ml_core.CompletenessProcessor'>\n"
        "- representativeness - <class 'dqm_ml_core.RepresentativenessProcessor'>\n"
        "- visual_metrique - <class 'dqm_ml_images.visual_features.VisualFeaturesProcessor'>\n"
        "Available data loaders\n"
        "- parquet - <class 'dqm_ml_pipeline.dataloaders.parquet.ParquetDataLoader'>\n"
        "Available outputs writter\n"
        "- parquet - <class 'dqm_ml_pipeline.outputwriter.parquet.ParquetOutputWriter'>",
    ),
]
command_list = {"version": None}


@pytest.mark.parametrize(("command", "expected_output"), test_cases)
def test_main(capsys: pytest.CaptureFixture[str], command: str, expected_output: str) -> None:
    execute(shlex.split(command))
    output = capsys.readouterr().out.rstrip()
    assert output == expected_output


@pytest.mark.parametrize(("command", "expected_output"), test_cases)
def test_app(command: str, expected_output: str) -> None:
    full_command = ["dqm-ml"] + shlex.split(command)
    result = subprocess.run(full_command, capture_output=True, text=True)
    output = result.stdout.rstrip()
    assert output == expected_output


@pytest.mark.parametrize(
    ("prompt", "command", "verbose", "quiet"),
    [
        # no params
        ("version", "version", False, False),
        # short params
        ("version -q", "version", False, True),
        ("version -v", "version", True, False),
        # long params TODO
    ],
)
def test_parse_args(prompt: str, command: str, quiet: str, verbose: str) -> None:
    args, _ = parse_args(shlex.split(prompt), command_list)

    # or split them up, either works
    assert args.command == command
    assert args.quiet == quiet
    assert args.verbose == verbose
