from pathlib import Path
import shlex
import subprocess

import pytest

from dqm_ml_pipeline.cli import execute, parse_args

test_cases = [
    ("-p packages/dqm-ml-pipeline/config/completeness.yaml", ""),  # no args
    ("-p packages/dqm-ml-pipeline/config/representativness.yaml", ""),  # no args
]


@pytest.mark.parametrize(("command", "expected_output"), test_cases)
def test_main(capsys: pytest.CaptureFixture[str], command: str, expected_output: str) -> None:
    # Create output dir if it doesn't exist
    path = Path("output")
    Path.mkdir(path, exist_ok=True)

    execute(shlex.split(command))
    output = capsys.readouterr().out.rstrip()
    assert output == expected_output


@pytest.mark.parametrize(("command", "expected_output"), test_cases)
def test_app(command: str, expected_output: str) -> None:
    full_command = ["python", "hello.py"] + shlex.split(command)
    result = subprocess.run(full_command, capture_output=True, text=True)
    output = result.stdout.rstrip()
    assert output == expected_output


@pytest.mark.parametrize(
    ("prompt", "pipeline"),
    [
        # no params
        ("", "pipeline.yaml"),
        # short params
        ("-p dummy.yaml", "dummy.yaml"),
        # long params TODO
    ],
)
def test_parse_args(prompt: str, pipeline: str) -> None:
    args = parse_args(shlex.split(prompt))

    # or split them up, either works
    assert args.pipeline == pipeline
