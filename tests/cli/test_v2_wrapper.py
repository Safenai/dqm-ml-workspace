"""Unit tests for the DQM-ML CLI wrapper.

This module contains unit tests that verify the dqm-ml CLI
correctly parses arguments, displays version, and lists available plugins.
"""

import shlex
import subprocess

import pytest

from dqm_ml.__main__ import execute, parse_args
from dqm_ml_core._version_ import version

test_cases = [
    ("version", f"DQML version : {version}"),  # no args
    [
        "list",
        "Available data metrics_registry\n"
        "- completeness - <class 'dqm_ml_core.metrics.completeness.CompletenessProcessor'>\n"
        "- diversity - <class 'dqm_ml_core.metrics.diversity.DiversityProcessor'>\n"
        "- representativeness - <class 'dqm_ml_core.metrics.representativeness.RepresentativenessProcessor'>\n"
        "- image_features - <class 'dqm_ml_images.visual_features.VisualFeaturesProcessor'>\n"
        "- domain_gap - <class 'dqm_ml_pytorch.domain_gap.DomainGapProcessor'>\n"
        "- features_embeddings - <class 'dqm_ml_pytorch.image_embedding.ImageEmbeddingProcessor'>\n"
        "Available data loaders\n"
        "- csv - <class 'dqm_ml_job.dataloaders.pandas.PandasDataLoader'>\n"
        "- parquet - <class 'dqm_ml_job.dataloaders.parquet.ParquetDataLoader'>\n"
        "Available outputs writers\n"
        "- parquet - <class 'dqm_ml_job.outputwriter.parquet.ParquetOutputWriter'>",
    ],
]
command_list = {"version": None}


@pytest.mark.parametrize(("command", "expected_output"), test_cases)
def test_main(capsys: pytest.CaptureFixture[str], command: str, expected_output: list[str]) -> None:
    """Test that the v2 CLI execute function runs correctly."""
    execute(shlex.split(command))
    output = capsys.readouterr().out.rstrip()
    assert all(val in output for val in expected_output)


@pytest.mark.parametrize(("command", "expected_output"), test_cases)
def test_app(command: str, expected_output: list[str]) -> None:
    """Test that the v2 CLI can be invoked as a Python module."""
    import sys

    full_command = [sys.executable, "-m", "dqm_ml"] + shlex.split(command)
    result = subprocess.run(full_command, capture_output=True, text=True)
    output = result.stdout.rstrip()
    assert all(val in output for val in expected_output)


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
    """Test that parse_args correctly extracts command, verbose, and quiet flags."""
    args, _ = parse_args(shlex.split(prompt), command_list)

    # or split them up, either works
    assert args.command == command
    assert args.quiet == quiet
    assert args.verbose == verbose
