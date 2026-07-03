"""Unit tests for the DQM job CLI.

This module contains unit tests that verify the dqm-ml-job CLI
correctly parses arguments and executes jobs.
"""

from pathlib import Path
import shlex
import subprocess
from unittest.mock import MagicMock, mock_open, patch

from dqm_ml_job.cli import _init_components_from_list, _merge_errors, execute, parse_args, run
import pytest
import yaml

test_cases = [
    ("-p examples/config/completeness.yaml", ""),  # no args
    ("-p examples/config/representativeness.yaml", ""),  # no args
]


@pytest.mark.parametrize(("command", "expected_output"), test_cases)
def test_main(
    ensure_example_data: None,
    capsys: pytest.CaptureFixture[str],
    command: str,
    expected_output: str,
) -> None:
    """Test that the CLI execute function runs without errors."""
    # Create output dir if it doesn't exist
    path = Path("tests/outputs/data")
    Path.mkdir(path, exist_ok=True, parents=True)

    execute(shlex.split(command))
    output = capsys.readouterr().out.rstrip()
    assert output == expected_output


@pytest.mark.parametrize(("command", "expected_output"), test_cases)
def test_app(ensure_example_data: None, command: str, expected_output: str) -> None:
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


class TestExecuteEdgeCases:
    def test_yaml_parse_error(self, caplog):
        with (
            patch("builtins.open", mock_open(read_data="{invalid: yaml: [")),
            patch("dqm_ml_job.cli.Path.open", mock_open(read_data="{invalid: yaml: [")),
            patch("dqm_ml_job.cli.yaml.safe_load", side_effect=yaml.YAMLError("parse error")),
        ):
            execute(["-p", "config.yaml"])

        # should not raise

    def test_save_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        save_path = tmp_path / "saved.yaml"
        config_path.write_text("dataloaders:\n  loaders:\n    - path: data.parquet\n")

        with patch("dqm_ml_job.cli.run") as mock_run:
            execute(["-p", str(config_path), "--save-config", str(save_path)])

        assert save_path.exists()
        saved = yaml.safe_load(save_path.read_text())
        assert "dataloaders" in saved
        mock_run.assert_called_once()


class TestInitComponentsFromList:
    def test_missing_name(self):
        mock_registry = {"completeness": MagicMock()}
        with pytest.raises(ValueError, match="must contain 'name'"):
            _init_components_from_list([{"type": "completeness"}], mock_registry, "processor")

    def test_missing_type(self):
        mock_registry = {"completeness": MagicMock()}
        with pytest.raises(ValueError, match="must contain 'type'"):
            _init_components_from_list([{"name": "test"}], mock_registry, "processor")

    def test_invalid_type(self):
        mock_registry = {"completeness": MagicMock()}
        with pytest.raises(ValueError, match="has invalid type"):
            _init_components_from_list([{"name": "test", "type": "nonexistent"}], mock_registry, "processor")

    def test_success(self):
        mock_cls = MagicMock()
        components = _init_components_from_list(
            [{"name": "test", "type": "completeness"}], {"completeness": mock_cls}, "processor"
        )
        assert "test" in components
        mock_cls.assert_called_once_with(name="test", config={"name": "test", "type": "completeness"})


class TestRunEdgeCases:
    def test_empty_config(self):
        with pytest.raises(ValueError, match="Job requires a configuration dictionary"):
            run({})

    def test_run_with_valid_config(self):
        config = {
            "dataloaders": {"loaders": [{"path": "dummy.parquet"}]},
            "compute": {"seed": 42},
        }
        with patch("dqm_ml_job.cli.PluginLoadedRegistry.get_dataloaders_registry") as mock_dl_reg:
            mock_dl_reg.return_value = {}
            with pytest.raises(ValueError, match="Field required"):
                run(config)


class TestMergeErrors:
    def test_both_none(self):
        from dqm_ml_core.models.global_ import ErrorsConfig

        result = _merge_errors(None, None)
        assert result.default == ErrorsConfig.model_fields["default"].default
        assert result.images is None

    def test_interface_takes_precedence(self):
        from dqm_ml_core.models.global_ import ErrorsConfig, ImageErrorsConfig

        global_errors = ErrorsConfig(default="silent_fail", images=ImageErrorsConfig())
        interface_errors = ErrorsConfig(default="fail_fast")
        result = _merge_errors(global_errors, interface_errors)
        assert result.default == "fail_fast"
        assert isinstance(result.images, ImageErrorsConfig)
