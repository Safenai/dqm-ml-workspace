"""Unit tests for the dependency utilities.

This module contains tests for the optional dependency handling and
command availability utilities in the v2 package.
"""

import pytest

from dqm_ml.dependency import get_available_command, optional_dependencies


class TestOptionalDependencies:
    """Tests for optional_dependencies context manager."""

    def test_ignore_error_by_default(self):
        """Verify ImportError is suppressed when mode is 'ignore'."""
        with optional_dependencies("ignore"):
            raise ImportError("missing_pkg")

    def test_warn_error_outputs_warning(self, capsys):
        """Verify ImportError triggers warning output when mode is 'warn'.

        Args:
            capsys: Pytest fixture to capture stdout/stderr.
        """
        with optional_dependencies("warn"):
            raise ImportError("some_package")
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_raise_error_propagates(self):
        """Verify ImportError propagates when mode is 'raise'."""
        with pytest.raises(ImportError, match="missing_pkg"), optional_dependencies("raise"):
            raise ImportError("missing_pkg")

    def test_success_path_does_nothing(self):
        """Verify context manager works normally when no error occurs."""
        with optional_dependencies("raise"):
            pass


class TestGetAvailableCommand:
    """Tests for get_available_command function."""

    def test_returns_process_command(self):
        """Verify returned commands include expected CLI commands."""
        commands = get_available_command()
        assert "version" in commands
        assert "list" in commands
        assert "process" in commands
