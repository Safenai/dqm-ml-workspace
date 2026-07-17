#!/usr/bin/env python3
"""Smoke test: all packages + notebooks.

Verifies that dqm-ml[all] can be installed and notebook dependencies are importable.
"""

import sys


def test_notebook_imports():
    """Verify that optional notebook dependencies are importable."""
    import jupyter  # noqa: F401
    import matplotlib  # noqa: F401
    import plotly  # noqa: F401
    import tabulate  # noqa: F401


def test_dqm_ml_all_imports():
    """Verify that all dqm-ml packages can be imported together."""
    import dqm_ml  # noqa: F401
    import dqm_ml_core  # noqa: F401
    import dqm_ml_images  # noqa: F401
    import dqm_ml_job  # noqa: F401
    import dqm_ml_pytorch  # noqa: F401


def test_cli_version():
    """Verify dqm-ml CLI is accessible."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "dqm_ml", "version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI version failed: {result.stderr}"
    assert result.stdout.strip(), "CLI version returned empty output"


if __name__ == "__main__":
    try:
        test_notebook_imports()
        test_dqm_ml_all_imports()
        test_cli_version()
        print("Notebooks smoke test PASSED")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
