"""Fixtures for packaging / package-isolation tests.

Builds all wheels once per session and provides helpers for creating isolated
venvs that install only a specific subset of packages.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import venv

import pytest

WHEEL_GLOBS: dict[str, str] = {
    "dqm-ml-core": "dqm_ml_core-*.whl",
    "dqm-ml-images": "dqm_ml_images-*.whl",
    "dqm-ml-job": "dqm_ml_job-*.whl",
    "dqm-ml-pytorch": "dqm_ml_pytorch-*.whl",
    "dqm-ml": "dqm_ml-*.whl",
}


def _build_all_wheels(tmp_path: Path) -> Path:
    """Build wheels for every workspace package into *tmp_path*/wheels.

    Returns the directory containing the built wheels.
    """
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()

    packages = [
        "dqm-ml-core",
        "dqm-ml-images",
        "dqm-ml-job",
        "dqm-ml-pytorch",
        "dqm-ml",
    ]

    for pkg in packages:
        subprocess.run(
            [
                "uv",
                "build",
                "--package",
                pkg,
                "--wheel",
                "--out-dir",
                str(wheels_dir),
            ],
            check=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )

    return wheels_dir


@pytest.fixture(scope="session")
def wheels_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped fixture that builds all workspace wheels once."""
    tmp = tmp_path_factory.mktemp("wheels_session")
    return _build_all_wheels(tmp)


def resolve_wheels(wheels_dir: Path, package_names: list[str]) -> list[tuple[str, str]]:
    """Resolve wheel file paths for the given package names.

    Returns a list of (wheel_path, pip_spec) tuples where pip_spec includes
    any extras syntax (e.g. ``dqm-ml[notebooks]``).
    """
    results: list[tuple[str, str]] = []
    for spec in package_names:
        base_name = spec.split("[", 1)[0]
        glob_pattern = WHEEL_GLOBS[base_name]
        matches = sorted(wheels_dir.glob(glob_pattern))
        if not matches:
            raise FileNotFoundError(f"No wheel found for {base_name!r} matching {glob_pattern}")
        results.append((str(matches[0]), spec))
    return results


def create_venv(venv_dir: Path) -> Path:
    """Create a fresh virtual environment and return the python executable path."""
    venv.create(str(venv_dir), with_pip=True)
    return venv_dir / "bin" / "python"


def install_wheels(python: Path, wheel_specs: list[tuple[str, str]]) -> None:
    """Install the given wheels into the venv.

    *wheel_specs* is a list of (wheel_path, pip_spec) tuples as returned by
    :func:`resolve_wheels`.  The pip_spec includes any extras syntax.
    """
    install_args: list[str] = []
    for wheel_path, pip_spec in wheel_specs:
        if "[" in pip_spec:
            extras = pip_spec.split("[", 1)[1].rstrip("]")
            install_args.append(f"{wheel_path}[{extras}]")
        else:
            install_args.append(wheel_path)
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            *install_args,
        ],
        check=True,
    )


def run_script(python: Path, script: Path) -> subprocess.CompletedProcess:
    """Run a Python script inside the venv."""
    return subprocess.run(
        [str(python), str(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture
def isolated_venv(wheels_dir: Path, tmp_path: Path):
    """Factory fixture: returns a callable that creates an isolated venv.

    Usage::

        venv_python = isolated_venv(["dqm-ml-core"])
    """

    def _make(package_names: list[str]) -> tuple[Path, Path]:
        venv_dir = tmp_path / "venv"
        python = create_venv(venv_dir)
        wheel_specs = resolve_wheels(wheels_dir, package_names)
        install_wheels(python, wheel_specs)
        return python, venv_dir

    return _make
