"""Package-isolation tests.

Each test creates a fresh virtual environment, installs **only** the specified
subset of dqm-ml wheels, and runs a smoke-test script inside that venv.  This
ensures that:

1. Each package can be installed independently.
2. No unintended transitive imports pull in packages that should be optional.
3. The documented installation recipes actually work.

Run with::

    pytest -m packaging tests/packaging/
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

SCRIPTS_DIR = Path(__file__).parent / "scripts"

# Each entry: (test-id, packages-to-install, script-to-run)
SCENARIOS: list[tuple[str, list[str], str]] = [
    # --- Metrics ---
    ("metrics_only", ["dqm-ml-core"], "smoke_core.py"),
    ("metrics_via_job", ["dqm-ml-core", "dqm-ml-job"], "smoke_core_job.py"),
    # --- Visual features ---
    ("visual_features_only", ["dqm-ml-core", "dqm-ml-images"], "smoke_images.py"),
    ("visual_features_job", ["dqm-ml-core", "dqm-ml-images", "dqm-ml-job"], "smoke_images_job.py"),
    # --- Embeddings ---
    ("embeddings_only", ["dqm-ml-core", "dqm-ml-pytorch"], "smoke_embeddings.py"),
    # --- Gap metrics ---
    ("gap_only", ["dqm-ml-core", "dqm-ml-pytorch"], "smoke_gap.py"),
    # --- Embeddings + gap via job ---
    ("embeddings_gap_job", ["dqm-ml-core", "dqm-ml-pytorch", "dqm-ml-job"], "smoke_pytorch.py"),
    # --- Everything (no notebooks) ---
    (
        "all_no_notebooks",
        ["dqm-ml-core", "dqm-ml-images", "dqm-ml-pytorch", "dqm-ml-job"],
        "smoke_all.py",
    ),
    # --- Everything + notebooks ---
    (
        "all_with_notebooks",
        ["dqm-ml-core", "dqm-ml-images", "dqm-ml-pytorch", "dqm-ml-job", "dqm-ml[notebooks]"],
        "smoke_notebooks.py",
    ),
]


@pytest.mark.packaging
@pytest.mark.parametrize(
    ("packages", "script_name"),
    [item[1:] for item in SCENARIOS],
    ids=[item[0] for item in SCENARIOS],
)
def test_package_installation(
    packages: list[str],
    script_name: str,
    wheels_dir: Path,
    tmp_path: Path,
) -> None:
    """Install *packages* in an isolated venv and run *script_name*."""
    from tests.fixtures.packaging_fixtures import create_venv, install_wheels, resolve_wheels, run_script

    venv_dir = tmp_path / "venv"
    python = create_venv(venv_dir)

    wheel_specs = resolve_wheels(wheels_dir, packages)
    install_wheels(python, wheel_specs)

    # Copy the smoke-test script into the venv working directory so it can
    # find its sibling scripts if needed.
    script_src = SCRIPTS_DIR / script_name
    script_dst = venv_dir / script_name
    shutil.copy2(script_src, script_dst)

    result = run_script(python, script_dst)

    assert result.returncode == 0, (
        f"Smoke test {script_name!r} failed in venv with packages={packages}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
