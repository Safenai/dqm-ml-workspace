#!/usr/bin/env bash
# Test packaging scenarios with release packages from PyPI.
#
# Usage:
#   scripts/packaging/test_pypi_release.sh        # run all 13 scenarios
#   scripts/packaging/test_pypi_release.sh 10     # run scenario 10 only

set -euo pipefail
source "$(dirname "$0")/utils.sh"

CORE="dqm-ml-core"
IMAGES="dqm-ml-images"
JOB="dqm-ml-job"
PYTORCH="dqm-ml-pytorch"

setup_tmpdir

install_packages() {
    local venv_dir="$1"
    shift
    local packages=("$@")
    local python="$venv_dir/bin/python"
    uv pip install --python "$python" \
        --index-url https://pypi.org/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "${packages[@]}" 1>&2
}

run() {
    run_scenarios "${1:-0}"
    summary
}

scenario=$(parse_scenario_arg "${1:-0}" 13)
run "$scenario"
