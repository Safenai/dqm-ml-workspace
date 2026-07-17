#!/usr/bin/env bash
# Test packaging scenarios with pre-release packages from test.pypi.org.
#
# Usage:
#   scripts/packaging/test_testpypi_prerelease.sh        # run all 9 scenarios
#   scripts/packaging/test_testpypi_prerelease.sh 3      # run scenario 3 only

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
        --prerelease=allow \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "${packages[@]}" 1>&2
}

run() {
    run_scenarios "${1:-0}"
    summary
}

scenario=$(parse_scenario_arg "${1:-0}" 9)
run "$scenario"
