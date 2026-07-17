#!/usr/bin/env bash
# Test packaging scenarios with pre-release packages from PyPI.
#
# Usage:
#   scripts/packaging/test_pypi_prerelease.sh        # run all 9 scenarios
#   scripts/packaging/test_pypi_prerelease.sh 3      # run scenario 3 only

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
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "${packages[@]}" 1>&2
}

run() {
    local scenario="${1:-0}"

    if [[ "$scenario" == "0" || "$scenario" == "1" ]]; then
        run_scenario "1_core_only" "smoke_core.py" "$CORE"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "2" ]]; then
        run_scenario "2_images" "smoke_images.py" "$CORE" "$IMAGES"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "3" ]]; then
        run_scenario "3_job" "smoke_core_job.py" "$CORE" "$JOB"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "4" ]]; then
        run_scenario "4_images_job" "smoke_images_job.py" "$CORE" "$IMAGES" "$JOB"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "5" ]]; then
        run_scenario "5_embeddings" "smoke_embeddings.py" "$CORE" "$PYTORCH"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "6" ]]; then
        run_scenario "6_gap" "smoke_gap.py" "$CORE" "$PYTORCH"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "7" ]]; then
        run_scenario "7_pytorch_job" "smoke_pytorch.py" "$CORE" "$PYTORCH" "$JOB"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "8" ]]; then
        run_scenario "8_all" "smoke_all.py" "$CORE" "$IMAGES" "$PYTORCH" "$JOB"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "9" ]]; then
        run_scenario "9_notebooks" "smoke_notebooks.py" "$CORE" "$IMAGES" "$PYTORCH" "$JOB" "dqm-ml[notebooks]"
    fi

    summary
}

scenario=$(parse_scenario_arg "${1:-0}" 9)
run "$scenario"
