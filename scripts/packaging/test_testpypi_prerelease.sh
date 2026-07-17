#!/usr/bin/env bash
# Test packaging scenarios with pre-release packages from test.pypi.org.
#
# Usage:
#   scripts/packaging/test_testpypi_prerelease.sh        # run all 9 scenarios
#   scripts/packaging/test_testpypi_prerelease.sh 3      # run scenario 3 only

set -euo pipefail
source "$(dirname "$0")/utils.sh"

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
    local scenario="${1:-0}"

    if [[ "$scenario" == "0" || "$scenario" == "1" ]]; then
        run_scenario "1_core_only" "smoke_core.py" "dqm-ml-core"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "2" ]]; then
        run_scenario "2_images" "smoke_images.py" "dqm-ml-core" "dqm-ml-images"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "3" ]]; then
        run_scenario "3_job" "smoke_core_job.py" "dqm-ml-core" "dqm-ml-job"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "4" ]]; then
        run_scenario "4_images_job" "smoke_images_job.py" "dqm-ml-core" "dqm-ml-images" "dqm-ml-job"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "5" ]]; then
        run_scenario "5_embeddings" "smoke_embeddings.py" "dqm-ml-core" "dqm-ml-pytorch"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "6" ]]; then
        run_scenario "6_gap" "smoke_gap.py" "dqm-ml-core" "dqm-ml-pytorch"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "7" ]]; then
        run_scenario "7_pytorch_job" "smoke_pytorch.py" "dqm-ml-core" "dqm-ml-pytorch" "dqm-ml-job"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "8" ]]; then
        run_scenario "8_all" "smoke_all.py" "dqm-ml-core" "dqm-ml-images" "dqm-ml-pytorch" "dqm-ml-job"
    fi
    if [[ "$scenario" == "0" || "$scenario" == "9" ]]; then
        run_scenario "9_notebooks" "smoke_notebooks.py" "dqm-ml-core" "dqm-ml-images" "dqm-ml-pytorch" "dqm-ml-job" "dqm-ml[notebooks]"
    fi

    summary
}

scenario=$(parse_scenario_arg "${1:-0}" 9)
run "$scenario"
