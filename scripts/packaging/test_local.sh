#!/usr/bin/env bash
# Test packaging scenarios with locally built wheels.
#
# Usage:
#   scripts/packaging/test_local.sh        # run all 9 scenarios
#   scripts/packaging/test_local.sh 3      # run scenario 3 only

set -euo pipefail
source "$(dirname "$0")/utils.sh"

CORE="dqm-ml-core"
IMAGES="dqm-ml-images"
JOB="dqm-ml-job"
PYTORCH="dqm-ml-pytorch"

setup_tmpdir

WHEELS_DIR="${PACKAGING_DIR}/tmp/wheels"

build_wheels() {
    echo -e "${_BOLD}Building wheels...${_RESET}"
    rm -rf "$WHEELS_DIR"
    mkdir -p "$WHEELS_DIR"
    local root="${PACKAGING_DIR}/../.."
    for pkg in "$CORE" "$IMAGES" "$JOB" "$PYTORCH" dqm-ml; do
        uv build --package "$pkg" --wheel --out-dir "$WHEELS_DIR" --directory "$root" 1>&2
    done
    echo -e "${_GREEN}Wheels built in ${WHEELS_DIR}${_RESET}\n"
}

install_packages() {
    local venv_dir="$1"
    shift
    local packages=("$@")
    local python="$venv_dir/bin/python"
    local install_args=()
    for pkg in "${packages[@]}"; do
        local base="${pkg%%\[*}"
        case "$base" in
            dqm-ml-core)    install_args+=("$WHEELS_DIR"/dqm_ml_core-*.whl) ;;
            dqm-ml-images)  install_args+=("$WHEELS_DIR"/dqm_ml_images-*.whl) ;;
            dqm-ml-job)     install_args+=("$WHEELS_DIR"/dqm_ml_job-*.whl) ;;
            dqm-ml-pytorch) install_args+=("$WHEELS_DIR"/dqm_ml_pytorch-*.whl) ;;
            dqm-ml)
                local extras="${pkg#*dqm-ml}"
                if [[ -n "$extras" ]]; then
                    install_args+=("$(ls "$WHEELS_DIR"/dqm_ml-*.whl)$extras")
                else
                    install_args+=("$WHEELS_DIR"/dqm_ml-*.whl)
                fi
                ;;
            *) echo "Unknown package: $base" >&2; return 1 ;;
        esac
    done
    uv pip install --python "$python" --extra-index-url https://download.pytorch.org/whl/cpu "${install_args[@]}" 1>&2
}

run() {
    local scenario="${1:-0}"
    build_wheels

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

    rm -rf "$WHEELS_DIR"
    summary
}

scenario=$(parse_scenario_arg "${1:-0}" 9)
run "$scenario"
