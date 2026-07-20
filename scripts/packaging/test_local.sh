#!/usr/bin/env bash
# Test packaging scenarios with locally built wheels.
#
# Usage:
#   scripts/packaging/test_local.sh         # run all 13 scenarios
#   scripts/packaging/test_local.sh 10      # run scenario 10 only

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
    run_scenarios "$scenario"
    rm -rf "$WHEELS_DIR"
    summary
}

scenario=$(parse_scenario_arg "${1:-0}" 13)
run "$scenario"
