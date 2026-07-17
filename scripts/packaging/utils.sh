#!/usr/bin/env bash
# Shared functions for packaging test scripts.
# Source this file: source "$(dirname "$0")/utils.sh"

set -euo pipefail

PACKAGING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${PACKAGING_DIR}/smoke"
VENV_BASE="${PACKAGING_DIR}/tmp/venvs"

# Counters
_passed=0
_failed=0
_skipped=0
_current_scenario=""

# ── Colors ──────────────────────────────────────────────────────────────────
_RED='\033[0;31m'
_GREEN='\033[0;32m'
_YELLOW='\033[0;33m'
_CYAN='\033[0;36m'
_BOLD='\033[1m'
_RESET='\033[0m'

# ── Helpers ─────────────────────────────────────────────────────────────────

setup_tmpdir() {
    export TMPDIR="${PACKAGING_DIR}/tmp"
    mkdir -p "$TMPDIR"
}

# run_scenario <name> <script> <packages...>
#   Installs packages in an isolated venv and runs the smoke test.
#   The install function (install_packages) must be defined by the caller.
run_scenario() {
    local name="$1"
    local script="$2"
    shift 2
    local packages=("$@")

    _current_scenario="$name"
    local venv_dir="${VENV_BASE}/${name}"

    echo -e "\n${_BOLD}${_CYAN}── Scenario ${name} ──${_RESET}"

    # Create venv
    rm -rf "$venv_dir"
    mkdir -p "$venv_dir"
    uv venv --python 3.12 "$venv_dir" 1>&2 || { fail "$name" "venv creation failed"; return; }

    # Install packages (caller defines install_packages)
    if ! install_packages "$venv_dir" "${packages[@]}"; then
        fail "$name" "package installation failed"
        rm -rf "$venv_dir"
        return
    fi

    # Copy smoke script + utils into venv
    cp "$SCRIPTS_DIR/$script" "$venv_dir/$script"
    cp "$SCRIPTS_DIR/utils.py" "$venv_dir/utils.py"

    # Run smoke test
    if "$venv_dir/bin/python" "$venv_dir/$script"; then
        pass "$name"
    else
        fail "$name" "smoke test failed"
    fi

    rm -rf "$venv_dir"
}

pass() {
    local name="$1"
    _passed=$((_passed + 1))
    echo -e "${_GREEN}✓ PASS${_RESET} ${name}"
}

fail() {
    local name="$1"
    local reason="${2:-}"
    _failed=$((_failed + 1))
    echo -e "${_RED}✗ FAIL${_RESET} ${name} ${reason:+— ${reason}}"
}

summary() {
    echo -e "\n${_BOLD}── Summary ──${_RESET}"
    echo -e "${_GREEN}Passed: ${_passed}${_RESET}"
    echo -e "${_RED}Failed: ${_failed}${_RESET}"
    if [[ $_failed -gt 0 ]]; then
        echo -e "${_RED}${_BOLD}Some scenarios failed.${_RESET}"
        return 1
    else
        echo -e "${_GREEN}${_BOLD}All scenarios passed.${_RESET}"
        return 0
    fi
}

# run_scenarios <scenario>
#   Run scenarios matching <scenario> (0 = all).
#   Requires $CORE, $IMAGES, $JOB, $PYTORCH to be set by the caller.
run_scenarios() {
    local scenario="${1:-0}"
    if [[ "$scenario" == "0" || "$scenario" == "1" ]]; then run_scenario "1_core_only"      "smoke_core.py"       "$CORE"; fi
    if [[ "$scenario" == "0" || "$scenario" == "2" ]]; then run_scenario "2_images"          "smoke_images.py"     "$CORE" "$IMAGES"; fi
    if [[ "$scenario" == "0" || "$scenario" == "3" ]]; then run_scenario "3_job"             "smoke_core_job.py"   "$CORE" "$JOB"; fi
    if [[ "$scenario" == "0" || "$scenario" == "4" ]]; then run_scenario "4_images_job"      "smoke_images_job.py" "$CORE" "$IMAGES" "$JOB"; fi
    if [[ "$scenario" == "0" || "$scenario" == "5" ]]; then run_scenario "5_embeddings"      "smoke_embeddings.py" "$CORE" "$PYTORCH"; fi
    if [[ "$scenario" == "0" || "$scenario" == "6" ]]; then run_scenario "6_gap"             "smoke_gap.py"        "$CORE" "$PYTORCH"; fi
    if [[ "$scenario" == "0" || "$scenario" == "7" ]]; then run_scenario "7_pytorch_job"     "smoke_pytorch.py"    "$CORE" "$PYTORCH" "$JOB"; fi
    if [[ "$scenario" == "0" || "$scenario" == "8" ]]; then run_scenario "8_all"             "smoke_all.py"        "$CORE" "$IMAGES" "$PYTORCH" "$JOB"; fi
    if [[ "$scenario" == "0" || "$scenario" == "9" ]]; then run_scenario "9_notebooks"       "smoke_notebooks.py"  "$CORE" "$IMAGES" "$PYTORCH" "$JOB" "dqm-ml[notebooks]"; fi
}

# parse_scenario_arg <arg> <total>
#   Returns scenario number or 0 for "all".
parse_scenario_arg() {
    local arg="${1:-0}"
    local total="$2"
    if [[ "$arg" =~ ^[0-9]+$ ]] && [[ "$arg" -ge 1 ]] && [[ "$arg" -le "$total" ]]; then
        echo "$arg"
    elif [[ "$arg" == "0" ]]; then
        echo "0"
    else
        echo "Invalid scenario: $arg (must be 1-${total})" >&2
        exit 1
    fi
}
