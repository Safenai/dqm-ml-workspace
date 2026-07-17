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
