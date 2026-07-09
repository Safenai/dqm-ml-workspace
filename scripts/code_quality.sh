#!/bin/bash
set -e

mkdir -p "$(dirname "$0")/../logs"
timestamp=$(date +%Y%m%d_%H%M%S)
exec > >(tee "$(dirname "$0")/../logs/code_quality_${timestamp}.log") 2>&1

CI_MODE="${CI:-false}"

uv sync

if [[ "$CI_MODE" != "true" ]]; then
    uv run nox -s fmt
fi

uv run nox -s lint
uv run nox -s type_check
