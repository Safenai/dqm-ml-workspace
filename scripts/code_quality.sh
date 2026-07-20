#!/bin/bash
set -e

mkdir -p "$(dirname "$0")/../logs"
timestamp=$(date +%Y%m%d_%H%M%S)
exec > >(tee "$(dirname "$0")/../logs/code_quality_${timestamp}.log") 2>&1

CI_MODE="${CI:-false}"

uv sync --frozen

if [[ "$CI_MODE" != "true" ]]; then
    uv run --frozen nox -s fmt
fi

uv run --frozen nox -s lint
uv run --frozen nox -s type_check
