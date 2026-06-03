#!/bin/bash
set -e

CI_MODE="${CI:-false}"

uv sync

if [ "$CI_MODE" != "true" ]; then
    uv run nox -s fmt
fi

uv run nox -s lint
uv run nox -s type_check
