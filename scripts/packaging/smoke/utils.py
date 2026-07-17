"""Shared utilities for packaging smoke tests."""

from __future__ import annotations

import os


def get_test_seed() -> int:
    """Return the test seed from the environment, defaulting to 42."""
    return int(os.environ.get("DQM_ML_TEST_SEED", "42"))
