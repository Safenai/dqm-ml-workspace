"""Test seed utilities for configurable random seeds."""

import os


def get_test_seed() -> int:
    """Get test seed from environment variable, default 42."""
    return int(os.environ.get("DQM_ML_TEST_SEED", "42"))
