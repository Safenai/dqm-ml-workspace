"""CLI test fixtures."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="session")
def ensure_example_data() -> None:
    """Generate example data if not present."""
    data_file = Path("examples/data/large_test_2m.parquet")
    if not data_file.exists():
        subprocess.run(
            [sys.executable, "examples/script/generate_data.py"],
            check=True,
        )
