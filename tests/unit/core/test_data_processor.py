from typing import Any

import pytest

from dqm_ml_core import DatametricProcessor


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("void", {}),
    ],
)
def test_processor(name: str, config: dict[str, Any]) -> None:
    test = DatametricProcessor(name=name, config=config)

    assert test.name == name
