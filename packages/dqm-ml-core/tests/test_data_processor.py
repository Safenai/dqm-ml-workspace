import pytest

from dqm_ml_core.data_processor import DatametricProcessor


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, 1),
        (2, 2),
        (3, 6),
        (10, 3628800),
    ],
)
def test_factorial(n: int, expected: int) -> None:
    test = DatametricProcessor(name = "test")
    # assert factorial(n) == expected

