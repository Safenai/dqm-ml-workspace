"""Unit tests for the Processor base class shared plumbing.

This module contains tests that verify the Processor base class
correctly handles initialization, needed_columns, reset, and
failure-rate checking.
"""

from dqm_ml_core.api.processor import Processor
from dqm_ml_core.models.global_ import ErrorsConfig
import pytest


def test_needed_columns_default():
    """Verify needed_columns returns input_columns by default."""
    proc = Processor(name="test", config={"columns": {"input": ["a", "b"]}})
    assert proc.needed_columns() == ["a", "b"]


def test_needed_columns_no_input():
    """Verify needed_columns returns empty list when no input columns configured."""
    proc = Processor(name="test", config={})
    assert proc.needed_columns() == []


def test_reset_is_noop():
    """Verify reset does not raise."""
    proc = Processor(name="test", config={"columns": {"input": ["a"]}})
    proc.reset()  # should not raise


def test_init_parses_exclude_columns():
    """Verify exclude_columns parsed from config."""
    proc = Processor(name="test", config={"columns": {"input": ["a", "b"], "exclude": ["b"]}})
    assert proc.input_columns == ["a", "b"]
    assert proc.exclude_columns == ["b"]


def test_init_no_columns():
    """Verify input_columns is empty list when no columns config given."""
    proc = Processor(name="test", config={})
    assert proc.input_columns == []
    assert proc.exclude_columns is None


def test_check_failure_rate_does_not_raise_below_threshold():
    """Verify no error when failure rate is below threshold."""
    proc = Processor(name="test", config={"errors": {"max_failure_rate": 0.5}})
    proc.errors_config = ErrorsConfig(max_failure_rate=0.5)
    proc._failure_count = 1
    proc._total_count = 10
    proc._check_failure_rate()  # should not raise


def test_check_failure_rate_exceeds_threshold():
    """Verify RuntimeError when failure rate exceeds threshold."""
    proc = Processor(name="test", config={"errors": {"max_failure_rate": 0.1}})
    proc.errors_config = ErrorsConfig(max_failure_rate=0.1)
    proc._failure_count = 3
    proc._total_count = 10
    with pytest.raises(RuntimeError, match="Failure rate"):
        proc._check_failure_rate()
