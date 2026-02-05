from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pytest

from dqm_ml_core.api.data_processor import DatametricProcessor
from dqm_ml_core.utils.metric_runner import MetricRunner


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_metric_runner_init():
    runner = MetricRunner(config={})
    assert isinstance(runner, MetricRunner)


def test_metric_runner_run_empty_df():
    runner = MetricRunner()
    df = pd.DataFrame(columns=["a", "b"])
    metrics = []
    result = runner.run(df, metrics)
    assert result == {}


def test_metric_runner_run_with_mock_metric(sample_df):
    runner = MetricRunner()

    mock_metric = MagicMock(spec=DatametricProcessor)
    mock_metric.compute_features.return_value = {"feat1": pa.array([1, 1, 1])}
    mock_metric.compute_batch_metric.return_value = {"metric1": pa.array([10])}
    mock_metric.compute.return_value = {"final_metric1": 0.95}

    result = runner.run(sample_df, [mock_metric])

    assert result == {"final_metric1": 0.95}
    mock_metric.compute_features.assert_called_once()
    mock_metric.compute_batch_metric.assert_called_once()
    mock_metric.compute.assert_called_once()


def test_metric_runner_overwrite_behavior(sample_df):
    """
    Test that if multiple metrics share keys, the later one overwrites.
    This documents the current behavior.
    """
    runner = MetricRunner()

    metric1 = MagicMock(spec=DatametricProcessor)
    metric1.compute_features.return_value = {}
    metric1.compute_batch_metric.return_value = {"shared": pa.array([1])}
    metric1.compute.return_value = {}

    metric2 = MagicMock(spec=DatametricProcessor)
    metric2.compute_features.return_value = {}
    metric2.compute_batch_metric.return_value = {"shared": pa.array([2])}
    metric2.compute.return_value = {"final": 1}

    result = runner.run(sample_df, [metric1, metric2])

    # Verify overwriting happened in the internal metrics_array
    last_call_args = metric2.compute.call_args[1]["batch_metrics"]
    assert len(last_call_args["shared"]) == 1
    assert last_call_args["shared"].to_pylist() == [2]

    # Verify the final result content
    assert result == {"final": 1}
