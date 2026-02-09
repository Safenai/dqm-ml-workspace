import logging

import pyarrow as pa
import pytest

from dqm_ml_core import DatametricProcessor


def test_processor_init_defaults():
    proc = DatametricProcessor(name="test", config=None)
    assert proc.name == "test"
    assert proc.config == {}
    assert proc.input_columns == []
    assert proc.outputs_columns == {}

def test_processor_init_with_config():
    config = {
        "input_columns": ["col1", "col2"],
        "output_columns": {"m1": "res1"}
    }
    proc = DatametricProcessor(name="test", config=config)
    assert proc.input_columns == ["col1", "col2"]
    assert proc.outputs_columns == {"m1": "res1"}

def test_processor_init_invalid_input_columns():
    with pytest.raises(ValueError, match="need 'input_columns'"):
        DatametricProcessor(name="test", config={"input_columns": "not_a_list"})

def test_processor_init_invalid_output_columns():
    with pytest.raises(ValueError, match="need of 'output_columns'"):
        DatametricProcessor(name="test", config={"output_columns": ["not_a_dict"]})

def test_processor_needed_columns():
    proc = DatametricProcessor(name="test", config={"input_columns": ["a"]})
    assert proc.needed_columns() == ["a"]
    
    # Test getattr fallback
    del proc.input_columns
    assert proc.needed_columns() == []

def test_processor_generated_features():
    proc = DatametricProcessor(name="test", config={})
    assert proc.generated_features() == []
    
    proc.output_features = {"f1": "feat_out"}
    assert proc.generated_features() == ["feat_out"]

def test_processor_generated_metrics():
    proc = DatametricProcessor(name="test", config={})
    assert proc.generated_metrics() == []
    
    proc.output_metrics = {"m1": "metric_out"}
    assert proc.generated_metrics() == ["metric_out"]

def test_processor_compute_features(caplog):
    proc = DatametricProcessor(name="test", config={"input_columns": ["col1", "col2", "missing"]})
    
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2]), pa.array([3, 4])],
        names=["col1", "col2"]
    )
    
    prev_features = {"col1": pa.array([1, 2])}
    
    with caplog.at_level(logging.WARNING):
        features = proc.compute_features(batch, prev_features)
    
    # col1 is in prev_features, so skipped
    # col2 is in batch, so added
    # missing is not in batch, so logged and skipped
    assert "col1" not in features
    assert "col2" in features
    assert features["col2"].to_pylist() == [3, 4]
    assert "column 'missing' not found in batch" in caplog.text

def test_processor_default_implementations():
    proc = DatametricProcessor(name="test", config={})
    assert proc.compute_batch_metric({}) == {}
    assert proc.compute({}) == {}
    assert proc.compute_delta({}, {}) == {}