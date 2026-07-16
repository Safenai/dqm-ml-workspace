#!/usr/bin/env python3
"""Smoke test: dqm-ml-core only (no job, no images, no pytorch).

Verifies that metrics can be computed via the Python API.
"""

import sys

import numpy as np
import pandas as pd
import pyarrow as pa
from tests.utils.seeds import get_test_seed


def test_completeness():
    from dqm_ml_core import CompletenessProcessor, ProcessorRunner

    df = pd.DataFrame({"col_a": [1, 2, None, 4, 5], "col_b": [1, 2, 3, None, 5]})
    processor = CompletenessProcessor(
        name="test",
        config={
            "columns": {"input": ["col_a", "col_b"]},
            "include_per_column": True,
            "include_overall": True,
        },
    )
    runner = ProcessorRunner()
    result = runner.run(df, [processor])
    assert "completeness_col_a" in result
    assert "completeness_overall" in result


def test_representativeness():
    from dqm_ml_core import RepresentativenessProcessor

    rng = np.random.default_rng(get_test_seed())
    data = {"feature": rng.normal(0, 1, 1000)}
    batch = pa.record_batch(data)
    processor = RepresentativenessProcessor(
        name="test",
        config={
            "columns": {"input": ["feature"]},
            "distribution": "normal",
            "metrics": ["chi-square", "kolmogorov-smirnov"],
            "mean_std_estimation": "from_first_batch",
        },
    )
    features = processor.select_columns(batch, prev_features={})
    batch_metrics = processor.compute_batch_metric(features)
    result = processor.compute(batch_metrics)
    assert "feature_chi-square_statistic" in result


if __name__ == "__main__":
    try:
        test_completeness()
        test_representativeness()
        print("dqm-ml-core smoke test PASSED")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
