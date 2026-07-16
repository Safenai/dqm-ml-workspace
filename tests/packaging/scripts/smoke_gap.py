#!/usr/bin/env python3
"""Smoke test: dqm-ml-core + dqm-ml-pytorch (no job).

Verifies that gap metrics can be computed with pre-computed embeddings.
"""

import sys

import numpy as np
import pandas as pd
from tests.utils.seeds import get_test_seed


def test_domain_gap_mmd_linear():
    from dqm_ml_core import ProcessorRunner
    from dqm_ml_pytorch import DomainGapProcessor

    rng = np.random.default_rng(get_test_seed())
    source_emb = rng.standard_normal((100, 128)).astype(np.float32)
    target_emb = rng.standard_normal((100, 128)).astype(np.float32)

    source_df = pd.DataFrame({"embedding": list(source_emb)})
    target_df = pd.DataFrame({"embedding": list(target_emb)})

    processor = DomainGapProcessor(
        name="test",
        config={"columns": {"input": ["embedding"]}, "distance": {"metric": "mmd_linear"}},
    )
    runner = ProcessorRunner()
    result = runner.run_gap(source_df, target_df, processor)
    assert "mmd_linear" in result
    assert result["mmd_linear"][0].as_py() >= 0


def test_domain_gap_mmd_rbf():
    from dqm_ml_core import ProcessorRunner
    from dqm_ml_pytorch import DomainGapProcessor

    rng = np.random.default_rng(get_test_seed())
    source_emb = rng.standard_normal((50, 128)).astype(np.float32)
    target_emb = rng.standard_normal((50, 128)).astype(np.float32)

    source_df = pd.DataFrame({"embedding": list(source_emb)})
    target_df = pd.DataFrame({"embedding": list(target_emb)})

    processor = DomainGapProcessor(
        name="test",
        config={
            "columns": {"input": ["embedding"]},
            "distance": {"metric": "mmd_rbf", "kernel_params": {"gamma": 0.1}},
        },
    )
    runner = ProcessorRunner()
    result = runner.run_gap(source_df, target_df, processor)
    assert "mmd_rbf" in result
    assert result["mmd_rbf"][0].as_py() >= 0


if __name__ == "__main__":
    try:
        test_domain_gap_mmd_linear()
        test_domain_gap_mmd_rbf()
        print("dqm-ml-pytorch gap smoke test PASSED")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
