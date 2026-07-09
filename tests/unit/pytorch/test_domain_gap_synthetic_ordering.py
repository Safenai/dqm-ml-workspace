"""Synthetic ordering tests for domain gap metrics.

Tests that all domain gap metrics produce values that are monotonic with
increasing distribution shift, using controlled synthetic embedding data
directly (no image pipeline needed).
"""

from dqm_ml_pytorch.domain_gap import DomainGapProcessor
import numpy as np
import pyarrow as pa
import pytest

N_SAMPLES = 500
EMBED_DIM = 32
SHIFTS = [0.0, 0.1, 0.5, 2.0]


def _summary_dicts(src: np.ndarray, tgt: np.ndarray, include_outer: bool = False) -> tuple[dict, dict]:
    """Build source/target dicts for summary-based metrics."""
    d = src.shape[1]

    def _d(emb: np.ndarray) -> dict:
        n = len(emb)
        dct: dict = {
            "count": pa.array([n], type=pa.int64()),
            "sum": pa.FixedSizeListArray.from_arrays(pa.array(emb.sum(axis=0).astype(np.float64)), d),
            "sum_sq": pa.FixedSizeListArray.from_arrays(pa.array((emb * emb).sum(axis=0).astype(np.float64)), d),
        }
        if include_outer:
            dct["sum_outer"] = pa.FixedSizeListArray.from_arrays(
                pa.array((emb.T @ emb).reshape(-1).astype(np.float64)), d * d
            )
        return dct

    return _d(src), _d(tgt)


def _wasserstein_dicts(
    src: np.ndarray, tgt: np.ndarray, dims: int, bins: int, rng: tuple[float, float]
) -> tuple[dict, dict]:
    """Build source/target dicts for Wasserstein-1D."""
    low, high = rng
    use_dims = min(src.shape[1], dims)

    def _d(emb: np.ndarray) -> dict:
        hist_list = []
        for j in range(use_dims):
            h, _ = np.histogram(emb[:, j], bins=bins, range=(low, high))
            hist_list.append(h.astype(np.int64))
        hist_all = np.concatenate(hist_list)
        return {
            "hist_counts": pa.FixedSizeListArray.from_arrays(pa.array(hist_all), bins * use_dims),
        }

    return _d(src), _d(tgt)


def _emb_dicts(src: np.ndarray, tgt: np.ndarray) -> tuple[dict, dict]:
    """Build source/target dicts for full-embedding metrics."""
    d = src.shape[1]

    def _d(emb: np.ndarray) -> dict:
        return {
            "__emb__": pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1)), d),
        }

    return _d(src), _d(tgt)


def _cmd_dicts(src: np.ndarray, tgt: np.ndarray, k: int) -> tuple[dict, dict]:
    """Build source/target dicts for CMD (applies sigmoid)."""
    d = src.shape[1]
    src_sig = 1.0 / (1.0 + np.exp(-src))
    tgt_sig = 1.0 / (1.0 + np.exp(-tgt))

    def _d(emb_sig: np.ndarray, n_eff: int) -> dict:
        dct: dict = {"cmd_emb_n": pa.array([n_eff], type=pa.int64())}
        for j in range(1, k + 1):
            power_sum = (emb_sig**j).sum(axis=0).astype(np.float64)
            dct[f"cmd_emb_sum_{j}"] = pa.FixedSizeListArray.from_arrays(pa.array(power_sum), d)
        return dct

    return _d(src_sig, len(src)), _d(tgt_sig, len(tgt))


def _extract_value(metric: str, result: dict) -> float:
    """Extract the scalar value from a compute_delta result dict."""
    val = result.get(metric)
    if val is not None:
        return float(val.to_pylist()[0])
    note = result.get("note")
    if note is not None:
        pytest.fail(f"Metric '{metric}' failed: {note.to_pylist()}")
    raise ValueError(f"Metric '{metric}' not found in result keys: {list(result)}")


METRIC_CONFIGS = {
    "mmd_linear": {"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_linear"}},
    "klmvn_diag": {"columns": {"input": ["emb"]}, "distance": {"metric": "klmvn_diag"}},
    "fid": {"columns": {"input": ["emb"]}, "distance": {"metric": "fid", "epsilon": 1e-6}},
    "wasserstein_1d": {
        "columns": {"input": ["emb"]},
        "distance": {"metric": "wasserstein_1d"},
        "summary": {"histogram": {"dims": 32, "bins": 32, "range": [-3.0, 3.0]}},
    },
    "mmd_rbf": {"columns": {"input": ["emb"]}, "distance": {"metric": "mmd_rbf"}},
    "mmd_poly": {
        "columns": {"input": ["emb"]},
        "distance": {
            "metric": "mmd_poly",
            "kernel_params": {"degree": 3.0, "gamma": 1.0, "coefficient0": 1.0},
        },
    },
    "pad": {"columns": {"input": ["emb"]}, "distance": {"metric": "pad"}},
    "cmd": {"columns": {"input": ["emb"]}, "distance": {"metric": "cmd"}},
}


@pytest.mark.parametrize("metric", list(METRIC_CONFIGS.keys()))
def test_domain_gap_monotonic_synthetic(metric: str) -> None:
    """Verify monotonic ordering for a domain gap metric with controlled shift."""
    rng = np.random.default_rng(42)
    n, d = N_SAMPLES, EMBED_DIM

    proc = DomainGapProcessor(name="test", config=METRIC_CONFIGS[metric])

    values: list[float] = []
    for shift in SHIFTS:
        if shift == pytest.approx(0.0):
            src = rng.normal(0, 1, (n, d)).astype(np.float64)
            tgt = src.copy()
        else:
            src = rng.normal(0, 1, (n, d)).astype(np.float64)
            tgt = rng.normal(shift, 1, (n, d)).astype(np.float64)

        if metric in {"mmd_linear", "klmvn_diag", "fid"}:
            source, target = _summary_dicts(src, tgt, include_outer=(metric == "fid"))
        elif metric == "wasserstein_1d":
            source, target = _wasserstein_dicts(src, tgt, proc.hist_dims, proc.hist_bins, proc.hist_range)
        elif metric in {"mmd_rbf", "mmd_poly", "pad"}:
            source, target = _emb_dicts(src, tgt)
        elif metric == "cmd":
            source, target = _cmd_dicts(src, tgt, proc.cmd_k)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        result = proc.compute_delta(source, target)
        val = _extract_value(metric, result)
        values.append(val)

    print(
        f"  {metric:15s}: identical={values[0]:.8g}  "
        f"small={values[1]:.8g}  medium={values[2]:.8g}  strong={values[3]:.8g}"
    )

    assert all(v >= 0 for v in values), f"All values must be non-negative for {metric}: {values}"

    atol = 1.0 if metric == "pad" else 1e-10
    assert values[0] == pytest.approx(0.0, abs=atol), f"Identical datasets should give ≈0 for {metric}, got {values[0]}"

    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1], (
            f"Monotonicity violated for {metric}: "
            f"shift={SHIFTS[i]} ({values[i]:.8g}) > "
            f"shift={SHIFTS[i + 1]} ({values[i + 1]:.8g})"
        )
