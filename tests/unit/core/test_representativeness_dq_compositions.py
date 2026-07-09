"""DQ use-case tests for representativeness metrics.

Tests that representativeness metrics behave correctly for realistic data
quality scenarios. Each use case uses contamination-based level design:
a "clean" base distribution is progressively mixed with increasing proportions
of "bad" data, guaranteeing monotonic ordering of statistical distance.

See docs/metrics/representativeness.md for the 5 documented use cases.

Design note: at n=100k, chi-square is overpowered — even perfectly healthy
data can produce p < 0.05 (type I error). Tests assert monotonic ordering
of statistics (chi-sq, KS D, GRTE) rather than pass/fail interpretations.
"""

from dqm_ml_core.metrics.representativeness import RepresentativenessProcessor
import numpy as np
import pyarrow as pa

N_SAMPLES = 100_000
RNG_SEED = 42


def _run_repr(
    data: np.ndarray,
    distribution: str = "normal",
    user_params: dict | None = None,
) -> dict:
    """Run the full representativeness pipeline on synthetic data.

    Returns the flat result dict from compute().
    """
    config: dict = {
        "columns": {"input": ["data_1"]},
        "distribution": distribution,
        "metrics": ["chi-square", "kolmogorov-smirnov", "shannon-entropy", "grte"],
        "expected_counts_method": "cdf",
    }
    if user_params is not None:
        config["mean_std_estimation"] = "user_provided"
        config["distribution_params"] = [
            {"column": "data_1", **user_params},
        ]

    proc = RepresentativenessProcessor(name="test", config=config)
    proc.compute_seed = RNG_SEED
    proc._rng = np.random.default_rng(RNG_SEED)

    features = {"data_1": pa.array(data)}
    batch = proc.compute_batch_metric(features)
    return proc.compute(batch)


def _get(results: dict, metric: str, field: str) -> float:
    return float(results[f"data_1_{metric}_{field}"])


def _print_level(name: str, results: dict) -> None:
    cs = _get(results, "chi-square", "statistic")
    cp = _get(results, "chi-square", "p_value")
    ks = _get(results, "kolmogorov-smirnov", "statistic")
    kp = _get(results, "kolmogorov-smirnov", "p_value")
    gr = _get(results, "grte", "grte_value")
    sh = _get(results, "shannon-entropy", "entropy")
    ci = results["data_1_chi-square_interpretation"]
    ki = results["data_1_kolmogorov-smirnov_interpretation"]
    gi = results["data_1_grte_interpretation"]
    print(
        f"  {name:15s}  χ²={cs:10.4f}  p={cp:.6g}  "
        f"KS={ks:.4f}  p={kp:.6g}  "
        f"GRTE={gr:.4f}  H={sh:.4f}\n"
        f"                     "
        f"χ²: {ci}  |  KS: {ki}  |  GRTE: {gi}"
    )


def _contaminate(
    rng: np.random.Generator,
    clean_gen,
    bad_gen,
    rates: list[float],
) -> list[np.ndarray]:
    """Generate datasets with increasing contamination rates.

    Each dataset is: (1 - rate) * clean_samples + rate * bad_samples.
    """
    n = N_SAMPLES
    clean = clean_gen(rng, n)
    bad = bad_gen(rng, n)
    results = []
    for rate in rates:
        n_bad = int(n * rate)
        combined = clean.copy()
        combined[:n_bad] = bad[:n_bad]
        rng.shuffle(combined)
        results.append(combined)
    return results


# ── Use Case 1: Preprocessing — Standard Scaling ─────────────────────
# "I standard-scaled my features. Did they become N(0,1)?" (target: normal)
# Design: increasing proportion of mean-shifted contaminant


def test_preprocessing_standard_scaling() -> None:
    names = ["clean", "contam_5pct", "contam_10pct", "contam_20pct", "contam_50pct"]
    rates = [0.0, 0.05, 0.10, 0.20, 0.50]
    rng = np.random.default_rng(RNG_SEED)
    datasets = _contaminate(
        rng,
        lambda rng, n: rng.normal(0.0, 1.0, n),
        lambda rng, n: rng.normal(3.0, 1.0, n),
        rates,
    )
    params = {"mean": 0.0, "std": 1.0}
    chi2_stats, ks_stats, grtes = [], [], []

    for nm, data in zip(names, datasets, strict=True):
        res = _run_repr(data, distribution="normal", user_params=params)
        _print_level(nm, res)
        chi2_stats.append(_get(res, "chi-square", "statistic"))
        ks_stats.append(_get(res, "kolmogorov-smirnov", "statistic"))
        grtes.append(_get(res, "grte", "grte_value"))

    for i in range(len(chi2_stats) - 1):
        assert chi2_stats[i] <= chi2_stats[i + 1], (
            f"chi-square not monotonic: {names[i]}={chi2_stats[i]:.4f} > {names[i + 1]}={chi2_stats[i + 1]:.4f}"
        )
        assert grtes[i] >= grtes[i + 1], (
            f"GRTE not monotonic: {names[i]}={grtes[i]:.4f} < {names[i + 1]}={grtes[i + 1]:.4f}"
        )
    # KS: stochastic (500-sample subsampling) — only assert global trend
    assert ks_stats[0] < ks_stats[-1], (
        f"KS failed overall increase: {names[0]}={ks_stats[0]:.4f} >= {names[-1]}={ks_stats[-1]:.4f}"
    )


# ── Use Case 2: Preprocessing — Min-Max Scaling ──────────────────────
# "I min-max scaled to [0,1]. Are my features uniform?" (target: uniform)
# Design: increasing proportion of concentrated contaminant


def test_preprocessing_minmax_scaling() -> None:
    names = ["clean", "contam_5pct", "contam_10pct", "contam_20pct", "contam_50pct"]
    rates = [0.0, 0.05, 0.10, 0.20, 0.50]
    rng = np.random.default_rng(RNG_SEED)

    def _clean(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(0.0, 1.0, n)

    def _bad(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.clip(rng.normal(0.5, 0.05, n), 0.0, 1.0)

    datasets = _contaminate(rng, _clean, _bad, rates)
    params = {"min": 0.0, "max": 1.0}
    chi2_stats, ks_stats, grtes = [], [], []

    for nm, data in zip(names, datasets, strict=True):
        res = _run_repr(data, distribution="uniform", user_params=params)
        _print_level(nm, res)
        chi2_stats.append(_get(res, "chi-square", "statistic"))
        ks_stats.append(_get(res, "kolmogorov-smirnov", "statistic"))
        grtes.append(_get(res, "grte", "grte_value"))

    for i in range(len(chi2_stats) - 1):
        assert chi2_stats[i] <= chi2_stats[i + 1], (
            f"chi-square not monotonic: {names[i]}={chi2_stats[i]:.4f} > {names[i + 1]}={chi2_stats[i + 1]:.4f}"
        )
        assert grtes[i] >= grtes[i + 1], (
            f"GRTE not monotonic: {names[i]}={grtes[i]:.4f} < {names[i + 1]}={grtes[i + 1]:.4f}"
        )
    # KS: stochastic (500-sample subsampling) — only assert global trend
    assert ks_stats[0] < ks_stats[-1], (
        f"KS failed overall increase: {names[0]}={ks_stats[0]:.4f} >= {names[-1]}={ks_stats[-1]:.4f}"
    )


# ── Use Case 3: Synthetic Data Quality ─────────────────────────────
# "My GAN generated features. Do they match the real distribution?"
# (target: normal)
# Design: increasing contamination from mode-collapse (bimodal) generator


def test_synthetic_data_quality() -> None:
    names = ["good", "contam_5pct", "contam_10pct", "contam_20pct", "contam_50pct"]
    rates = [0.0, 0.05, 0.10, 0.20, 0.50]
    rng = np.random.default_rng(RNG_SEED)

    def _real(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, 1.0, n)

    def _mode_collapse(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.concatenate(
            [
                rng.normal(-2.0, 0.3, n // 2),
                rng.normal(2.0, 0.3, n - n // 2),
            ]
        )

    datasets = _contaminate(rng, _real, _mode_collapse, rates)
    params = {"mean": 0.0, "std": 1.0}
    chi2_stats, ks_stats, grtes = [], [], []

    for nm, data in zip(names, datasets, strict=True):
        res = _run_repr(data, distribution="normal", user_params=params)
        _print_level(nm, res)
        chi2_stats.append(_get(res, "chi-square", "statistic"))
        ks_stats.append(_get(res, "kolmogorov-smirnov", "statistic"))
        grtes.append(_get(res, "grte", "grte_value"))

    for i in range(len(chi2_stats) - 1):
        assert chi2_stats[i] <= chi2_stats[i + 1], (
            f"chi-square not monotonic: {names[i]}={chi2_stats[i]:.4f} > {names[i + 1]}={chi2_stats[i + 1]:.4f}"
        )
        assert grtes[i] >= grtes[i + 1], (
            f"GRTE not monotonic: {names[i]}={grtes[i]:.4f} < {names[i + 1]}={grtes[i + 1]:.4f}"
        )
    # KS: stochastic (500-sample subsampling) — only assert global trend
    assert ks_stats[0] < ks_stats[-1], (
        f"KS failed overall increase: {names[0]}={ks_stats[0]:.4f} >= {names[-1]}={ks_stats[-1]:.4f}"
    )


# ── Use Case 4: Acquisition Artifacts ──────────────────────────────
# "I have data from multiple sources. Is one corrupted?" (target: normal)
# Design: increasing contamination from saturated/outlier source


def test_acquisition_artifacts() -> None:
    names = ["clean", "contam_5pct", "contam_10pct", "contam_20pct", "contam_50pct"]
    rates = [0.0, 0.05, 0.10, 0.20, 0.50]
    rng = np.random.default_rng(RNG_SEED)

    def _clean(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, 1.0, n)

    def _corrupted(rng: np.random.Generator, n: int) -> np.ndarray:
        """Simulate a corrupted source with outliers and clipping."""
        x = rng.normal(0.5, 1.5, n)
        x = np.clip(x, -3.0, 3.0)
        n_out = int(n * 0.02)
        idx = rng.choice(n, n_out, replace=False)
        x[idx] = rng.choice([-1, 1], n_out) * 10.0
        return x

    datasets = _contaminate(rng, _clean, _corrupted, rates)
    params = {"mean": 0.0, "std": 1.0}
    chi2_stats, ks_stats, grtes = [], [], []

    for nm, data in zip(names, datasets, strict=True):
        res = _run_repr(data, distribution="normal", user_params=params)
        _print_level(nm, res)
        chi2_stats.append(_get(res, "chi-square", "statistic"))
        ks_stats.append(_get(res, "kolmogorov-smirnov", "statistic"))
        grtes.append(_get(res, "grte", "grte_value"))

    for i in range(len(chi2_stats) - 1):
        assert chi2_stats[i] <= chi2_stats[i + 1], (
            f"chi-square not monotonic: {names[i]}={chi2_stats[i]:.4f} > {names[i + 1]}={chi2_stats[i + 1]:.4f}"
        )
        assert grtes[i] >= grtes[i + 1], (
            f"GRTE not monotonic: {names[i]}={grtes[i]:.4f} < {names[i + 1]}={grtes[i + 1]:.4f}"
        )
    # KS: stochastic (500-sample subsampling) — only assert global trend
    assert ks_stats[0] < ks_stats[-1], (
        f"KS failed overall increase: {names[0]}={ks_stats[0]:.4f} >= {names[-1]}={ks_stats[-1]:.4f}"
    )


# ── Use Case 5: Feature Engineering Validation ─────────────────────
# "I applied a log-transform. Did it normalize my feature?" (target: normal)
# Design: increasing contamination from right-skewed residual


def test_feature_engineering_validation() -> None:
    names = ["success", "contam_5pct", "contam_10pct", "contam_20pct", "contam_50pct"]
    rates = [0.0, 0.05, 0.10, 0.20, 0.50]
    rng = np.random.default_rng(RNG_SEED)

    def _normalized(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, 1.0, n)

    def _skewed(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.exponential(1.0, n) - 1.0

    datasets = _contaminate(rng, _normalized, _skewed, rates)
    params = {"mean": 0.0, "std": 1.0}
    chi2_stats, ks_stats, grtes = [], [], []

    for nm, data in zip(names, datasets, strict=True):
        res = _run_repr(data, distribution="normal", user_params=params)
        _print_level(nm, res)
        chi2_stats.append(_get(res, "chi-square", "statistic"))
        ks_stats.append(_get(res, "kolmogorov-smirnov", "statistic"))
        grtes.append(_get(res, "grte", "grte_value"))

    for i in range(len(chi2_stats) - 1):
        assert chi2_stats[i] <= chi2_stats[i + 1], (
            f"chi-square not monotonic: {names[i]}={chi2_stats[i]:.4f} > {names[i + 1]}={chi2_stats[i + 1]:.4f}"
        )
        assert grtes[i] >= grtes[i + 1], (
            f"GRTE not monotonic: {names[i]}={grtes[i]:.4f} < {names[i + 1]}={grtes[i + 1]:.4f}"
        )
    # KS: stochastic (500-sample subsampling) — only assert global trend
    assert ks_stats[0] < ks_stats[-1], (
        f"KS failed overall increase: {names[0]}={ks_stats[0]:.4f} >= {names[-1]}={ks_stats[-1]:.4f}"
    )


# ── Extra: Constant / Degenerate Features ──────────────────────────
# "Do I have dead features in my dataset?" (target: normal)
# Design: increasing degree of degeneracy (monotonic by construction)


def test_constant_degenerate_features() -> None:
    names = ["healthy", "near_constant", "mostly_zero", "dead_constant"]
    rng = np.random.default_rng(RNG_SEED)
    n = N_SAMPLES

    gens = {
        "healthy": lambda: rng.normal(0.0, 1.0, n),
        "near_constant": lambda: rng.normal(0.0, 1e-6, n),
        "mostly_zero": lambda: np.where(rng.random(n) < 0.9, 0.0, rng.normal(0.0, 1.0, n)),
        "dead_constant": lambda: np.full(n, 42.0),
    }
    params = {"mean": 0.0, "std": 1.0}
    chi2_stats, ks_stats, grtes = [], [], []

    for nm in names:
        data = gens[nm]()
        res = _run_repr(data, distribution="normal", user_params=params)
        _print_level(nm, res)
        chi2_stats.append(_get(res, "chi-square", "statistic"))
        ks_stats.append(_get(res, "kolmogorov-smirnov", "statistic"))
        grtes.append(_get(res, "grte", "grte_value"))

    for i in range(len(chi2_stats) - 1):
        assert chi2_stats[i] <= chi2_stats[i + 1], (
            f"chi-square not monotonic: {names[i]}={chi2_stats[i]:.4f} > {names[i + 1]}={chi2_stats[i + 1]:.4f}"
        )
        assert grtes[i] >= grtes[i + 1], (
            f"GRTE not monotonic: {names[i]}={grtes[i]:.4f} < {names[i + 1]}={grtes[i + 1]:.4f}"
        )
    # KS: stochastic (500-sample subsampling) — only assert global trend
    assert ks_stats[0] < ks_stats[-1], (
        f"KS failed overall increase: {names[0]}={ks_stats[0]:.4f} >= {names[-1]}={ks_stats[-1]:.4f}"
    )
