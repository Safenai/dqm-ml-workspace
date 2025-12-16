from __future__ import annotations

import logging
from typing import Any, override

import numpy as np
import pyarrow as pa

from dqm_ml_core import DatametricProcessor

logger = logging.getLogger(__name__)


class DomainGapProcessor(DatametricProcessor):
    """
    Reduce per-row embeddings into dataset-level summaries and compute deltas.

    Config:
      INPUT:
        embedding_col: "embedding"
      calcul au niveau du batch:
        collect_sum_outer: bool   # needed for FID
        collect_hist_1d: bool     # needed for Wasserstein-1D
        hist_dims: int            # number of dims to histogram (<= d)
        hist_bins: int
        hist_range: [low, high]
      DELTA: calcul au niveau du dataset
        metric: "klmvn_diag" | "mmd_linear" | "fid" | "wasserstein_1d"
    """

    def __init__(self, name: str = "image_embedding", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._checked = False

    # ---------------- API ----------------
    def check_config(self) -> None:
        cfg = self.config or {}
        icfg = cfg.get("INPUT", {})
        self.embedding_col: str = icfg.get("embedding_col", "embedding")

        dcfg = cfg.get("DELTA", {})
        self.delta_metric: str = str(dcfg.get("metric", "klmvn_diag")).lower()
        scfg = cfg.get("SUMMARY", {})

        if self.delta_metric == "fid":
            auto_sum_outer = True
            auto_hist_1d = False
        elif self.delta_metric == "wasserstein_1d":
            auto_sum_outer = False
            auto_hist_1d = True
        else:  # klmvn_diag, mmd_linear
            auto_sum_outer = False
            auto_hist_1d = False

        self.collect_sum_outer: bool = bool(scfg.get("collect_sum_outer", auto_sum_outer))
        self.collect_hist_1d: bool = bool(scfg.get("collect_hist_1d", auto_hist_1d))

        # Paramètres Wasserstein-1D
        self.hist_dims: int = int(scfg.get("hist_dims", 64))
        self.hist_bins: int = int(scfg.get("hist_bins", 32))
        rng = scfg.get("hist_range", [-3.0, 3.0])
        self.hist_range: tuple[float, float] = (float(rng[0]), float(rng[1]))

        self._checked = True

    @override
    def needed_columns(self) -> list[str]:
        if not getattr(self, "_checked", False):
            self.check_config()
        return [self.embedding_col]

    def generated_columns(self) -> list[str]:
        return []

    @override
    def compute_batch_metric(self, features: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Produce per-batch summaries from the 'embedding' column.
        Ici on peut aussi faire des histogrammes sur les embeddings, mais aussi la moyenne et la variance des embeddings
        """
        if not getattr(self, "_checked", False):
            self.check_config()

        emb = features.get(self.embedding_col)
        if emb is None or not isinstance(emb, pa.FixedSizeListArray):
            return {}

        n = len(emb)
        # d = emb.list_size
        d = len(emb[0])
        child = emb.values  # flat
        arr = np.asarray(child.to_numpy()).reshape(n, d)

        out: dict[str, pa.Array] = {}
        out["count"] = pa.array([n], type=pa.int64())
        out["sum"] = pa.FixedSizeListArray.from_arrays(pa.array(arr.sum(axis=0).astype(np.float64)), d)
        out["sum_sq"] = pa.FixedSizeListArray.from_arrays(pa.array((arr * arr).sum(axis=0).astype(np.float64)), d)

        # pptional: sum_outer for FID
        if self.collect_sum_outer:
            s = (arr.T @ arr).reshape(-1).astype(np.float64)
            out["sum_outer"] = pa.FixedSizeListArray.from_arrays(pa.array(s), d * d)

        # ptional: histograms for Wasserstein-1D
        if self.collect_hist_1d:
            use_dims = min(d, self.hist_dims)
            low, high = self.hist_range
            hist_list: list[np.ndarray] = []
            for j in range(use_dims):
                h, _ = np.histogram(arr[:, j], bins=self.hist_bins, range=(low, high))
                hist_list.append(h.astype(np.int64))
            h = np.stack(hist_list, axis=0).reshape(-1)
            out["hist_counts"] = pa.FixedSizeListArray.from_arrays(pa.array(h), self.hist_bins * use_dims)

        return out

    @override
    def compute(self, batch_metrics: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Aggregate per-batch summaries into a single dataset-level summary row."""
        if not batch_metrics:
            return {}

        def _sum_scalar(a: pa.Array) -> int:
            return int(np.asarray(a.to_numpy()).sum())

        def _sum_fixed(v: pa.FixedSizeListArray) -> tuple[np.ndarray, int]:
            vals = np.asarray(v.values.to_numpy(), dtype=np.float64)
            d = len(v[0])
            # d = v.list_size
            return vals.reshape(-1, d).sum(axis=0), d

        out: dict[str, pa.Array] = {}

        # count
        if "count" not in batch_metrics:
            return {}
        total_n = _sum_scalar(batch_metrics["count"])
        out["count"] = pa.array([total_n], type=pa.int64())

        # sum / sum_sq
        if "sum" in batch_metrics:
            s, d = _sum_fixed(batch_metrics["sum"])
            out["sum"] = pa.FixedSizeListArray.from_arrays(pa.array(s), d)
        if "sum_sq" in batch_metrics:
            s2, d2 = _sum_fixed(batch_metrics["sum_sq"])
            out["sum_sq"] = pa.FixedSizeListArray.from_arrays(pa.array(s2), d2)

        # optional sum_outer
        if "sum_outer" in batch_metrics:
            so_vals = np.asarray(batch_metrics["sum_outer"].values.to_numpy(), dtype=np.float64)
            dd = len(batch_metrics["sum_outer"][0])
            # dd = batch_metrics["sum_outer"].list_size
            out["sum_outer"] = pa.FixedSizeListArray.from_arrays(pa.array(so_vals.reshape(-1, dd).sum(axis=0)), dd)

        # optional hist_counts
        if "hist_counts" in batch_metrics:
            h_vals = np.asarray(batch_metrics["hist_counts"].values.to_numpy(), dtype=np.int64)
            h_len = len(batch_metrics["hist_counts"][0])
            # h_len = batch_metrics["hist_counts"].list_size
            out["hist_counts"] = pa.FixedSizeListArray.from_arrays(
                pa.array(h_vals.reshape(-1, h_len).sum(axis=0)), h_len
            )

        return out

    @override
    def compute_delta(self, source: dict[str, pa.Array], target: dict[str, pa.Array]) -> dict[str, pa.Array]:
        """Compute a domain-gap metric from two dataset-level summaries."""
        if not getattr(self, "_checked", False):
            self.check_config()

        def vec(a: pa.FixedSizeListArray) -> Any:  # TODO : check type error np.ndarray
            len_a = len(a[0])
            # len_a = a.list_size
            array = np.asarray(a.values.to_numpy(), dtype=np.float64).reshape(-1, len_a).sum(axis=0)
            return array  # type : ignore[no-any-return]

        def scalar(a: pa.Array) -> float:
            return float(np.asarray(a.to_numpy()).sum())

        metric = self.delta_metric

        if metric in {"klmvn_diag", "mmd_linear", "fid"}:
            need = {"count", "sum"}
            if metric in {"klmvn_diag", "fid"}:
                need |= {"sum_sq"}
            if metric == "fid":
                need |= {"sum_outer"}
            for side, name in ((source, "source"), (target, "target")):
                if not need.issubset(side.keys()):
                    return {"metric": pa.array([metric]), "note": pa.array([f"missing keys in {name}: {sorted(need)}"])}

            n1, n2 = scalar(source["count"]), scalar(target["count"])
            if n1 <= 0 or n2 <= 0:
                return {"metric": pa.array([metric]), "note": pa.array(["empty summaries"])}

            mu1 = vec(source["sum"]) / n1
            mu2 = vec(target["sum"]) / n2

            if metric == "mmd_linear":
                diff = mu1 - mu2
                val = float(np.dot(diff, diff))
                return {"metric": pa.array([metric]), "value": pa.array([val], type=pa.float64())}

            v1 = np.maximum(vec(source["sum_sq"]) / n1 - mu1 * mu1, 1e-9)
            v2 = np.maximum(vec(target["sum_sq"]) / n2 - mu2 * mu2, 1e-9)

            if metric == "klmvn_diag":
                term_var = np.sum(v1 / v2 - 1.0 - np.log(v1 / v2))
                term_mean = np.sum((mu2 - mu1) ** 2 / v2)
                val = 0.5 * (term_var + term_mean)
                return {"metric": pa.array([metric]), "value": pa.array([float(val)], type=pa.float64())}

            if metric == "fid":
                so1 = vec(source["sum_outer"])
                so2 = vec(target["sum_outer"])
                d = int(np.sqrt(so1.size))
                s1 = (so1.reshape(d, d) / n1) - np.outer(mu1, mu1)
                s2 = (so2.reshape(d, d) / n2) - np.outer(mu2, mu2)
                from scipy.linalg import sqrtm

                diff = mu1 - mu2
                # The `disp` argument is deprecated and will be
                # removed in SciPy 1.18.0. The previously returned error estimate
                # can be computed as ``norm(X @ X - A, 'fro')**2 / norm(A, 'fro')``
                # covmean, _ = sqrtm(s1.dot(s2), disp=False)
                covmean = sqrtm(s1.dot(s2))

                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                fid = diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean)
                return {"metric": pa.array([metric]), "value": pa.array([float(abs(fid))], type=pa.float64())}

        if metric == "wasserstein_1d":
            if "hist_counts" not in source or "hist_counts" not in target:
                return {"metric": pa.array([metric]), "note": pa.array(["missing hist_counts"])}
            h1 = np.asarray(source["hist_counts"].values.to_numpy(), dtype=np.int64)
            h2 = np.asarray(target["hist_counts"].values.to_numpy(), dtype=np.int64)
            # derive dims from summary config
            use_dims = self.hist_dims
            bins = self.hist_bins
            if h1.size != h2.size or h1.size != bins * use_dims:
                return {"metric": pa.array([metric]), "note": pa.array(["hist_counts length mismatch"])}
            width = (self.hist_range[1] - self.hist_range[0]) / bins
            total = 0.0
            used = 0
            for j in range(use_dims):
                h1 = h1[j * bins : (j + 1) * bins].astype(np.float64)
                h2 = h2[j * bins : (j + 1) * bins].astype(np.float64)
                if h1.sum() == 0 and h2.sum() == 0:
                    continue
                p = h1 / max(1.0, h1.sum())
                q = h2 / max(1.0, h2.sum())
                cdf_p = np.cumsum(p)
                cdf_q = np.cumsum(q)
                total += float(np.sum(np.abs(cdf_p - cdf_q)) * width)
                used += 1
            val = total / max(1, used)
            return {"metric": pa.array([metric]), "value": pa.array([val], type=pa.float64())}

        return {"metric": pa.array([metric]), "note": pa.array(["unsupported metric or invalid inputs"])}
