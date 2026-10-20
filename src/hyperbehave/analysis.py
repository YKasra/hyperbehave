"""Population-level analysis of the feature table (Layer A).

Functions here stay dependency-light (numpy and pandas). Clustering uses scikit-learn
and is imported lazily (``pip install hyperbehave[analysis]``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hyperbehave.markers import CORE_MARKERS, MARKERS


def included(features: pd.DataFrame) -> pd.DataFrame:
    """Wallets that passed the exclusion rules."""
    if "exclude" not in features:
        return features
    return features.loc[~features["exclude"].astype(bool)].copy()


def exclusion_log(features: pd.DataFrame) -> pd.DataFrame:
    """Count of excluded wallets per reason (for the paper's flow diagram)."""
    ex = features.loc[features["exclude"].astype(bool), "reason"]
    return ex.value_counts().rename_axis("reason").reset_index(name="n_wallets")


def describe(features: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """n, median, IQR and 90th percentile of each marker."""
    cols = cols or [m.name for m in MARKERS if m.name in features]
    rows = []
    for c in cols:
        x = pd.to_numeric(features[c], errors="coerce").dropna()
        if x.empty:
            rows.append({"marker": c, "n": 0})
            continue
        rows.append({"marker": c, "n": int(x.size), "median": x.median(), "q25": x.quantile(0.25),
                     "q75": x.quantile(0.75), "p90": x.quantile(0.90)})
    return pd.DataFrame(rows)


def high_risk_flags(features: pd.DataFrame, cols: list[str] | None = None,
                    quantile: float = 0.90, min_markers: int = 3) -> pd.DataFrame:
    """Flag wallets in the top decile of the population on several markers at once.

    Returns the input with ``n_high_markers`` and ``multi_marker_high_risk`` added. This
    is a descriptive screen, not a diagnosis.
    """
    cols = [c for c in (cols or CORE_MARKERS) if c in features]
    out = features.copy()
    high = pd.DataFrame(index=out.index)
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        thr = x.quantile(quantile)
        high[c] = x > thr
    out["n_high_markers"] = high.sum(axis=1)
    out["multi_marker_high_risk"] = out["n_high_markers"] >= min_markers
    return out


def co_occurrence(features: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Spearman correlation matrix of the markers (do gambling-like signs cluster?)."""
    cols = [c for c in (cols or CORE_MARKERS) if c in features]
    return features[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U / n1 n0); ties count half."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def marker_auc_table(features: pd.DataFrame, group_col: str, positive: str,
                     cols: list[str] | None = None) -> pd.DataFrame:
    """How well each marker separates ``positive`` from the rest (AUC, coverage)."""
    cols = cols or [m.name for m in MARKERS if m.name in features]
    y = (features[group_col] == positive).to_numpy()
    rows = []
    for c in cols:
        x = pd.to_numeric(features[c], errors="coerce").to_numpy()
        rows.append({"marker": c, "auc": auc(x, y), "coverage": float(np.isfinite(x).mean())})
    return pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last")


def behavioural_profiles(features: pd.DataFrame, cols: list[str] | None = None,
                         k_range: range = range(2, 7), seed: int = 0) -> tuple[pd.Series, int]:
    """Gaussian-mixture profiles on rank-transformed markers; k chosen by BIC.

    Missing values are filled with the median after a percentile-rank transform, so
    outliers and heavy tails do not drive the clusters. Returns (labels, k).
    """
    from sklearn.mixture import GaussianMixture

    cols = [c for c in (cols or CORE_MARKERS) if c in features]
    x = features[cols].apply(pd.to_numeric, errors="coerce").rank(pct=True)
    x = x.fillna(x.median()).fillna(0.5).to_numpy()
    best, best_bic, best_k = None, np.inf, None
    for k in k_range:
        if k >= len(x):
            break
        gm = GaussianMixture(k, covariance_type="diag", random_state=seed, n_init=3).fit(x)
        bic = gm.bic(x)
        if bic < best_bic:
            best, best_bic, best_k = gm, bic, k
    if best is None:
        return pd.Series(0, index=features.index, name="profile"), 1
    return pd.Series(best.predict(x), index=features.index, name="profile"), best_k
