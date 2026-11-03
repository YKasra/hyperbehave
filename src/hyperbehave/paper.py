"""Paper tables and figures from a feature table (``hyperbehave paper``).

Every number in the working paper comes from this module, so the analysis can be
re-run by anyone from ``features.parquet``. Sections:

1. Sample: how many wallets were analysed, excluded (and why), included.
2. Prevalence: share of wallets showing each behaviour, by fixed, pre-stated
   thresholds, with Wilson 95% confidence intervals.
3. Reliability: split-half (even vs odd calendar days) Spearman correlation of each
   marker, with the Spearman-Brown correction.
4. Co-occurrence: do gambling-like behaviours go together? Spearman and partial
   correlations (activity, capital and active days removed), and a permutation test
   of how many wallets are extreme on several markers at once versus chance.
5. Harm: association of each marker with drawdown, loss and liquidation, with
   Benjamini-Hochberg correction; a composite score in a regression with controls.

Output: ``results.md`` plus CSV tables and PNG figures in the output folder. The
output contains aggregates only, never wallet identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hyperbehave import analysis as an
from hyperbehave.report import GRID, INK, INK_2, SERIES, SURFACE, md_table

# Behavioural markers used for co-occurrence and the composite score. Harm markers are
# outcomes, not part of the score.
BEHAVIOUR = [
    "loss_chasing_index", "post_loss_size_ratio", "size_trend_weekly", "rest_window_share",
    "sessions_per_active_day", "doubling_down_share", "median_eff_leverage", "lottery_share",
]
COVARIATES = ["orders_per_day", "capital_at_risk", "active_days"]
RELIABILITY = [
    "loss_chasing_index", "post_loss_size_ratio", "doubling_down_share", "disposition_ratio",
    "rest_window_share", "hour_week_entropy", "sessions_per_active_day", "median_eff_leverage",
    "lottery_share",
]
LABELS = {
    "loss_chasing_index": "Loss chasing (re-entry)",
    "post_loss_size_ratio": "Bigger size after losses",
    "size_trend_weekly": "Size escalation over weeks",
    "rest_window_share": "Trading in own rest window",
    "sessions_per_active_day": "Sessions per day",
    "doubling_down_share": "Doubling down on losers",
    "median_eff_leverage": "Effective leverage",
    "lottery_share": "Lottery-like coins",
    "disposition_ratio": "Holding losers longer",
    "hour_week_entropy": "Spread over the week",
    "redeposit_after_liq_share": "Re-deposit after liquidation",
}


@dataclass(frozen=True)
class Indicator:
    label: str
    column: str
    test: str  # pandas query on the column, e.g. ">= 0.5"
    base: str | None = None  # optional pandas query defining the denominator


# Fixed thresholds stated before looking at the data they are applied to.
INDICATORS = [
    Indicator("Liquidated at least once", "n_liquidations", ">= 1"),
    Indicator("Re-deposited within 24 h of a liquidation (liquidated wallets)",
              "redeposit_after_liq_share", "> 0", "n_liquidations >= 1"),
    Indicator("Net trading loss over the window", "net_trading_pnl", "< 0"),
    Indicator("Lost at least half of capital at risk", "loss_to_capital", ">= 0.5"),
    Indicator("Fell at least 50% from peak equity", "max_drawdown_rel", ">= 0.5"),
    Indicator("Loss chasing beyond chance (p < 0.05)", "loss_chasing_p", "< 0.05"),
    Indicator("Bigger positions after losses beyond chance (p < 0.05)", "post_loss_size_p", "< 0.05"),
    Indicator("Added to losing positions in at least half of trips", "doubling_down_share", ">= 0.5"),
    Indicator("Median effective leverage of 10x or more", "median_eff_leverage", ">= 10"),
    Indicator("Half or more of volume in lottery-like coins", "lottery_share", ">= 0.5"),
    Indicator("10% or more of new positions in own rest window", "rest_window_share", ">= 0.10"),
    Indicator("Holds losers longer than winners", "disposition_ratio", "> 1"),
]
CHANCE_INDICATORS = {"loss_chasing_p", "post_loss_size_p"}


# --------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(q)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    from scipy import stats

    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 10 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return float("nan"), float("nan"), n
    r = stats.spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue), n


def ols_hc3(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and HC3 robust standard errors."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (resid / (1 - np.clip(h, 0, 0.999))) ** 2
    cov = XtX_inv @ (X.T * w) @ X @ XtX_inv
    return beta, np.sqrt(np.diag(cov))


def _covariate_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [np.log1p(pd.to_numeric(df[c], errors="coerce").clip(lower=0).fillna(0)) for c in COVARIATES if c in df]
    return np.column_stack(cols) if cols else np.empty((len(df), 0))


def _rank_residual(y: pd.Series, X: np.ndarray) -> pd.Series:
    r = y.rank()
    m = r.notna().to_numpy()
    out = pd.Series(np.nan, index=y.index)
    if m.sum() < 10:
        return out
    Xm = np.column_stack([np.ones(m.sum()), X[m]])
    beta = np.linalg.lstsq(Xm, r[m].to_numpy(), rcond=None)[0]
    out[m] = r[m].to_numpy() - Xm @ beta
    return out


# --------------------------------------------------------------------------- sections
def prevalence(inc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ind in INDICATORS:
        if ind.column not in inc:
            continue
        base = inc.query(ind.base) if ind.base else inc
        x = pd.to_numeric(base[ind.column], errors="coerce").dropna()
        n = len(x)
        k = int(pd.DataFrame({"v": x}).query(f"v {ind.test}").shape[0]) if n else 0
        lo, hi = wilson(k, n)
        row = {"behaviour": ind.label, "wallets": k, "of": n, "share": k / n if n else np.nan,
               "ci_low": lo, "ci_high": hi, "column": ind.column}
        if ind.column in CHANCE_INDICATORS and n:
            from scipy import stats

            row["vs_chance_p"] = float(stats.binomtest(k, n, 0.05, alternative="greater").pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def reliability(inc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in RELIABILITY:
        a, b = f"{m}_h0", f"{m}_h1"
        if a not in inc or b not in inc:
            continue
        r, p, n = spearman(inc[a], inc[b])
        sb = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan
        rows.append({"marker": m, "label": LABELS.get(m, m), "n": n, "split_half_r": r,
                     "spearman_brown": sb})
    return pd.DataFrame(rows)


def cooccurrence(inc: pd.DataFrame, n_perm: int = 2000, seed: int = 0) -> dict:
    cols = [c for c in BEHAVIOUR if c in inc]
    raw = inc[cols].apply(pd.to_numeric, errors="coerce")
    corr = raw.corr(method="spearman")
    X = _covariate_matrix(inc)
    partial = pd.DataFrame({c: _rank_residual(raw[c], X) for c in cols}).corr()
    iu = np.triu_indices(len(cols), 1)
    ranks = raw.rank(pct=True)
    top = ranks > 0.9
    observed = float((top.sum(axis=1) >= 3).mean())
    rng = np.random.default_rng(seed)
    vals = ranks.to_numpy()
    sims = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = np.column_stack([rng.permutation(vals[:, j]) for j in range(vals.shape[1])])
        sims[i] = np.mean(np.sum(shuffled > 0.9, axis=1) >= 3)
    return {
        "columns": cols, "spearman": corr, "partial": partial,
        "mean_r": float(np.nanmean(corr.to_numpy()[iu])),
        "mean_partial_r": float(np.nanmean(partial.to_numpy()[iu])),
        "observed": observed, "null": sims,
        "p": float((1 + np.sum(sims >= observed)) / (n_perm + 1)),
    }


def harm(inc: pd.DataFrame) -> dict:
    d = inc.copy()
    d["loss_capped"] = pd.to_numeric(d.get("loss_to_capital"), errors="coerce").clip(upper=1)
    d["any_liquidation"] = (pd.to_numeric(d["n_liquidations"], errors="coerce") > 0).astype(float)
    outcomes = {"max_drawdown_rel": "Drawdown from peak", "loss_capped": "Loss / capital",
                "any_liquidation": "Liquidated"}
    rows = []
    for m in BEHAVIOUR + ["redeposit_after_liq_share", "disposition_ratio"]:
        if m not in d:
            continue
        row = {"marker": m, "label": LABELS.get(m, m)}
        for out in outcomes:
            if m == "redeposit_after_liq_share" and out == "any_liquidation":
                row[f"{out}_rho"], row[f"{out}_p"], row[f"{out}_n"] = np.nan, np.nan, 0
                continue
            r, p, n = spearman(pd.to_numeric(d[m], errors="coerce"), d[out])
            row[f"{out}_rho"], row[f"{out}_p"], row[f"{out}_n"] = r, p, n
        rows.append(row)
    table = pd.DataFrame(rows)
    for out in outcomes:
        table[f"{out}_q"] = bh(table[f"{out}_p"].to_numpy())

    models = []
    for with_leverage in (True, False):
        cols = [c for c in BEHAVIOUR if c in d and (with_leverage or c != "median_eff_leverage")]
        score = d[cols].apply(pd.to_numeric, errors="coerce").rank(pct=True).mean(axis=1)
        X = _covariate_matrix(d)
        for out in ("max_drawdown_rel", "loss_capped", "any_liquidation"):
            y = pd.to_numeric(d[out], errors="coerce")
            m = y.notna() & score.notna()
            Xm = np.column_stack([np.ones(m.sum()), score[m].to_numpy(), X[m.to_numpy()]])
            beta, se = ols_hc3(y[m].to_numpy(), Xm)
            models.append({"score": "with leverage" if with_leverage else "without leverage",
                           "outcome": outcomes[out], "n": int(m.sum()), "coef": beta[1], "se": se[1],
                           "z": beta[1] / se[1] if se[1] > 0 else np.nan})
    from scipy import stats

    mod = pd.DataFrame(models)
    mod["p"] = 2 * stats.norm.sf(np.abs(mod["z"]))
    return {"table": table, "models": mod, "outcomes": outcomes}


# --------------------------------------------------------------------------- figures
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    return plt


def _clean(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_prevalence(prev: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    p = prev[prev["of"] > 0].sort_values("share").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(p) + 1.3), facecolor=SURFACE)
    _clean(ax)
    y = np.arange(len(p))
    chance = p["column"].isin(CHANCE_INDICATORS).to_numpy()
    colors = [SERIES[1] if c else SERIES[0] for c in chance]
    ax.barh(y, p["share"], color=colors, height=0.62)
    err = np.clip(np.vstack([(p["share"] - p["ci_low"]).to_numpy(float),
                             (p["ci_high"] - p["share"]).to_numpy(float)]), 0, None)
    ax.errorbar(p["share"].to_numpy(float), y, xerr=err,
                fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    for yi, (_, r) in zip(y, p.iterrows(), strict=True):
        ax.text(r["ci_high"] + 0.01, yi, f"{r['share']:.0%}  ({r['wallets']}/{r['of']})", va="center",
                fontsize=8.5, color=INK_2)
    if chance.any():
        ax.axvline(0.05, color=SERIES[1], linestyle="--", linewidth=1)
        ax.text(0.05, len(p) - 0.4, " 5% expected by chance", color=INK_2, fontsize=8, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(p["behaviour"])
    ax.set_xlim(0, min(1.0, float(p["ci_high"].max()) + 0.22))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle("How common is each behaviour? Share of wallets with 95% CI", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_reliability(rel: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    r = rel.dropna(subset=["spearman_brown"]).sort_values("spearman_brown")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(r) + 1.3), facecolor=SURFACE)
    _clean(ax)
    y = np.arange(len(r))
    ax.barh(y, r["spearman_brown"], color=SERIES[0], height=0.62)
    for yi, (_, row) in zip(y, r.iterrows(), strict=True):
        ax.text(max(row["spearman_brown"], 0) + 0.01, yi, f"{row['spearman_brown']:.2f} (n={row['n']})",
                va="center", fontsize=8.5, color=INK_2)
    ax.axvline(0.7, color=INK_2, linestyle="--", linewidth=1)
    ax.text(0.7, len(r) - 0.4, " 0.70 conventional threshold", color=INK_2, fontsize=8, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(r["label"])
    ax.set_xlim(min(0, float(r["spearman_brown"].min()) - 0.05), 1.15)
    fig.suptitle("Split-half reliability (even vs odd days, Spearman-Brown)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_partial(co: dict, path: Path) -> None:
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    cols = co["columns"]
    P = co["partial"].to_numpy().astype(float).copy()
    np.fill_diagonal(P, np.nan)
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f4f3ef", "#eb6834"])
    cmap.set_bad(GRID)
    fig, ax = plt.subplots(figsize=(7.6, 6.4), facecolor=SURFACE)
    im = ax.imshow(P, cmap=cmap, vmin=-0.5, vmax=0.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and np.isfinite(P[i, j]):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=INK)
    labels = [LABELS.get(c, c) for c in cols]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("partial Spearman correlation")
    ax.set_title("Do gambling-like behaviours go together?\n(activity, capital and active days held constant)",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_null(co: dict, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
    _clean(ax, "y")
    step = float(np.min(np.diff(np.unique(co["null"])))) if len(np.unique(co["null"])) > 1 else 0.01
    bins = np.arange(co["null"].min() - step / 2, co["null"].max() + step, step)
    ax.hist(co["null"], bins=bins, color=SERIES[0], alpha=0.85, edgecolor=SURFACE)
    ax.axvline(co["observed"], color=SERIES[1], linewidth=2)
    ax.annotate(f"observed {co['observed']:.1%}\nexpected {np.mean(co['null']):.1%}",
                xy=(co["observed"], ax.get_ylim()[1] * 0.98), xytext=(8, -4), textcoords="offset points",
                va="top", fontsize=9, color=INK,
                bbox={"boxstyle": "round,pad=0.25", "fc": SURFACE, "ec": GRID})
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of wallets in the top 10% on 3 or more of 8 markers")
    ax.set_ylabel("permutations")
    ax.set_title(f"Extreme on several markers at once: observed vs chance (p = {co['p']:.2f})",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_leverage(inc: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    d = inc[["median_eff_leverage", "max_drawdown_rel", "n_liquidations"]].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[d["median_eff_leverage"] > 0]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _clean(ax, "both")
    liq = d["n_liquidations"] > 0
    ax.scatter(d.loc[~liq, "median_eff_leverage"], d.loc[~liq, "max_drawdown_rel"], s=18, color=SERIES[0],
               alpha=0.7, edgecolors=SURFACE, linewidths=0.5, label="never liquidated")
    ax.scatter(d.loc[liq, "median_eff_leverage"], d.loc[liq, "max_drawdown_rel"], s=18, color=SERIES[1],
               alpha=0.8, edgecolors=SURFACE, linewidths=0.5, label="liquidated at least once")
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("median effective leverage (log scale)")
    ax.set_ylabel("drawdown from peak equity")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8.5, loc="upper left",
              markerscale=1.4)
    ax.set_title("Leverage is the behaviour most tied to losses", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- report
def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def paper_report(features: pd.DataFrame, out_dir: str | Path, n_perm: int = 2000) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inc = an.included(features)

    # 1 sample
    excl = an.exclusion_log(features) if features["exclude"].astype(bool).any() else pd.DataFrame()
    models = inc["balance_model"].value_counts() if "balance_model" in inc else pd.Series(dtype=int)

    # 2-5
    prev = prevalence(inc)
    rel = reliability(inc)
    co = cooccurrence(inc, n_perm=n_perm)
    hm = harm(inc)

    prev.to_csv(out / "table_prevalence.csv", index=False)
    rel.to_csv(out / "table_reliability.csv", index=False)
    co["spearman"].to_csv(out / "table_spearman.csv")
    co["partial"].to_csv(out / "table_partial.csv")
    hm["table"].to_csv(out / "table_harm.csv", index=False)
    hm["models"].to_csv(out / "table_harm_models.csv", index=False)

    fig_prevalence(prev, out / "fig1_prevalence.png")
    if len(rel):
        fig_reliability(rel, out / "fig2_reliability.png")
    fig_partial(co, out / "fig3_cooccurrence.png")
    fig_null(co, out / "fig4_cooccurrence_null.png")
    fig_leverage(inc, out / "fig5_leverage_drawdown.png")

    pv = prev.assign(share=lambda t: t["share"].map("{:.1%}".format),
                     ci=lambda t: t.apply(lambda r: f"{r['ci_low']:.1%} to {r['ci_high']:.1%}", axis=1))
    pv_cols = ["behaviour", "wallets", "of", "share", "ci"] + (["vs_chance_p"] if "vs_chance_p" in pv else [])
    if "vs_chance_p" in pv:
        pv["vs_chance_p"] = pv["vs_chance_p"].map(lambda v: _fmt_p(v) if pd.notna(v) else "")
    ht = hm["table"]
    harm_md = pd.DataFrame({
        "marker": ht["label"],
        "drawdown rho": ht["max_drawdown_rel_rho"], "q": ht["max_drawdown_rel_q"].map(_fmt_p),
        "loss/capital rho": ht["loss_capped_rho"], "q ": ht["loss_capped_q"].map(_fmt_p),
        "liquidated rho": ht["any_liquidation_rho"], "q  ": ht["any_liquidation_q"].map(_fmt_p),
    })
    mods = hm["models"].assign(p=lambda t: t["p"].map(_fmt_p))

    lines = [
        "# Paper results",
        "",
        "Generated by `hyperbehave paper` from the feature table. Aggregates only.",
        "",
        "## 1. Sample",
        "",
        f"Wallets analysed: {len(features)}. Included after exclusions: {len(inc)}.",
        "",
        md_table(excl) if len(excl) else "No exclusions.",
        "",
        "Balance model of included wallets: "
        + ", ".join(f"{k} {v} ({v / len(inc):.0%})" for k, v in models.items()) + ".",
        "",
        "## 2. How common is each behaviour?",
        "",
        md_table(pv[pv_cols]),
        "",
        "![prevalence](fig1_prevalence.png)",
        "",
        "## 3. Are the markers reliable?",
        "",
        (md_table(rel[["label", "n", "split_half_r", "spearman_brown"]], "{:.2f}") if len(rel)
         else "Split-half columns not found: re-run `hyperbehave features` with v0.4 or later."),
        "",
        "![reliability](fig2_reliability.png)" if len(rel) else "",
        "",
        "## 4. Do gambling-like behaviours go together?",
        "",
        f"Mean pairwise Spearman correlation: {co['mean_r']:.3f}; with activity, capital and active "
        f"days held constant: {co['mean_partial_r']:.3f}.",
        "",
        f"Wallets in the top 10% on 3 or more of {len(co['columns'])} markers: observed "
        f"{co['observed']:.1%}, expected by chance {np.mean(co['null']):.1%} "
        f"(95% range {np.quantile(co['null'], 0.025):.1%} to {np.quantile(co['null'], 0.975):.1%}; "
        f"permutation p = {co['p']:.3f}).",
        "",
        "![partial correlations](fig3_cooccurrence.png)",
        "",
        "![co-occurrence vs chance](fig4_cooccurrence_null.png)",
        "",
        "## 5. Are the behaviours linked to financial harm?",
        "",
        "Spearman correlations; q = Benjamini-Hochberg adjusted p-value within each outcome.",
        "",
        md_table(harm_md, "{:+.2f}"),
        "",
        "Composite score (mean percentile rank of the behavioural markers) in a regression with "
        "log activity, log capital and log active days as controls (HC3 robust standard errors):",
        "",
        md_table(mods[["score", "outcome", "n", "coef", "se", "p"]], "{:.3f}"),
        "",
        "![leverage and drawdown](fig5_leverage_drawdown.png)",
        "",
    ]
    path = out / "results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
