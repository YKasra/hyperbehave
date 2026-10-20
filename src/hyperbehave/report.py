"""Markdown and figure reports for the synthetic validation and for Layer A."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hyperbehave import analysis as an
from hyperbehave.markers import CORE_MARKERS, MARKERS

# Reference categorical palette (validated for colour-vision deficiency), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#1f1f1e"
INK_2 = "#6b6a63"
GRID = "#e6e5e0"

ARCHETYPE_ORDER = ["disciplined", "recreational", "gambler"]

# Markers whose direction in the simulation is set by an assumption rather than by a
# programmed gambling rule. Reported, but flagged, so nobody reads AUC as proof.
NOT_PROGRAMMED = {
    "volatility_reactivity": "no reaction to market shocks was programmed",
    "streak_escalation_ratio": "simulated losing streaks mostly end in ruin and a smaller fresh deposit, which caps the next size",
    "session_end_loss_streak": "simulated gamblers pause after wins by design, so they rarely end a session on losses",
    "house_money_ratio": "simulated gamblers reset size after a win by design",
    "leverage_trend_weekly": "leverage is fixed per simulated trader",
    "relapse_count": "breaks happen only after rare full ruin",
    "relapse_size_ratio": "breaks happen only after rare full ruin",
}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Small dependency-free Markdown table renderer."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def strip_figure(features: pd.DataFrame, group_col: str, groups: list[str], cols: list[str],
                 path: Path, title: str) -> None:
    """Small multiples: one dot per wallet, one column per group, median as a tick."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.9 * nrows), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)
    for ax, col in zip(axes, cols, strict=False):
        _style(ax)
        for i, g in enumerate(groups):
            y = pd.to_numeric(features.loc[features[group_col] == g, col], errors="coerce").dropna()
            if y.empty:
                continue
            x = i + rng.uniform(-0.18, 0.18, len(y))
            ax.scatter(x, y, s=16, color=SERIES[i % len(SERIES)], alpha=0.75, linewidths=0.6,
                       edgecolors=SURFACE, label=g)
            ax.hlines(y.median(), i - 0.3, i + 0.3, color=INK, linewidth=2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_title(col, loc="left", fontsize=10, color=INK)
    for ax in axes[n:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncols=len(groups))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, "One dot per wallet; black tick = median.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def validation_report(features: pd.DataFrame, out_dir: str | Path) -> Path:
    """Does each marker separate the programmed gambler from the other archetypes?"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bots = features[features["archetype"] == "bot"]
    humans_excluded = features[(features["archetype"] != "bot") & features["exclude"].astype(bool)]
    inc = an.included(features)
    inc = inc[inc["archetype"] != "bot"]

    names = [m.name for m in MARKERS]
    med = inc.groupby("archetype")[names].median(numeric_only=True).T
    med = med[[a for a in ARCHETYPE_ORDER if a in med.columns]].reset_index(names="marker")
    aucs = an.marker_auc_table(inc, "archetype", "gambler", names)
    aucs["note"] = aucs["marker"].map(NOT_PROGRAMMED).fillna("")

    fig_path = out_dir / "synthetic_core_markers.png"
    strip_figure(inc, "archetype", ARCHETYPE_ORDER, CORE_MARKERS, fig_path,
                 "Core markers by synthetic archetype")

    text = [
        "# Synthetic validation",
        "",
        f"Wallets: {len(features)} ({(features['archetype'] != 'bot').sum()} human archetypes, {len(bots)} bots).",
        f"Bots removed by the exclusion rules: {int(bots['exclude'].astype(bool).sum())} of {len(bots)}.",
        f"Human wallets wrongly excluded: {len(humans_excluded)}.",
        "",
        "## Can each marker tell the programmed gambler apart?",
        "",
        "AUC = probability that a random gambler scores higher than a random non-gambler "
        "(0.5 = no separation, 1.0 = perfect). Coverage = share of wallets where the marker "
        "could be computed. A note marks markers whose simulated direction comes from an "
        "assumption, not from a programmed gambling rule: only real labelled data can test those.",
        "",
        md_table(aucs),
        "",
        "## Medians by archetype",
        "",
        md_table(med),
        "",
        f"![core markers]({fig_path.name})",
        "",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def layer_a_report(features: pd.DataFrame, out_dir: str | Path, with_profiles: bool = True) -> Path:
    """Descriptive population report on real wallets (no addresses, aggregates only)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inc = an.included(features)
    flagged = an.high_risk_flags(inc)
    desc = an.describe(inc)
    corr = an.co_occurrence(inc).round(2).reset_index(names="marker")
    excl = an.exclusion_log(features) if features["exclude"].astype(bool).any() else pd.DataFrame(
        {"reason": [], "n_wallets": []})
    prev = float(flagged["multi_marker_high_risk"].mean()) if len(flagged) else float("nan")

    text = [
        "# Layer A: population report",
        "",
        f"Wallets analysed: {len(features)}; included after exclusions: {len(inc)}.",
        f"Wallets in the top decile on 3 or more core markers: {prev:.1%}.",
        "",
        "This is a descriptive screen of behaviour, not a diagnosis of any person.",
        "",
        "## Exclusions",
        "",
        md_table(excl) if len(excl) else "None.",
        "",
        "## Marker distributions",
        "",
        md_table(desc),
        "",
        "## Do gambling-like signs co-occur? (Spearman correlations)",
        "",
        md_table(corr, "{:.2f}"),
        "",
    ]
    if with_profiles and len(inc) >= 20:
        labels, k = an.behavioural_profiles(inc)
        prof = inc.assign(profile=labels).groupby("profile")[CORE_MARKERS].median(numeric_only=True)
        prof.insert(0, "n_wallets", labels.value_counts().sort_index())
        text += [f"## Behavioural profiles (Gaussian mixture, k = {k} by BIC)", "",
                 md_table(prof.reset_index()), ""]
    path = out_dir / "README.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path
