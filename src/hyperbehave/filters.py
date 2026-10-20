"""Exclusion rules: keep human discretionary traders, drop bots and thin wallets.

The rules are deliberately simple and transparent so they can be pre-registered and
reported. Every excluded wallet gets a reason, which goes into the exclusion log.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def bot_flags(fills: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    f = cfg["filters"]
    n = len(fills)
    out: dict[str, Any] = {"exclude": False, "reason": ""}
    if n < f["min_fills"]:
        out.update(exclude=True, reason=f"fewer than {f['min_fills']} fills")
        return out
    ts = pd.to_datetime(fills["time"], unit="ms", utc=True)
    days = max(ts.dt.floor("D").nunique(), 1)
    maker_share = 1 - fills["crossed"].mean()
    fills_per_day = n / days
    median_gap_s = float(np.median(np.diff(fills["time"].to_numpy())) / 1000) if n > 1 else np.inf
    span_days = (fills["time"].max() - fills["time"].min()) / 86_400_000
    # Median, over calendar weeks, of the share of the week's 168 hours with any fill.
    # Measured per week so that a busy human with months of history is not mistaken for
    # a bot that never stops within a single week.
    how = ts.dt.dayofweek * 24 + ts.dt.hour
    week = ts.dt.isocalendar()
    per_week = how.groupby([week.year.to_numpy(), week.week.to_numpy()]).nunique() / 168
    hour_coverage = float(per_week.median())
    twap_share = float(fills["is_twap"].mean())

    out.update(maker_share=float(maker_share), fills_per_day=float(fills_per_day),
               median_gap_s=median_gap_s, hour_coverage=float(hour_coverage),
               twap_share=twap_share)

    reasons = []
    if maker_share > f["bot_maker_share"] and fills_per_day > f["bot_fills_per_day"]:
        reasons.append("market-maker pattern")
    if median_gap_s < f["bot_median_gap_s"] and n >= f["bot_min_fills_for_gap_rule"]:
        reasons.append("machine-speed fills")
    if span_days >= 7 and hour_coverage > f["bot_hour_coverage"]:
        reasons.append("active in almost every hour of a typical week")
    if twap_share > 0.5:
        reasons.append("mostly TWAP execution")
    if reasons:
        out.update(exclude=True, reason="; ".join(reasons))
    return out
