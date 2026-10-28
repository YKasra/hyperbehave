"""Behavioural markers of gambling-like trading.

Each marker is a computable definition tied to a DSM-5 gambling-disorder criterion (or
to a well-known behavioural-finance bias), so that results speak to the addiction
question. ``MARKERS`` is the single source of truth for names, criteria and expected
direction; ``docs/feature_dictionary.md`` mirrors it.

All functions take the reconstructed tables from :mod:`hyperbehave.reconstruct` and
return plain floats (``nan`` when there is not enough data to compute the marker
honestly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NAN = float("nan")
MS_MIN = 60_000
MS_H = 3_600_000
MS_DAY = 86_400_000


@dataclass(frozen=True)
class MarkerSpec:
    name: str
    criterion: str
    description: str
    risk_direction: str  # "higher" = higher value is more gambling-like


MARKERS: list[MarkerSpec] = [
    MarkerSpec("loss_chasing_index", "Chasing losses",
               "P(new position within N min after a losing trip) minus the same after a winning trip", "higher"),
    MarkerSpec("post_loss_size_ratio", "Chasing losses",
               "Median size of the next position after a loss / after a gain", "higher"),
    MarkerSpec("streak_escalation_ratio", "Chasing losses, tolerance",
               "Size of the position after a losing streak / size at the start of the streak", "higher"),
    MarkerSpec("redeposit_after_liq_share", "Chasing losses, loss of control",
               "Share of liquidations followed by new money within N hours", "higher"),
    MarkerSpec("size_trend_weekly", "Tolerance",
               "Weekly slope of log median position size", "higher"),
    MarkerSpec("leverage_trend_weekly", "Tolerance",
               "Weekly slope of log median effective leverage", "higher"),
    MarkerSpec("median_eff_leverage", "Tolerance, risk taking",
               "Median open notional / realised balance", "higher"),
    MarkerSpec("sessions_per_active_day", "Preoccupation",
               "Trading sessions per active day", "higher"),
    MarkerSpec("max_session_min", "Preoccupation, loss of control",
               "Longest continuous session in minutes", "higher"),
    MarkerSpec("rest_window_share", "Loss of control",
               "Share of opening trades inside the wallet's own quietest 5 h daily window (sleep proxy)", "higher"),
    MarkerSpec("hour_week_entropy", "Preoccupation",
               "Normalised entropy of activity over the 168 hours of the week", "higher"),
    MarkerSpec("session_end_loss_streak", "Loss of control",
               "Mean losing streak at which sessions end (stops only when forced)", "higher"),
    MarkerSpec("doubling_down_share", "Chasing losses",
               "Share of positions that were increased while under water", "higher"),
    MarkerSpec("disposition_ratio", "Behavioural bias (disposition effect)",
               "Median holding time of losing trips / winning trips", "higher"),
    MarkerSpec("house_money_ratio", "Behavioural bias (house money)",
               "Median size of the next position after a gain / median position size", "higher"),
    MarkerSpec("volatility_reactivity", "Impulsivity",
               "New positions per hour after market shocks / baseline rate", "higher"),
    MarkerSpec("relapse_count", "Unsuccessful efforts to stop",
               "Breaks of N days or more followed by a return", "higher"),
    MarkerSpec("relapse_size_ratio", "Unsuccessful efforts to stop, tolerance",
               "Position size after returning from a break / before it", "higher"),
    MarkerSpec("lottery_share", "Gambling-like asset choice",
               "Share of traded notional outside the major coins", "higher"),
    MarkerSpec("loss_to_capital", "Harm, jeopardised finances",
               "Net trading loss in the window / capital at risk (balance at window start + money added; 0 when profitable)", "higher"),
    MarkerSpec("max_drawdown_rel", "Harm, jeopardised finances",
               "Max drawdown of cumulative trading PnL in the window / capital at risk", "higher"),
]

CORE_MARKERS = [
    "loss_chasing_index",
    "post_loss_size_ratio",
    "redeposit_after_liq_share",
    "size_trend_weekly",
    "rest_window_share",
    "sessions_per_active_day",
    "doubling_down_share",
    "loss_to_capital",
]


def _median(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else NAN


def _ratio(a: float, b: float) -> float:
    return a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else NAN


# ----------------------------------------------------------------------- chasing
def loss_chasing(trips: pd.DataFrame, window_min: float = 60, min_events: int = 5) -> dict[str, float]:
    out = {"loss_chasing_index": NAN, "post_loss_size_ratio": NAN, "house_money_ratio": NAN}
    if len(trips) < 2:
        return out
    opens = np.sort(trips["open_time"].to_numpy(dtype=np.int64))
    order = np.argsort(trips["open_time"].to_numpy(dtype=np.int64), kind="mergesort")
    sizes_by_open = trips["entry_notional"].to_numpy(dtype=float)[order]
    w = int(window_min * MS_MIN)

    reentry, next_size, outcome = [], [], []
    for close, pnl in zip(trips["close_time"].to_numpy(dtype=np.int64), trips["net_pnl"].to_numpy(dtype=float), strict=True):
        if pnl == 0:
            continue
        i = np.searchsorted(opens, close, side="left")
        j = np.searchsorted(opens, close + w, side="right")
        reentry.append(j > i)
        next_size.append(sizes_by_open[i] if i < len(opens) else NAN)
        outcome.append(pnl < 0)
    reentry = np.asarray(reentry)
    next_size = np.asarray(next_size, dtype=float)
    loss = np.asarray(outcome, dtype=bool)
    if loss.sum() >= min_events and (~loss).sum() >= min_events:
        out["loss_chasing_index"] = float(reentry[loss].mean() - reentry[~loss].mean())
        out["post_loss_size_ratio"] = _ratio(_median(next_size[loss]), _median(next_size[~loss]))
    overall = _median(trips["entry_notional"])
    if (~loss).sum() >= min_events:
        out["house_money_ratio"] = _ratio(_median(next_size[~loss]), overall)
    return out


def streak_escalation(trips: pd.DataFrame, min_streak: int = 3) -> float:
    """Median of (size of the next position / size at the start of the losing streak).

    Comparing with the streak's own first position, not the wallet's overall median,
    keeps the marker from being dragged down by the balance shrinking during the streak.
    """
    if len(trips) < min_streak + 2:
        return NAN
    t = trips.sort_values("open_time", kind="mergesort")
    losses = (t["net_pnl"] < 0).to_numpy()
    sizes = t["entry_notional"].to_numpy(dtype=float)
    ratios = []
    streak, first = 0, 0
    for k in range(len(t) - 1):
        if losses[k]:
            if streak == 0:
                first = k
            streak += 1
        else:
            streak = 0
        if streak >= min_streak:
            ratios.append(_ratio(sizes[k + 1], sizes[first]))
    if len(ratios) < 2:
        return NAN
    return _median(ratios)


def redeposit_after_liquidation(
    liq_times: np.ndarray, ledger: pd.DataFrame, window_h: float = 24
) -> dict[str, float]:
    out = {"n_liquidations": float(len(liq_times)), "redeposit_after_liq_share": NAN,
           "median_redeposit_delay_h": NAN}
    if len(liq_times) == 0:
        return out
    inflows = np.sort(ledger.loc[ledger["is_inflow"], "time"].to_numpy(dtype=np.int64)) if not ledger.empty else np.array([], dtype=np.int64)
    w = int(window_h * MS_H)
    hits, delays = [], []
    for t in liq_times:
        i = np.searchsorted(inflows, t, side="left")
        if i < len(inflows) and inflows[i] - t <= w:
            hits.append(True)
            delays.append((inflows[i] - t) / MS_H)
        else:
            hits.append(False)
    out["redeposit_after_liq_share"] = float(np.mean(hits))
    out["median_redeposit_delay_h"] = _median(delays)
    return out


# ----------------------------------------------------------------------- tolerance
def _weekly_log_slope(times_ms: np.ndarray, values: np.ndarray, min_weeks: int = 4) -> float:
    ok = np.isfinite(values) & (values > 0)
    if ok.sum() < min_weeks:
        return NAN
    week = (times_ms[ok] // (7 * MS_DAY)).astype(np.int64)
    df = pd.DataFrame({"week": week, "v": np.log(values[ok])})
    med = df.groupby("week")["v"].median()
    if len(med) < min_weeks:
        return NAN
    x = med.index.to_numpy(dtype=float)
    return float(np.polyfit(x - x.min(), med.to_numpy(), 1)[0])


def tolerance(trips: pd.DataFrame, exposure: pd.DataFrame) -> dict[str, float]:
    out = {"size_trend_weekly": NAN, "leverage_trend_weekly": NAN, "median_eff_leverage": NAN}
    if not trips.empty:
        out["size_trend_weekly"] = _weekly_log_slope(
            trips["open_time"].to_numpy(dtype=np.int64), trips["entry_notional"].to_numpy(dtype=float))
    if not exposure.empty and exposure["eff_leverage"].notna().any():
        lev = exposure["eff_leverage"].to_numpy(dtype=float)
        open_ = lev > 0
        out["median_eff_leverage"] = _median(lev[open_])
        out["leverage_trend_weekly"] = _weekly_log_slope(
            exposure["time"].to_numpy(dtype=np.int64)[open_], lev[open_])
    return out


# ----------------------------------------------------------------------- time use
def session_markers(fills: pd.DataFrame, sess: pd.DataFrame, trips: pd.DataFrame) -> dict[str, float]:
    out = {"sessions_per_active_day": NAN, "median_session_min": NAN, "max_session_min": NAN,
           "session_end_loss_streak": NAN}
    if fills.empty or sess.empty:
        return out
    days = pd.to_datetime(fills["time"], unit="ms", utc=True).dt.floor("D").nunique()
    out["sessions_per_active_day"] = len(sess) / max(days, 1)
    out["median_session_min"] = _median(sess["duration_min"])
    out["max_session_min"] = float(sess["duration_min"].max())
    if not trips.empty:
        starts = sess["start"].to_numpy(dtype=np.int64)
        t = trips.sort_values("close_time", kind="mergesort")
        sid = np.searchsorted(starts, t["close_time"].to_numpy(dtype=np.int64), side="right") - 1
        streaks = []
        for _, g in t.assign(sid=sid).groupby("sid"):
            losses = (g["net_pnl"] < 0).to_numpy()[::-1]
            n = 0
            for lost in losses:
                if not lost:
                    break
                n += 1
            streaks.append(n)
        out["session_end_loss_streak"] = float(np.mean(streaks)) if streaks else NAN
    return out


def time_of_day(fills: pd.DataFrame, width_h: int = 5, no_rest_threshold: float = 0.02) -> dict[str, float]:
    """Rest-window and weekly-spread markers, measured on deliberate actions.

    Uses active orders (taker orders, one row per order) that open or add to a position:
    a resting limit order or a stop that fills at 3 am is not a decision to trade at
    3 am. Falls back to all active orders when there are too few opening ones.
    """
    from hyperbehave.reconstruct import active_orders

    out = {"rest_window_share": NAN, "no_rest": NAN, "hour_week_entropy": NAN}
    fills = active_orders(fills)
    opens = fills[fills["dir"].astype(str).str.startswith("Open")]
    if len(opens) >= 20:
        fills = opens
    if len(fills) < 20:
        return out
    ts = pd.to_datetime(fills["time"], unit="ms", utc=True)
    hist = np.bincount(ts.dt.hour.to_numpy(), minlength=24).astype(float)
    share = hist / hist.sum()
    circ = np.r_[share, share[: width_h - 1]]
    window = np.convolve(circ, np.ones(width_h), mode="valid")[:24]
    rest = float(window.min())
    out["rest_window_share"] = rest
    out["no_rest"] = float(rest > no_rest_threshold)
    how = (ts.dt.dayofweek.to_numpy() * 24 + ts.dt.hour.to_numpy())
    p = np.bincount(how, minlength=168).astype(float)
    p = p[p > 0] / p.sum()
    out["hour_week_entropy"] = float(-(p * np.log(p)).sum() / np.log(168))
    return out


def volatility_reactivity(trips: pd.DataFrame, shock_times: np.ndarray | None, window_h: float = 2) -> float:
    if trips.empty or shock_times is None or len(shock_times) == 0:
        return NAN
    opens = np.sort(trips["open_time"].to_numpy(dtype=np.int64))
    span_h = (opens[-1] - opens[0]) / MS_H
    if span_h < 24 or len(opens) < 10:
        return NAN
    shocks = shock_times[(shock_times >= opens[0]) & (shock_times <= opens[-1])]
    if len(shocks) < 3:
        return NAN
    w = int(window_h * MS_H)
    after = sum(np.searchsorted(opens, s + w, "right") - np.searchsorted(opens, s, "left") for s in shocks)
    rate_after = after / (len(shocks) * window_h)
    base = len(opens) / span_h
    return _ratio(rate_after, base)


def relapse(trips: pd.DataFrame, break_days: int = 14, before_days: int = 30, after_days: int = 7) -> dict[str, float]:
    out = {"relapse_count": NAN, "relapse_size_ratio": NAN}
    if trips.empty:
        return out
    t = trips.sort_values("open_time", kind="mergesort")
    opens = t["open_time"].to_numpy(dtype=np.int64)
    sizes = t["entry_notional"].to_numpy(dtype=float)
    gaps = np.diff(opens)
    idx = np.where(gaps >= break_days * MS_DAY)[0]
    out["relapse_count"] = float(len(idx))
    ratios = []
    for k in idx:
        stop, back = opens[k], opens[k + 1]
        before = sizes[(opens > stop - before_days * MS_DAY) & (opens <= stop)]
        after = sizes[(opens >= back) & (opens < back + after_days * MS_DAY)]
        if len(before) and len(after):
            ratios.append(_ratio(_median(after), _median(before)))
    if ratios:
        out["relapse_size_ratio"] = _median(ratios)
    return out


# ----------------------------------------------------------------------- control and bias
def position_management(trips: pd.DataFrame) -> dict[str, float]:
    out = {"doubling_down_share": NAN, "disposition_ratio": NAN}
    if trips.empty:
        return out
    if len(trips) >= 10:
        out["doubling_down_share"] = float((trips["n_adverse_adds"] > 0).mean())
    clean = trips[~trips["left_censored"]]
    losers = clean.loc[clean["net_pnl"] < 0, "duration_min"]
    winners = clean.loc[clean["net_pnl"] > 0, "duration_min"]
    if len(losers) >= 5 and len(winners) >= 5:
        out["disposition_ratio"] = _ratio(_median(losers), _median(winners))
    return out


def lottery_share(fills: pd.DataFrame, majors: list[str]) -> float:
    if fills.empty:
        return NAN
    total = fills["notional"].sum()
    if total <= 0:
        return NAN
    return float(fills.loc[~fills["coin"].isin(majors), "notional"].sum() / total)


# ----------------------------------------------------------------------- harm
def harm(balance: pd.DataFrame, ledger: pd.DataFrame, window_start: int | None = None) -> dict[str, float]:
    """Financial harm within the study window, relative to the money at stake.

    Capital at risk = realised balance when the window opens + money added during the
    window. Using lifetime deposits instead would mix a 90-day loss with years of
    history. ``balance_consistent`` is False when the reconstructed balance goes clearly
    negative, which means some money flows are missing from the public ledger (for
    example funds moved in through a channel the ledger does not show). Callers should
    then discard balance-based markers for that wallet.
    """
    out = {"net_trading_pnl": NAN, "start_balance": NAN, "window_inflow": NAN,
           "window_outflow": NAN, "capital_at_risk": NAN, "withdraw_deposit_ratio": NAN,
           "loss_to_capital": NAN, "max_drawdown_usd": NAN, "max_drawdown_rel": NAN,
           "balance_consistent": NAN}
    if balance.empty:
        return out
    if window_start is None:
        window_start = int(balance["time"].min())
    before = balance.loc[balance["time"] < window_start, "balance"]
    start_bal = float(before.iloc[-1]) if len(before) else 0.0
    inside = balance.loc[balance["time"] >= window_start]
    if not ledger.empty:
        lw = ledger.loc[ledger["time"] >= window_start, "flow"]
        inflow, outflow = float(lw[lw > 0].sum()), float(-lw[lw < 0].sum())
    else:
        inflow = outflow = 0.0
    capital = max(start_bal, 0.0) + inflow

    # Cumulative trading result since the window opened (pre-window rows hold only
    # ledger flows, but subtract their trading_pnl anyway to be exact).
    pre = balance.loc[balance["time"] < window_start, "trading_pnl"]
    base = float(pre.iloc[-1]) if len(pre) else 0.0
    curve = np.r_[0.0, inside["trading_pnl"].to_numpy(dtype=float) - base]
    dd = float(np.max(np.maximum.accumulate(curve) - curve))
    net = float(curve[-1])
    min_bal = float(inside["balance"].min()) if len(inside) else start_bal
    consistent = min_bal >= -max(10.0, 0.05 * capital)

    out.update(net_trading_pnl=net, start_balance=start_bal, window_inflow=inflow,
               window_outflow=outflow, capital_at_risk=capital, max_drawdown_usd=dd,
               balance_consistent=float(consistent))
    if capital > 0 and consistent:
        out["loss_to_capital"] = max(-net, 0.0) / capital
        out["max_drawdown_rel"] = dd / capital
        if inflow > 0:
            out["withdraw_deposit_ratio"] = outflow / inflow
    return out


# ----------------------------------------------------------------------- covariates
def activity(fills: pd.DataFrame, trips: pd.DataFrame) -> dict[str, float]:
    if fills.empty:
        return {k: NAN for k in ("n_fills", "n_trips", "active_days", "span_days",
                                 "fills_per_active_day", "total_volume", "median_trip_notional",
                                 "win_rate", "maker_share")}
    days = pd.to_datetime(fills["time"], unit="ms", utc=True).dt.floor("D").nunique()
    span = (fills["time"].max() - fills["time"].min()) / MS_DAY
    return {
        "n_fills": float(len(fills)),
        "n_trips": float(len(trips)),
        "active_days": float(days),
        "span_days": float(span),
        "fills_per_active_day": len(fills) / max(days, 1),
        "total_volume": float(fills["notional"].sum()),
        "median_trip_notional": _median(trips["entry_notional"]) if not trips.empty else NAN,
        "win_rate": float((trips["net_pnl"] > 0).mean()) if not trips.empty else NAN,
        "maker_share": float(1 - fills["crossed"].mean()),
    }


def market_shocks(candles: pd.DataFrame, quantile: float = 0.99) -> np.ndarray:
    """Start times of hours whose absolute return is above ``quantile``.

    ``candles`` needs columns ``t`` (open time, ms) and ``c`` (close).
    """
    if candles is None or len(candles) < 50:
        return np.array([], dtype=np.int64)
    c = candles.sort_values("t")
    r = np.abs(np.diff(np.log(c["c"].astype(float).to_numpy())))
    thr = np.quantile(r, quantile)
    return c["t"].to_numpy(dtype=np.int64)[1:][r >= thr]
