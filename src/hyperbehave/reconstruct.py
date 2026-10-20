"""Rebuild trading behaviour from raw fills.

Three views are reconstructed from the fill tape:

* **Round trips**: a position in one coin from the moment it leaves zero until it is
  flat again (or flips side). This is the unit of a "bet": it has a direction, a size,
  a holding time and a realised result.
* **Sessions**: bursts of activity separated by quiet gaps.
* **Balance curve**: realised account balance over time, used to turn position size into
  effective leverage. When the current account value is known (``anchor_balance``), the
  curve is rebuilt backwards from it, which is exact for realised flows. Without an
  anchor it is built forwards from zero, which is only reliable if the window starts at
  the wallet's first deposit.

``start_pos`` reported by the API is treated as the truth for each fill, so a missing
fill cannot silently corrupt the following trips.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-12

TRIP_COLUMNS = [
    "coin", "side", "open_time", "close_time", "duration_min", "pnl", "fees", "net_pnl",
    "entry_notional", "max_notional", "n_fills", "n_adds", "n_adverse_adds", "liquidated",
    "left_censored",
]


@dataclass
class _Trip:
    coin: str
    side: int
    open_time: int
    avg_entry: float
    size: float
    entry_notional: float
    left_censored: bool = False
    pnl: float = 0.0
    fees: float = 0.0
    max_notional: float = 0.0
    n_fills: int = 0
    n_adds: int = 0
    n_adverse_adds: int = 0
    liquidated: bool = False
    extra: dict = field(default_factory=dict)

    def to_row(self, close_time: int) -> dict:
        return {
            "coin": self.coin,
            "side": self.side,
            "open_time": self.open_time,
            "close_time": close_time,
            "duration_min": (close_time - self.open_time) / 60_000,
            "pnl": self.pnl,
            "fees": self.fees,
            "net_pnl": self.pnl - self.fees,
            "entry_notional": self.entry_notional,
            "max_notional": self.max_notional,
            "n_fills": self.n_fills,
            "n_adds": self.n_adds,
            "n_adverse_adds": self.n_adverse_adds,
            "liquidated": self.liquidated,
            "left_censored": self.left_censored,
        }


def _sign(x: float) -> int:
    return 0 if abs(x) < EPS else (1 if x > 0 else -1)


def round_trips(fills: pd.DataFrame) -> pd.DataFrame:
    """Closed round trips, one row per position from open to flat (or flip)."""
    if fills.empty:
        return pd.DataFrame(columns=TRIP_COLUMNS)
    rows: list[dict] = []
    for coin, g in fills.groupby("coin", sort=False):
        trip: _Trip | None = None
        for f in g.itertuples(index=False):
            prev, new = f.start_pos, f.end_pos
            s_prev, s_new = _sign(prev), _sign(new)

            if s_prev == 0:
                # Flat before this fill. A still-open trip means fills were missing: drop it.
                trip = None
            elif trip is None:
                # Position already open when our window starts.
                trip = _Trip(coin, s_prev, int(f.time), f.px, abs(prev), abs(prev) * f.px,
                             left_censored=True)

            if s_prev == 0 and s_new != 0:
                trip = _Trip(coin, s_new, int(f.time), f.px, abs(new), abs(new) * f.px)
                trip.n_fills = 1
                trip.fees += f.fee
                trip.pnl += f.closed_pnl
                trip.max_notional = abs(new) * f.px
                trip.liquidated |= bool(f.is_liquidation)
                continue

            if trip is None:
                continue

            trip.n_fills += 1
            trip.fees += f.fee
            trip.pnl += f.closed_pnl
            trip.liquidated |= bool(f.is_liquidation)

            if s_new == s_prev and abs(new) > abs(prev):  # adding to the position
                trip.n_adds += 1
                adverse = (f.px < trip.avg_entry) if trip.side > 0 else (f.px > trip.avg_entry)
                trip.n_adverse_adds += int(adverse)
                total = abs(new)
                trip.avg_entry = (trip.avg_entry * abs(prev) + f.px * f.sz) / total
                trip.size = total
                trip.max_notional = max(trip.max_notional, total * f.px)
            elif s_new == s_prev:  # partial reduce
                trip.size = abs(new)
            elif s_new == 0:  # closed flat
                rows.append(trip.to_row(int(f.time)))
                trip = None
            else:  # flipped side: close this trip, open the opposite one
                rows.append(trip.to_row(int(f.time)))
                trip = _Trip(coin, s_new, int(f.time), f.px, abs(new), abs(new) * f.px)
                trip.n_fills = 1
                trip.max_notional = abs(new) * f.px
    if not rows:
        return pd.DataFrame(columns=TRIP_COLUMNS)
    out = pd.DataFrame(rows, columns=TRIP_COLUMNS).sort_values("close_time", kind="mergesort")
    return out.reset_index(drop=True)


def sessions(fills: pd.DataFrame, gap_min: float = 30.0) -> pd.DataFrame:
    """Group fills into sessions separated by more than ``gap_min`` minutes."""
    cols = ["session", "start", "end", "n_fills", "duration_min"]
    if fills.empty:
        return pd.DataFrame(columns=cols)
    t = fills["time"].to_numpy()
    new = np.r_[True, np.diff(t) > gap_min * 60_000]
    sid = np.cumsum(new) - 1
    df = pd.DataFrame({"session": sid, "time": t})
    agg = df.groupby("session")["time"].agg(start="min", end="max", n_fills="count").reset_index()
    agg["duration_min"] = (agg["end"] - agg["start"]) / 60_000
    return agg[cols]


def balance_curve(
    fills: pd.DataFrame,
    ledger: pd.DataFrame,
    funding: pd.DataFrame,
    anchor_balance: float | None = None,
) -> pd.DataFrame:
    """Realised balance after every cash event (flow, realised PnL, fee, funding).

    Returns columns ``time, delta, balance, trading_pnl`` where ``trading_pnl`` is the
    cumulative realised result of trading only (no deposits), the basis for drawdowns.
    """
    parts = []
    if not fills.empty:
        parts.append(pd.DataFrame({"time": fills["time"], "delta": fills["closed_pnl"] - fills["fee"],
                                   "trading": True}))
    if not funding.empty:
        parts.append(pd.DataFrame({"time": funding["time"], "delta": funding["usdc"], "trading": True}))
    if not ledger.empty:
        parts.append(pd.DataFrame({"time": ledger["time"], "delta": ledger["flow"], "trading": False}))
    if not parts:
        return pd.DataFrame(columns=["time", "delta", "balance", "trading_pnl"])
    ev = pd.concat(parts, ignore_index=True).sort_values("time", kind="mergesort")
    ev["trading_pnl"] = ev["delta"].where(ev["trading"], 0.0).cumsum()
    cum = ev["delta"].cumsum()
    if anchor_balance is not None:
        ev["balance"] = anchor_balance - (cum.iloc[-1] - cum)
    else:
        ev["balance"] = cum
    return ev[["time", "delta", "balance", "trading_pnl"]].reset_index(drop=True)


def exposure_series(fills: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """Total open notional across coins after each fill, and effective leverage.

    Effective leverage = open notional / realised balance at that time. Rows with a
    balance under 1 USDC are left as NaN.
    """
    if fills.empty:
        return pd.DataFrame(columns=["time", "open_notional", "balance", "eff_leverage"])
    pos: dict[str, float] = {}
    px: dict[str, float] = {}
    notional = np.empty(len(fills))
    for i, f in enumerate(fills.itertuples(index=False)):
        pos[f.coin] = f.end_pos
        px[f.coin] = f.px
        notional[i] = sum(abs(pos[c]) * px[c] for c in pos)
    out = pd.DataFrame({"time": fills["time"].to_numpy(), "open_notional": notional})
    if balance.empty:
        out["balance"] = np.nan
    else:
        b = balance[["time", "balance"]].drop_duplicates("time", keep="last")
        out = pd.merge_asof(out.sort_values("time"), b.sort_values("time"), on="time",
                            direction="backward")
    bal = out["balance"].where(out["balance"] >= 1.0)
    out["eff_leverage"] = out["open_notional"] / bal
    return out


def liquidation_times(fills: pd.DataFrame, ledger: pd.DataFrame, merge_ms: int = 60_000) -> np.ndarray:
    """Distinct liquidation events (ledger events and liquidation fills, merged within 1 min)."""
    times = []
    if not ledger.empty:
        times.extend(ledger.loc[ledger["type"] == "liquidation", "time"].tolist())
    if not fills.empty:
        times.extend(fills.loc[fills["is_liquidation"], "time"].tolist())
    if not times:
        return np.array([], dtype=np.int64)
    t = np.sort(np.asarray(times, dtype=np.int64))
    keep = np.r_[True, np.diff(t) > merge_ms]
    return t[keep]
