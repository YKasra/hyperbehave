"""Wallet-level feature pipeline: raw records in, one row of markers out."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from hyperbehave import markers as mk
from hyperbehave.filters import bot_flags
from hyperbehave.normalize import (
    default_dex_only,
    fills_to_frame,
    funding_to_frame,
    ledger_to_frame,
    realised_balance_from_state,
    spot_cash_frame,
    spot_usdc,
    use_wallet_flows,
)
from hyperbehave.reconstruct import (
    active_orders,
    balance_curve,
    exposure_series,
    liquidation_times,
    round_trips,
    sessions,
)

ANCHOR_TOLERANCE_MS = 3_600_000

# Markers whose split-half reliability is estimated (even vs odd calendar days).
RELIABILITY_MARKERS = [
    "loss_chasing_index", "post_loss_size_ratio", "doubling_down_share", "disposition_ratio",
    "rest_window_share", "hour_week_entropy", "sessions_per_active_day", "median_eff_leverage",
    "lottery_share",
]


def _day_parity(times) -> np.ndarray:
    return (np.asarray(times, dtype=np.int64) // 86_400_000) % 2


def split_half(fills: pd.DataFrame, trips: pd.DataFrame, expo: pd.DataFrame,
               cfg: dict[str, Any]) -> dict[str, float]:
    """Each reliability marker computed separately on even and odd calendar days.

    A marker that reflects a stable trait should rank wallets the same way in both
    halves; the correlation across wallets (with the Spearman-Brown correction) is its
    split-half reliability. Returned as ``<marker>_h0`` and ``<marker>_h1``.
    """
    m = cfg["markers"]
    out: dict[str, float] = {}
    for h in (0, 1):
        f = fills[_day_parity(fills["time"]) == h]
        tr = trips[_day_parity(trips["open_time"]) == h] if len(trips) else trips
        e = expo[_day_parity(expo["time"]) == h] if len(expo) else expo
        vals: dict[str, float] = {}
        vals.update(mk.loss_chasing(tr, m["loss_chasing"]["reentry_window_min"],
                                    m["loss_chasing"]["min_events"], n_perm=0))
        vals.update(mk.position_management(tr))
        vals.update(mk.time_of_day(f, m["rest_window"]["width_h"], m["rest_window"]["no_rest_threshold"]))
        if len(f):
            vals.update(mk.session_markers(f, sessions(active_orders(f), m["sessions"]["gap_min"]), tr))
        lev = e["eff_leverage"].to_numpy(dtype=float) if len(e) else np.array([])
        lev = lev[np.isfinite(lev) & (lev > 0)]
        vals["median_eff_leverage"] = float(np.median(lev)) if lev.size else float("nan")
        vals["lottery_share"] = mk.lottery_share(f, cfg["study"]["major_coins"])
        for key in RELIABILITY_MARKERS:
            out[f"{key}_h{h}"] = vals.get(key, float("nan"))
    return out


def _anchor(raw: dict[str, Any]) -> float | None:
    """Use the account snapshot only if the data runs up to the moment it was taken."""
    meta = raw.get("meta") or {}
    end, fetched = meta.get("end_ms"), meta.get("fetched_ms")
    if end is None or fetched is None or fetched - end > ANCHOR_TOLERANCE_MS:
        return None
    return realised_balance_from_state(raw.get("state"))


def reconstruct_balance(
    raw: dict[str, Any],
    fills: pd.DataFrame,
    ledger: pd.DataFrame,
    funding: pd.DataFrame,
    window_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], str]:
    """Rebuild the trader's USDC balance on the exchange.

    Models, in order of preference:

    1. ``perp+spot``: default perp account and spot USDC taken together, counting only
       money that enters or leaves the wallet (moves between the wallet's own perp and
       spot sides cancel out). This is correct in every account mode, including unified
       accounts and portfolio margin, where spot USDC backs perp positions directly.
       Needs the spot snapshot.
    2. ``perp``: the default perp account alone. Used when there is no spot snapshot or
       when the first model does not add up (for example spot pairs not quoted in USDC).
    3. ``failed``: neither adds up; balance-based markers are discarded.

    Builder-deployed markets settle in their own accounts, outside both snapshots, so
    their results never enter the balance (money moved to them counts as leaving).
    Returns (balance curve, ledger used, harm markers, model).
    """
    anchor = _anchor(raw)
    main_fills, main_funding = default_dex_only(fills), default_dex_only(funding)
    spot = raw.get("spot_state")
    if anchor is not None and spot:
        led_w = use_wallet_flows(ledger)
        bal_w = balance_curve(main_fills, led_w, main_funding, anchor_balance=anchor + spot_usdc(spot),
                              extra_cash=spot_cash_frame(raw.get("fills", [])))
        harm_w = mk.harm(bal_w, led_w, window_start)
        if harm_w["balance_consistent"] == 1.0:
            return bal_w, led_w, harm_w, "perp+spot"
    bal = balance_curve(main_fills, ledger, main_funding, anchor_balance=anchor)
    harm = mk.harm(bal, ledger, window_start)
    if harm["balance_consistent"] == 1.0:
        return bal, ledger, harm, "perp"
    return bal, ledger, harm, "failed"


def wallet_features(
    address: str,
    raw: dict[str, Any],
    cfg: dict[str, Any],
    shock_times: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute every marker for one wallet. Never raises on thin data: returns NaNs."""
    m = cfg["markers"]
    fills = fills_to_frame(raw.get("fills", []), address)
    ledger = ledger_to_frame(raw.get("ledger", []), address)
    funding = funding_to_frame(raw.get("funding", []))

    row: dict[str, Any] = {}
    row.update(bot_flags(fills, cfg))
    if fills.empty:
        return row

    trips = round_trips(fills)
    # Sessions are built from the trader's own actions (taker orders), not from resting
    # limit orders that fill while they are away.
    sess = sessions(active_orders(fills), m["sessions"]["gap_min"])
    window_start = (raw.get("meta") or {}).get("start_ms")
    if window_start is None:
        window_start = int(fills["time"].min())
    bal, ledger_used, harm, model = reconstruct_balance(raw, fills, ledger, funding, int(window_start))
    main_fills = default_dex_only(fills)
    expo = exposure_series(main_fills, bal)
    liqs = liquidation_times(fills, ledger)

    row["balance_anchored"] = _anchor(raw) is not None
    row["balance_model"] = model
    row["builder_perp_share"] = float(1 - len(main_fills) / len(fills))
    row.update(mk.activity(fills, trips))
    row.update(mk.loss_chasing(trips, m["loss_chasing"]["reentry_window_min"], m["loss_chasing"]["min_events"]))
    row["streak_escalation_ratio"] = mk.streak_escalation(trips, m["loss_streak"]["min_streak"])
    row.update(mk.redeposit_after_liquidation(liqs, ledger_used, m["liquidation"]["redeposit_window_h"]))
    row.update(mk.tolerance(trips, expo))
    row.update(mk.session_markers(fills, sess, trips))
    row.update(mk.time_of_day(fills, m["rest_window"]["width_h"], m["rest_window"]["no_rest_threshold"]))
    row["volatility_reactivity"] = mk.volatility_reactivity(trips, shock_times, m["volatility"]["window_h"])
    row.update(mk.relapse(trips, m["relapse"]["break_days"], m["relapse"]["before_days"], m["relapse"]["after_days"]))
    row.update(mk.position_management(trips))
    row["lottery_share"] = mk.lottery_share(fills, cfg["study"]["major_coins"])
    row.update(harm)
    halves = split_half(fills, trips, expo, cfg)
    if model == "failed":
        # Missing money flows make every balance-based marker unreliable for this wallet.
        for key in ("median_eff_leverage", "leverage_trend_weekly",
                    "median_eff_leverage_h0", "median_eff_leverage_h1"):
            row[key] = float("nan")
            halves.pop(key, None)
    row.update(halves)
    return row


def build_feature_table(
    wallets: Iterable[tuple[str, dict[str, Any]]],
    cfg: dict[str, Any],
    salt: str,
    shock_times: np.ndarray | None = None,
    keep_address: bool = False,
    extra: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """One row per wallet, identified by a salted hash (never the raw address).

    ``extra`` maps address -> additional columns (for example a synthetic archetype).
    """
    from hyperbehave.storage import wallet_id

    rows = []
    for address, raw in wallets:
        row = {"wallet_id": wallet_id(address, salt)}
        if keep_address:
            row["address"] = address
        if extra and address in extra:
            row.update(extra[address])
        row.update(wallet_features(address, raw, cfg, shock_times))
        rows.append(row)
    return pd.DataFrame(rows)
