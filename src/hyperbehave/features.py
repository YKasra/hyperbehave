"""Wallet-level feature pipeline: raw records in, one row of markers out."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from hyperbehave import markers as mk
from hyperbehave.filters import bot_flags
from hyperbehave.normalize import (
    fills_to_frame,
    funding_to_frame,
    ledger_to_frame,
    realised_balance_from_state,
)
from hyperbehave.reconstruct import (
    balance_curve,
    exposure_series,
    liquidation_times,
    round_trips,
    sessions,
)

ANCHOR_TOLERANCE_MS = 3_600_000


def _anchor(raw: dict[str, Any]) -> float | None:
    """Use the account snapshot only if the data runs up to the moment it was taken."""
    meta = raw.get("meta") or {}
    end, fetched = meta.get("end_ms"), meta.get("fetched_ms")
    if end is None or fetched is None or fetched - end > ANCHOR_TOLERANCE_MS:
        return None
    return realised_balance_from_state(raw.get("state"))


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
    sess = sessions(fills, m["sessions"]["gap_min"])
    anchor = _anchor(raw)
    bal = balance_curve(fills, ledger, funding, anchor_balance=anchor)
    expo = exposure_series(fills, bal)
    liqs = liquidation_times(fills, ledger)

    row["balance_anchored"] = anchor is not None
    row.update(mk.activity(fills, trips))
    row.update(mk.loss_chasing(trips, m["loss_chasing"]["reentry_window_min"], m["loss_chasing"]["min_events"]))
    row["streak_escalation_ratio"] = mk.streak_escalation(trips, m["loss_streak"]["min_streak"])
    row.update(mk.redeposit_after_liquidation(liqs, ledger, m["liquidation"]["redeposit_window_h"]))
    row.update(mk.tolerance(trips, expo))
    row.update(mk.session_markers(fills, sess, trips))
    row.update(mk.time_of_day(fills, m["rest_window"]["width_h"], m["rest_window"]["no_rest_threshold"]))
    row["volatility_reactivity"] = mk.volatility_reactivity(trips, shock_times, m["volatility"]["window_h"])
    row.update(mk.relapse(trips, m["relapse"]["break_days"], m["relapse"]["before_days"], m["relapse"]["after_days"]))
    row.update(mk.position_management(trips))
    row["lottery_share"] = mk.lottery_share(fills, cfg["study"]["major_coins"])
    row.update(mk.harm(bal, ledger))
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
