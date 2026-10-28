"""Explain why a wallet's money flows do not add up (pseudonymous output only).

For each wallet the balance curve is rebuilt exactly as in the feature pipeline, then
the pieces that usually explain a gap are printed side by side:

* fills in builder-deployed markets (``dex:COIN``) and spot, which settle outside the
  default perp account;
* USDC held on the spot side, which can back perp positions in unified or portfolio
  margin modes without any transfer showing in the ledger;
* how far below zero the rebuilt balance goes, compared with the money at stake;
* the ledger event types present.

Run ``hyperbehave diagnose`` and paste the output when asking for help: it contains no
wallet addresses.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from hyperbehave.features import _anchor
from hyperbehave.normalize import (
    default_dex_only,
    fills_to_frame,
    funding_to_frame,
    is_builder_perp,
    is_spot_coin,
    ledger_to_frame,
    spot_usdc,
)
from hyperbehave.reconstruct import balance_curve


def wallet_diagnosis(address: str, raw: dict[str, Any], min_fills: int = 50) -> dict[str, Any] | None:
    coins = Counter()
    for f in raw.get("fills", []):
        c = str(f.get("coin", ""))
        coins["spot" if is_spot_coin(c) else ("builder" if is_builder_perp(c) else "perp")] += 1
    if coins["perp"] + coins["builder"] < min_fills:
        return None
    fills = fills_to_frame(raw.get("fills", []), address)
    ledger = ledger_to_frame(raw.get("ledger", []), address)
    funding = funding_to_frame(raw.get("funding", []))
    anchor = _anchor(raw)
    bal = balance_curve(default_dex_only(fills), ledger, default_dex_only(funding), anchor_balance=anchor)
    start = (raw.get("meta") or {}).get("start_ms") or (int(fills["time"].min()) if len(fills) else 0)
    inside = bal.loc[bal["time"] >= start] if len(bal) else bal
    before = bal.loc[bal["time"] < start, "balance"] if len(bal) else pd.Series(dtype=float)
    start_bal = float(before.iloc[-1]) if len(before) else 0.0
    inflow = float(ledger.loc[(ledger["time"] >= start) & (ledger["flow"] > 0), "flow"].sum()) if len(ledger) else 0.0
    capital = max(start_bal, 0.0) + inflow
    min_bal = float(inside["balance"].min()) if len(inside) else float("nan")
    pnl = float(default_dex_only(fills)["closed_pnl"].sum() - default_dex_only(fills)["fee"].sum())
    state = raw.get("state") or {}
    summary = state.get("marginSummary") or {}
    spot = raw.get("spot_state")
    types = Counter((r.get("delta") or {}).get("type", "?") for r in raw.get("ledger", []))
    return {
        "consistent": min_bal >= -max(10.0, 0.05 * capital),
        "perp_fills": coins["perp"],
        "builder_fills": coins["builder"],
        "spot_fills": coins["spot"],
        "perp_account_value": float(summary.get("accountValue", "nan") or "nan"),
        "spot_usdc": spot_usdc(spot) if spot else float("nan"),
        "start_balance": round(start_bal, 2),
        "window_inflow": round(inflow, 2),
        "window_trading_pnl": round(pnl, 2),
        "min_rebuilt_balance": round(min_bal, 2),
        "gap_vs_capital": round(-min_bal / capital, 2) if capital > 0 and min_bal < 0 else 0.0,
        "anchored": anchor is not None,
        "ledger_types": ", ".join(f"{k}:{v}" for k, v in types.most_common()),
        "state_keys": ", ".join(sorted(state.keys())),
    }


def diagnose(
    wallets: Iterable[tuple[str, dict[str, Any]]],
    salt: str,
    only_inconsistent: bool = True,
    fetch_spot: Callable[[str], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    from hyperbehave.storage import wallet_id

    rows = []
    for address, raw in wallets:
        if fetch_spot is not None and not raw.get("spot_state"):
            raw = {**raw, "spot_state": fetch_spot(address)}
        d = wallet_diagnosis(address, raw)
        if d is None or (only_inconsistent and d["consistent"]):
            continue
        rows.append({"wallet": wallet_id(address, salt)[:8], **d})
    return pd.DataFrame(rows)
