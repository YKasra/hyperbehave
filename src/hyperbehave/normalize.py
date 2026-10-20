"""Turn raw API records into typed pandas tables.

Hyperliquid returns every number as a string. These functions convert them, drop spot
trades (this project studies perpetual futures), and classify ledger events into money
flowing into or out of the trading account.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FILL_COLUMNS = [
    "time", "ts", "coin", "px", "sz", "side", "dir", "start_pos", "signed_sz", "end_pos",
    "notional", "closed_pnl", "fee", "crossed", "is_liquidation", "is_twap", "oid", "tid",
]

LEDGER_COLUMNS = ["time", "ts", "type", "flow", "is_inflow", "is_outflow", "external", "account_value"]

FUNDING_COLUMNS = ["time", "ts", "coin", "usdc"]


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def is_spot_coin(coin: str) -> bool:
    """Spot markets show up as ``@123`` or ``BASE/QUOTE``; perps are bare symbols."""
    return coin.startswith("@") or "/" in coin


def fills_to_frame(fills: list[dict[str, Any]], address: str | None = None) -> pd.DataFrame:
    """Typed, sorted, de-duplicated perp fills."""
    if not fills:
        return pd.DataFrame(columns=FILL_COLUMNS)
    rows = []
    addr = address.lower() if address else None
    for f in fills:
        coin = str(f.get("coin", ""))
        if not coin or is_spot_coin(coin):
            continue
        sz = _f(f.get("sz"))
        px = _f(f.get("px"))
        side = f.get("side")
        start = _f(f.get("startPosition"))
        signed = sz if side == "B" else -sz
        liq = f.get("liquidation")
        if isinstance(liq, dict):
            liq_user = str(liq.get("liquidatedUser", "")).lower()
            is_liq = (addr is None) or (liq_user == addr)
        else:
            is_liq = False
        rows.append(
            {
                "time": int(f["time"]),
                "coin": coin,
                "px": px,
                "sz": sz,
                "side": side,
                "dir": f.get("dir", ""),
                "start_pos": start,
                "signed_sz": signed,
                "end_pos": start + signed,
                "notional": px * sz,
                "closed_pnl": _f(f.get("closedPnl", 0)),
                "fee": _f(f.get("fee", 0)),
                "crossed": bool(f.get("crossed", False)),
                "is_liquidation": is_liq,
                "is_twap": f.get("twapId") not in (None, 0, "0"),
                "oid": f.get("oid"),
                "tid": f.get("tid"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=FILL_COLUMNS)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["tid", "oid", "time", "sz", "px"]).sort_values(
        ["time", "tid"], kind="mergesort"
    )
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df[["closed_pnl", "fee"]] = df[["closed_pnl", "fee"]].fillna(0.0)
    return df[FILL_COLUMNS].reset_index(drop=True)


def _ledger_flow(delta: dict[str, Any], address: str | None) -> tuple[float, bool]:
    """Signed USDC flow into the perp trading account, and whether it is external money."""
    kind = delta.get("type")
    usdc = _f(delta.get("usdc", 0))
    if np.isnan(usdc):
        usdc = 0.0
    addr = address.lower() if address else None
    if kind == "deposit":
        return usdc, True
    if kind == "withdraw":
        return -usdc, True
    if kind in ("internalTransfer", "subAccountTransfer", "spotTransfer", "send"):
        dest = str(delta.get("destination", "")).lower()
        incoming = addr is not None and dest == addr
        return (usdc if incoming else -usdc), True
    if kind == "accountClassTransfer":
        return (usdc if delta.get("toPerp") else -usdc), False
    if kind == "vaultDeposit":
        return -usdc, False
    if kind == "vaultWithdraw":
        amount = _f(delta.get("netWithdrawnUsd", delta.get("usdc", 0)))
        return (0.0 if np.isnan(amount) else amount), False
    return 0.0, False


def ledger_to_frame(ledger: list[dict[str, Any]], address: str | None = None) -> pd.DataFrame:
    if not ledger:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    rows = []
    for rec in ledger:
        delta = rec.get("delta") or {}
        flow, external = _ledger_flow(delta, address)
        rows.append(
            {
                "time": int(rec["time"]),
                "type": delta.get("type", "unknown"),
                "flow": flow,
                "external": external,
                "account_value": _f(delta.get("accountValue")),
            }
        )
    df = pd.DataFrame(rows).sort_values("time", kind="mergesort")
    df["is_inflow"] = df["flow"] > 0
    df["is_outflow"] = df["flow"] < 0
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df[LEDGER_COLUMNS].reset_index(drop=True)


def realised_balance_from_state(state: dict[str, Any] | None) -> float | None:
    """Account value minus unrealised PnL from a ``clearinghouseState`` snapshot.

    Returns ``None`` when the snapshot is missing, so callers fall back to a forward
    reconstruction.
    """
    if not state:
        return None
    summary = state.get("marginSummary") or {}
    value = _f(summary.get("accountValue"))
    if np.isnan(value):
        return None
    unrealised = 0.0
    for ap in state.get("assetPositions") or []:
        u = _f((ap.get("position") or {}).get("unrealizedPnl", 0))
        if not np.isnan(u):
            unrealised += u
    return value - unrealised


def funding_to_frame(funding: list[dict[str, Any]]) -> pd.DataFrame:
    if not funding:
        return pd.DataFrame(columns=FUNDING_COLUMNS)
    rows = []
    for rec in funding:
        delta = rec.get("delta") or {}
        rows.append(
            {"time": int(rec["time"]), "coin": delta.get("coin", ""), "usdc": _f(delta.get("usdc", 0))}
        )
    df = pd.DataFrame(rows).sort_values("time", kind="mergesort")
    df["usdc"] = df["usdc"].fillna(0.0)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df[FUNDING_COLUMNS].reset_index(drop=True)
