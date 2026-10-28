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

LEDGER_COLUMNS = ["time", "ts", "type", "flow", "flow_wallet", "is_inflow", "is_outflow", "external", "account_value"]

FUNDING_COLUMNS = ["time", "ts", "coin", "usdc"]


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def is_spot_coin(coin: str) -> bool:
    """Spot markets show up as ``@123`` or ``BASE/QUOTE``; perps are bare symbols."""
    return coin.startswith("@") or "/" in coin


def is_builder_perp(coin: str) -> bool:
    """Builder-deployed perps (HIP-3) are named ``dex:SYMBOL`` and settle in their own
    margin account, outside the default perp account that the snapshot reports."""
    return ":" in coin


def default_dex_only(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of a fills or funding table that belong to the default perp account."""
    if df.empty or "coin" not in df:
        return df
    return df.loc[~df["coin"].astype(str).str.contains(":", regex=False)]


def spot_usdc(spot_state: dict[str, Any] | None) -> float:
    """Total USDC held on the spot side, 0 when unknown."""
    for b in (spot_state or {}).get("balances") or []:
        if str(b.get("coin", "")).upper() == "USDC":
            v = _f(b.get("total"))
            return 0.0 if np.isnan(v) else v
    return 0.0


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


def _usdc_amount(delta: dict[str, Any]) -> float:
    """USDC value of a ledger delta, whichever field the event type uses.

    ``deposit``, ``withdraw`` and transfers use ``usdc``; ``send`` uses ``usdcValue`` or
    ``amount`` (with ``token``).
    """
    for key in ("usdc", "usdcValue"):
        if key in delta:
            v = _f(delta[key])
            if np.isfinite(v):
                return v
    if "amount" in delta and str(delta.get("token", "USDC")).upper().startswith("USDC"):
        v = _f(delta["amount"])
        if np.isfinite(v):
            return v
    return 0.0


def _is_usdc(delta: dict[str, Any]) -> bool:
    return str(delta.get("token", "USDC")).upper().startswith("USDC")


def _send_parties(delta: dict[str, Any], addr: str | None) -> tuple[bool, bool]:
    """(money arrives at this wallet, money leaves this wallet). Both are True for a
    transfer between the wallet's own accounts (for example perp to spot)."""
    dest = str(delta.get("destination", "")).lower()
    user = str(delta.get("user", "")).lower()
    incoming = addr is not None and dest == addr
    outgoing = addr is not None and (user == addr if user else not incoming)
    return incoming, outgoing


def _perp_flow(delta: dict[str, Any], addr: str | None) -> float:
    """USDC flow into the default perp account (standard account mode)."""
    kind = delta.get("type")
    usdc = _usdc_amount(delta)
    if kind == "deposit":
        return usdc
    if kind == "withdraw":
        return -usdc
    if kind == "send":
        if not _is_usdc(delta):
            return 0.0
        incoming, outgoing = _send_parties(delta, addr)
        flow = 0.0
        if incoming and delta.get("destinationDex", "") in ("", None):
            flow += usdc
        if outgoing and delta.get("sourceDex", "") in ("", None):
            flow -= usdc
        return flow
    if kind in ("internalTransfer", "subAccountTransfer"):
        incoming, outgoing = _send_parties(delta, addr)
        return (usdc if incoming else 0.0) - (usdc if outgoing else 0.0)
    if kind == "accountClassTransfer":
        return usdc if delta.get("toPerp") else -usdc
    if kind == "vaultDeposit":
        return -usdc
    if kind == "vaultWithdraw":
        amount = _f(delta.get("netWithdrawnUsd", delta.get("usdc", 0)))
        return 0.0 if np.isnan(amount) else amount
    return 0.0  # spotTransfer, spotGenesis, rewards, staking, borrowLend: not perp cash


def _wallet_flow(delta: dict[str, Any], addr: str | None) -> float:
    """USDC flow into the wallet's default perp + spot accounts taken together.

    Used for unified-account / portfolio-margin wallets, where spot USDC backs perp
    positions directly. Moves between the wallet's own perp and spot sides cancel out;
    moves to or from builder-deployed perp accounts count, since those balances are
    outside the two snapshots.
    """
    kind = delta.get("type")
    usdc = _usdc_amount(delta)
    if kind == "deposit":
        return usdc
    if kind == "withdraw":
        return -usdc
    if kind == "send":
        if not _is_usdc(delta):
            return 0.0
        incoming, outgoing = _send_parties(delta, addr)
        flow = 0.0
        if incoming and delta.get("destinationDex", "") in ("", "spot", None):
            flow += usdc
        if outgoing and delta.get("sourceDex", "") in ("", "spot", None):
            flow -= usdc
        return flow
    if kind in ("spotTransfer", "internalTransfer", "subAccountTransfer"):
        if kind == "spotTransfer" and not _is_usdc(delta):
            return 0.0
        incoming, outgoing = _send_parties(delta, addr)
        return (usdc if incoming else 0.0) - (usdc if outgoing else 0.0)
    if kind == "vaultDeposit":
        return -usdc
    if kind == "vaultWithdraw":
        amount = _f(delta.get("netWithdrawnUsd", delta.get("usdc", 0)))
        return 0.0 if np.isnan(amount) else amount
    return 0.0  # accountClassTransfer is internal; tokens, staking, borrowLend not USDC cash


EXTERNAL_TYPES = {"deposit", "withdraw", "send", "spotTransfer", "internalTransfer", "subAccountTransfer"}


def ledger_to_frame(ledger: list[dict[str, Any]], address: str | None = None) -> pd.DataFrame:
    """Ledger events with two flow columns: ``flow`` (default perp account only) and
    ``flow_wallet`` (perp + spot together, for unified accounts)."""
    if not ledger:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    addr = address.lower() if address else None
    rows = []
    for rec in ledger:
        delta = rec.get("delta") or {}
        kind = delta.get("type", "unknown")
        rows.append(
            {
                "time": int(rec["time"]),
                "type": kind,
                "flow": _perp_flow(delta, addr),
                "flow_wallet": _wallet_flow(delta, addr),
                "external": kind in EXTERNAL_TYPES,
                "account_value": _f(delta.get("accountValue")),
            }
        )
    df = pd.DataFrame(rows).sort_values("time", kind="mergesort")
    df["is_inflow"] = df["flow"] > 0
    df["is_outflow"] = df["flow"] < 0
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df[LEDGER_COLUMNS].reset_index(drop=True)


def use_wallet_flows(ledger: pd.DataFrame) -> pd.DataFrame:
    """Ledger view for the perp + spot balance model."""
    if ledger.empty:
        return ledger
    out = ledger.copy()
    out["flow"] = out["flow_wallet"]
    out["is_inflow"] = out["flow"] > 0
    out["is_outflow"] = out["flow"] < 0
    return out


def spot_cash_frame(fills: list[dict[str, Any]]) -> pd.DataFrame:
    """USDC cash effect of spot trades (buying spends USDC, selling receives it).

    Assumes spot pairs are quoted in USDC, which holds for most Hyperliquid spot pairs;
    the balance consistency check catches wallets where this is wrong.
    """
    rows = []
    for f in fills:
        if not is_spot_coin(str(f.get("coin", ""))):
            continue
        notional = _f(f.get("px")) * _f(f.get("sz"))
        if not np.isfinite(notional):
            continue
        cash = notional if f.get("side") == "A" else -notional
        if str(f.get("feeToken", "")).upper() == "USDC":
            fee = _f(f.get("fee", 0))
            cash -= 0.0 if np.isnan(fee) else fee
        rows.append({"time": int(f["time"]), "delta": cash})
    return pd.DataFrame(rows, columns=["time", "delta"])


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
