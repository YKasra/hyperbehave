"""Collect active wallet addresses for free from the public trades stream.

Every message on the Hyperliquid ``trades`` websocket channel lists the two counter-
parties of each trade in a ``users`` field ``[buyer, seller]``. Listening for a while
across a set of coins yields a sample of active wallets without any paid data.

Sampling note for the paper: this samples wallets proportional to how often they trade
during the listening window, so very active wallets are over-represented. Report the
window and coins, and treat Layer A prevalence figures as "among wallets active in
the window".
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

DEFAULT_WS = "wss://api.hyperliquid.xyz/ws"


def extract_users(message: dict) -> list[str]:
    """Return the addresses found in one websocket ``trades`` message."""
    if message.get("channel") != "trades":
        return []
    users: list[str] = []
    for trade in message.get("data") or []:
        for addr in trade.get("users") or []:
            if isinstance(addr, str) and addr.startswith("0x") and len(addr) == 42:
                users.append(addr.lower())
    return users


async def _listen(coins: Iterable[str], seconds: float, ws_url: str) -> Counter:
    import websockets  # optional dependency: pip install hyperbehave[harvest]

    counts: Counter = Counter()
    deadline = time.monotonic() + seconds
    async with websockets.connect(ws_url, ping_interval=20) as ws:
        for coin in coins:
            await ws.send(
                json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}})
            )
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                break
            try:
                counts.update(extract_users(json.loads(raw)))
            except json.JSONDecodeError:
                continue
    return counts


def harvest_addresses(
    coins: Iterable[str], minutes: float, out_csv: str | Path, ws_url: str = DEFAULT_WS
) -> int:
    """Listen for ``minutes`` and write ``address,n_trades_seen`` to ``out_csv``.

    Existing rows in ``out_csv`` are kept and their counts added up, so the command can be
    run several times (for example once a day for a week) to build a broader sample.
    """
    counts = asyncio.run(_listen(list(coins), minutes * 60, ws_url))
    out_csv = Path(out_csv)
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                counts[row["address"]] += int(row["n_trades_seen"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["address", "n_trades_seen"])
        for addr, n in counts.most_common():
            writer.writerow([addr, n])
    return len(counts)
