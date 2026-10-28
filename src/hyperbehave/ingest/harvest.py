"""Collect active wallet addresses for free from the public trades stream.

Every message on the Hyperliquid ``trades`` websocket channel lists the two counter-
parties of each trade in a ``users`` field ``[buyer, seller]``. Listening for a while
across a set of coins yields a sample of active wallets without any paid data.

The harvester prints progress every ``progress_s`` seconds, writes the CSV every
``save_s`` seconds, reconnects automatically if the connection drops, and saves on
Ctrl+C, so a long run never loses what it already collected.

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
from collections.abc import Callable, Iterable
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


def load_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                counts[row["address"]] += int(row["n_trades_seen"])
    return counts


def save_counts(counts: Counter, path: Path) -> None:
    """Write atomically (temp file, then rename) so a crash never leaves a broken CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["address", "n_trades_seen"])
        for addr, n in counts.most_common():
            writer.writerow([addr, n])
    tmp.replace(path)


async def _listen(
    coins: list[str],
    seconds: float,
    ws_url: str,
    counts: Counter,
    on_tick: Callable[[], None],
    tick_s: float,
    log: Callable[[str], None],
) -> None:
    import websockets  # optional dependency: pip install hyperbehave[harvest]

    deadline = time.monotonic() + seconds
    next_tick = time.monotonic() + tick_s
    backoff = 1.0
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(ws_url, ping_interval=20, open_timeout=20) as ws:
                for coin in coins:
                    await ws.send(json.dumps(
                        {"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}
                    ))
                log(f"connected, listening to {', '.join(coins)}")
                backoff = 1.0
                last_ping = time.monotonic()
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    if now - last_ping > 30:  # application-level heartbeat
                        await ws.send(json.dumps({"method": "ping"}))
                        last_ping = now
                    if now >= next_tick:
                        on_tick()
                        next_tick = now + tick_s
                    timeout = max(0.1, min(deadline, next_tick) - time.monotonic())
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        counts.update(extract_users(json.loads(raw)))
                    except (json.JSONDecodeError, AttributeError):
                        continue
        except (OSError, asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            if time.monotonic() >= deadline:
                break
            log(f"connection lost ({type(exc).__name__}); reconnecting in {backoff:.0f}s")
            on_tick()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def harvest_addresses(
    coins: Iterable[str],
    minutes: float,
    out_csv: str | Path,
    ws_url: str = DEFAULT_WS,
    progress_s: float = 30,
    log: Callable[[str], None] = print,
) -> int:
    """Listen for ``minutes`` and keep ``address,n_trades_seen`` up to date in ``out_csv``.

    Existing rows are kept and counts added up, so the command can be run several times
    (for example once a day for a week) to build a broader sample.
    """
    out_csv = Path(out_csv)
    counts = load_counts(out_csv)
    start_n = len(counts)
    started = time.monotonic()
    total = minutes * 60

    def tick() -> None:
        save_counts(counts, out_csv)
        elapsed = time.monotonic() - started
        left = max(total - elapsed, 0)
        log(f"[{elapsed / 60:5.1f} min] {len(counts):,} wallets "
            f"(+{len(counts) - start_n:,} new), {left / 60:.1f} min left, saved to {out_csv}")

    try:
        asyncio.run(_listen(list(coins), total, ws_url, counts, tick, progress_s, log))
    except KeyboardInterrupt:
        log("stopped by user")
    finally:
        save_counts(counts, out_csv)
    log(f"done: {len(counts):,} wallets in {out_csv}")
    return len(counts)
