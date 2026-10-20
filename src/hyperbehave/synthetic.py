"""Agent-based synthetic traders, written in the exact Hyperliquid API format.

Why this exists:

1. **Validation.** We know each synthetic trader's true behavioural rules, so we can
   check that every marker moves in the direction it should (a marker that cannot
   separate a programmed martingale gambler from a disciplined trader is broken).
2. **Testing and demos without real data.** The whole pipeline runs offline.
3. **Privacy.** The public repo and its figures can be built from synthetic wallets.

Archetypes
----------
``disciplined``  Trades in working hours of its own time zone, fixed small size, low
                 leverage, symmetric stop and target, mostly majors, withdraws profits.
``recreational`` More frequent, evenings included, moderate leverage, some alts.
``gambler``      Re-enters minutes after a loss and scales size up (martingale), adds to
                 losing positions, lets losers run but cuts winners early, high
                 leverage, mostly alts, trades late into the night and keeps chasing
                 through its usual rest hours after a loss, re-deposits after
                 liquidation, and sometimes takes a break before returning bigger.
``bot``          Market-maker style: tiny maker fills around the clock (should be
                 removed by the exclusion rules).

The rules are stylised, not a claim about how real people trade. Their only job is to
give each marker a known ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MS_MIN = 60_000
TAKER_FEE = 0.00045
MAKER_FEE = 0.00015

COINS = {"BTC": (60_000.0, 0.03), "ETH": (3_000.0, 0.04), "SOL": (150.0, 0.06),
         "DOGE": (0.15, 0.08), "WIF": (2.0, 0.11), "PEPE": (0.00001, 0.12)}
MAJORS = ["BTC", "ETH", "SOL"]
ALTS = ["DOGE", "WIF", "PEPE"]


@dataclass
class Profile:
    trades_per_day: float
    leverage: tuple[float, float]
    size_frac: float              # margin committed per trade, as a share of balance
    alt_share: float
    take_profit: float            # price move (fraction) that closes a winner
    stop_loss: float | None       # price move that closes a loser (None = let it run)
    max_hold_min: int
    wait_after_loss_min: float
    wait_after_gain_min: float
    martingale: float             # size multiplier after a loss (1 = none)
    add_prob: float               # probability of adding when the position is under water
    active_hours: tuple[int, int] | None  # local hours window, None = any time
    redeposit_prob: float
    relapse_prob: float
    withdraw_profits: bool
    chase_ignores_hours: bool = False  # after a loss, keep trading even in rest hours


PROFILES = {
    "disciplined": Profile(2.0, (1.5, 3.0), 0.10, 0.10, 0.02, 0.015, 24 * 60, 360, 360, 1.0,
                           0.0, (9, 19), 0.0, 0.0, True, False),
    "recreational": Profile(4.0, (3.0, 6.0), 0.15, 0.40, 0.025, 0.025, 12 * 60, 120, 150, 1.0,
                            0.1, (8, 24), 0.3, 0.1, False, False),
    "gambler": Profile(10.0, (10.0, 25.0), 0.25, 0.75, 0.012, None, 48 * 60, 8, 120, 1.8,
                       0.6, (13, 27), 0.85, 0.5, False, True),
}


def simulate_prices(start_ms: int, days: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Minute-level geometric random walks, one per coin, with occasional jumps."""
    n = days * 1440
    out = {}
    for coin, (p0, daily_vol) in COINS.items():
        sigma = daily_vol / np.sqrt(1440)
        shocks = rng.standard_t(4, size=n) * sigma / np.sqrt(2)
        jumps = rng.random(n) < 1 / 5000
        shocks[jumps] += rng.normal(0, 15 * sigma, size=jumps.sum())
        out[coin] = p0 * np.exp(np.cumsum(shocks))
    return out


def hourly_candles(prices: np.ndarray, start_ms: int) -> pd.DataFrame:
    """Hourly close candles (columns ``t`` and ``c``) for the market proxy."""
    hours = len(prices) // 60
    closes = prices[59: hours * 60: 60]
    return pd.DataFrame({"t": start_ms + np.arange(len(closes)) * 3_600_000, "c": closes})


def _fill(coin, px, sz, side, start_pos, closed_pnl, t, tid, crossed=True, liq_user=None):
    fee = px * sz * (TAKER_FEE if crossed else MAKER_FEE)
    long_ = (side == "B" and start_pos >= 0) or (side == "A" and start_pos > 0)
    opening = abs(start_pos + (sz if side == "B" else -sz)) > abs(start_pos)
    d = ("Open " if opening else "Close ") + ("Long" if long_ else "Short")
    f = {"coin": coin, "px": f"{px:.10g}", "sz": f"{sz:.10g}", "side": side, "time": int(t),
         "startPosition": f"{start_pos:.10g}", "dir": d, "closedPnl": f"{closed_pnl:.10g}",
         "hash": f"0x{tid:064x}", "oid": tid, "crossed": crossed, "fee": f"{fee:.10g}",
         "tid": tid, "feeToken": "USDC"}
    if liq_user:
        f["liquidation"] = {"liquidatedUser": liq_user, "markPx": f"{px:.10g}", "method": "market"}
    return f, fee


class _Trader:
    def __init__(self, address, archetype, prices, start_ms, days, rng):
        self.addr = address
        self.kind = archetype
        self.p = PROFILES[archetype]
        self.prices = prices
        self.start_ms = start_ms
        self.n_min = days * 1440
        self.rng = rng
        self.tz = int(rng.integers(-8, 9))
        self.lev = float(rng.uniform(*self.p.leverage))
        self.initial = float(rng.lognormal(np.log(2_000), 0.8))
        self.balance = 0.0
        self.fills: list[dict] = []
        self.ledger: list[dict] = []
        self.tid = int(rng.integers(1, 2**40))

    # helpers -------------------------------------------------------------------
    def _t(self, minute: int) -> int:
        return self.start_ms + minute * MS_MIN + int(self.rng.integers(0, MS_MIN))

    def _next_tid(self) -> int:
        self.tid += int(self.rng.integers(1, 1000))
        return self.tid

    def _deposit(self, minute: int, amount: float) -> None:
        self.balance += amount
        self.ledger.append({"time": self._t(minute), "hash": f"0x{self._next_tid():064x}",
                            "delta": {"type": "deposit", "usdc": f"{amount:.2f}"}})

    def _withdraw(self, minute: int, amount: float) -> None:
        self.balance -= amount
        self.ledger.append({"time": self._t(minute), "hash": f"0x{self._next_tid():064x}",
                            "delta": {"type": "withdraw", "usdc": f"{amount:.2f}", "fee": "1.0"}})

    def _in_hours(self, minute: int) -> bool:
        if self.p.active_hours is None:
            return True
        local = ((minute // 60) + self.tz) % 24
        lo, hi = self.p.active_hours  # hi may exceed 24 to wrap past midnight
        return lo <= local < hi or (hi > 24 and local < hi - 24)

    def _align(self, minute: int) -> int:
        """Move ``minute`` forward to the next minute inside the trader's active hours."""
        if self.p.active_hours is None:
            return minute
        for _ in range(48):
            if self._in_hours(minute):
                return minute + int(self.rng.integers(0, 30))
            minute += 60 - (minute % 60)
        return minute

    # one round trip --------------------------------------------------------------
    def _trade(self, minute: int, size_mult: float) -> tuple[int, float, bool]:
        """Simulate one position. Returns (exit minute, net pnl, liquidated)."""
        p, rng = self.p, self.rng
        coin = rng.choice(ALTS) if rng.random() < p.alt_share else rng.choice(MAJORS)
        path = self.prices[coin]
        side = 1 if rng.random() < 0.5 else -1
        margin = min(self.balance * p.size_frac * size_mult, self.balance * 0.95)
        if margin <= 1:
            return minute, 0.0, False
        entry = path[minute]
        qty = margin * self.lev / entry
        pos, avg, fees, added = qty, entry, 0.0, False
        f, fee = _fill(coin, entry, qty, "B" if side > 0 else "A", 0.0, 0.0, self._t(minute),
                       self._next_tid())
        self.fills.append(f)
        fees += fee
        liq_move = 0.9 / self.lev
        end = min(minute + p.max_hold_min, self.n_min - 1)
        exit_min, exit_px, liquidated = end, path[end], False
        for m in range(minute + 1, end + 1):
            move = side * (path[m] - avg) / avg
            if move <= -liq_move:
                exit_min, exit_px, liquidated = m, path[m], True
                break
            if move >= p.take_profit:
                exit_min, exit_px = m, path[m]
                break
            if p.stop_loss is not None and move <= -p.stop_loss:
                exit_min, exit_px = m, path[m]
                break
            if not added and move <= -0.4 * liq_move and rng.random() < p.add_prob / 30:
                add_qty = pos * 1.0
                if (pos + add_qty) * path[m] / self.lev < self.balance * 0.95:
                    f, fee = _fill(coin, path[m], add_qty, "B" if side > 0 else "A", side * pos, 0.0,
                                   self._t(m), self._next_tid())
                    self.fills.append(f)
                    fees += fee
                    avg = (avg * pos + path[m] * add_qty) / (pos + add_qty)
                    pos += add_qty
                    added = True
        pnl = side * (exit_px - avg) * pos
        if liquidated:
            pnl = max(pnl, -self.balance)
        f, fee = _fill(coin, exit_px, pos, "A" if side > 0 else "B", side * pos, pnl,
                       self._t(exit_min), self._next_tid(),
                       liq_user=self.addr if liquidated else None)
        self.fills.append(f)
        fees += fee
        if liquidated:
            self.ledger.append({"time": int(f["time"]), "hash": f["hash"],
                                "delta": {"type": "liquidation", "liquidatedNtlPos": f"{pos * exit_px:.2f}",
                                          "accountValue": f"{max(self.balance + pnl - fees, 0):.2f}",
                                          "leverageType": "Cross"}})
        net = pnl - fees
        self.balance += net
        return exit_min, net, liquidated

    # full life -----------------------------------------------------------------
    def run(self) -> None:
        p, rng = self.p, self.rng
        minute = int(rng.integers(0, 1440 * 3))
        self._deposit(minute, self.initial)
        minute = self._align(minute + 5)
        size_mult, last_withdraw = 1.0, minute
        while minute < self.n_min - p.max_hold_min:
            exit_min, net, liquidated = self._trade(minute, size_mult)
            if exit_min == minute:  # nothing left to trade with
                break
            loss = net < 0
            size_mult = min(size_mult * p.martingale, 8.0) if loss else 1.0
            wait = rng.exponential(p.wait_after_loss_min if loss else p.wait_after_gain_min)
            wait += rng.exponential(1440 / p.trades_per_day) * 0.3
            minute = exit_min + 1 + int(wait)

            ruined = self.balance < 0.1 * self.initial
            if liquidated or ruined:
                if ruined and rng.random() < p.relapse_prob:
                    minute += int(rng.uniform(15, 25) * 1440)  # a break, then a return
                    size_mult = 1.5
                    if minute < self.n_min:
                        self._deposit(minute, self.initial * rng.uniform(0.8, 1.6))
                elif rng.random() < p.redeposit_prob:
                    delay = int(rng.exponential(180))
                    self._deposit(exit_min + delay, self.initial * rng.uniform(0.4, 1.2))
                    minute = max(minute, exit_min + delay + 1)
                elif ruined:
                    break
            if p.withdraw_profits and minute - last_withdraw > 14 * 1440 and self.balance > 1.2 * self.initial:
                self._withdraw(minute, 0.5 * (self.balance - self.initial))
                last_withdraw = minute
            if not (loss and p.chase_ignores_hours):
                minute = self._align(minute)

    def data(self, end_ms: int) -> dict[str, Any]:
        self.fills.sort(key=lambda f: (f["time"], f["tid"]))
        self.ledger.sort(key=lambda r: r["time"])
        return {
            "fills": self.fills,
            "ledger": self.ledger,
            "funding": [],
            "state": {"marginSummary": {"accountValue": f"{self.balance:.6f}"}, "assetPositions": []},
            "meta": {"start_ms": self.start_ms, "end_ms": end_ms, "fetched_ms": end_ms,
                     "synthetic_archetype": self.kind},
        }


def _bot(address: str, prices: dict[str, np.ndarray], start_ms: int, rng: np.random.Generator,
         days: int = 10) -> dict[str, Any]:
    path = prices["BTC"]
    fills, t, pos, tid = [], start_ms, 0.0, int(rng.integers(1, 2**40))
    end = start_ms + days * 86_400_000
    q = 0.001
    while t < end:
        minute = (t - start_ms) // MS_MIN
        px = path[min(minute, len(path) - 1)] * (1 + rng.normal(0, 1e-4))
        side = "B" if pos <= 0 else "A"
        pnl = 0.0 if side == "B" else q * px * 2e-4
        tid += 1
        f, _ = _fill("BTC", px, q, side, pos, pnl, t, tid, crossed=rng.random() < 0.03)
        fills.append(f)
        pos += q if side == "B" else -q
        t += int(rng.exponential(12_000)) + 500
    ledger = [{"time": start_ms - 60_000, "hash": "0x0", "delta": {"type": "deposit", "usdc": "50000"}}]
    return {"fills": fills, "ledger": ledger, "funding": [], "state": {},
            "meta": {"start_ms": start_ms, "end_ms": end, "fetched_ms": end + 10 * 3_600_000,
                     "synthetic_archetype": "bot"}}


def generate(
    n_per_archetype: int = 30,
    days: int = 90,
    n_bots: int = 5,
    seed: int = 7,
    start_ms: int = 1_767_225_600_000,  # 2026-01-01 00:00 UTC
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Return ``({address: raw_wallet_data}, btc_hourly_candles)``."""
    rng = np.random.default_rng(seed)
    prices = simulate_prices(start_ms, days, rng)
    end_ms = start_ms + days * 86_400_000
    wallets: dict[str, dict[str, Any]] = {}
    k = 0
    for kind in PROFILES:
        for _ in range(n_per_archetype):
            k += 1
            addr = f"0x{k:040x}"
            trader = _Trader(addr, kind, prices, start_ms, days, rng)
            trader.run()
            wallets[addr] = trader.data(end_ms)
    for _ in range(n_bots):
        k += 1
        addr = f"0x{k:040x}"
        wallets[addr] = _bot(addr, prices, start_ms, rng)
    return wallets, hourly_candles(prices["BTC"], start_ms)
