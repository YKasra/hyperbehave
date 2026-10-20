"""Polite, paginated client for the public Hyperliquid info endpoint.

The endpoint is ``POST https://api.hyperliquid.xyz/info`` and needs no API key.
Known constraints this client handles:

* ``userFillsByTime`` returns at most 2,000 fills per call, and only the 10,000 most
  recent fills of a wallet are retrievable through the API at all.
* Ledger and funding endpoints are time-windowed and also paginate by time.
* The service is rate limited per IP, so every request waits ``min_interval_s`` and
  retries with exponential backoff on HTTP 429 and 5xx.

Pagination strategy: request ``[start, end]``, take the latest timestamp returned and
use it (inclusive) as the next ``startTime``. Records are de-duplicated on a stable key,
and the loop stops as soon as a page adds nothing new. Using the inclusive timestamp
avoids losing several fills that share the boundary millisecond.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_URL = "https://api.hyperliquid.xyz/info"

Record = dict[str, Any]


class HyperliquidAPIError(RuntimeError):
    """Raised when the API keeps failing after all retries."""


@dataclass
class HyperliquidInfoClient:
    base_url: str = DEFAULT_URL
    min_interval_s: float = 1.0
    max_retries: int = 5
    timeout_s: float = 30.0
    page_limit_fills: int = 2000
    # Ledger and funding page sizes are not documented, so by default those loops stop
    # only when a page brings nothing new (one extra request, no risk of truncation).
    page_limit_events: int | None = None
    session: requests.Session = field(default_factory=requests.Session)
    sleep: Callable[[float], None] = time.sleep
    _last_call: float = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------ low level
    def post(self, payload: dict[str, Any]) -> Any:
        """POST one payload to the info endpoint, with throttling and retries."""
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                self.sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self.session.post(self.base_url, json=payload, timeout=self.timeout_s)
            except requests.RequestException as exc:  # network hiccup
                if attempt == self.max_retries:
                    raise HyperliquidAPIError(f"network error: {exc}") from exc
                self.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                self.sleep(2**attempt)
                continue
            raise HyperliquidAPIError(
                f"HTTP {resp.status_code} for {payload.get('type')}: {resp.text[:200]}"
            )
        raise HyperliquidAPIError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ pagination
    def _paginate(
        self,
        build_payload: Callable[[int, int], dict[str, Any]],
        start_ms: int,
        end_ms: int,
        key: Callable[[Record], Any],
        page_limit: int | None = 2000,
        max_pages: int = 10_000,
    ) -> list[Record]:
        seen: dict[Any, Record] = {}
        cursor = start_ms
        for _ in range(max_pages):
            page = self.post(build_payload(cursor, end_ms)) or []
            new = 0
            for rec in page:
                k = key(rec)
                if k not in seen:
                    seen[k] = rec
                    new += 1
            if not page or new == 0:
                break
            if page_limit is not None and len(page) < page_limit:
                break  # a short page means the window is exhausted
            latest = max(int(r["time"]) for r in page)
            if latest >= end_ms:
                break
            # Re-request from the boundary millisecond (inclusive) so fills sharing it are
            # not lost; if a whole page sits in one millisecond, step past it instead.
            cursor = latest if latest > cursor else cursor + 1
        return sorted(seen.values(), key=lambda r: int(r["time"]))

    # ------------------------------------------------------------------ endpoints
    def user_fills(self, user: str, start_ms: int, end_ms: int) -> list[Record]:
        """All fills of ``user`` in ``[start_ms, end_ms]`` (subject to the 10k API cap)."""
        return self._paginate(
            lambda s, e: {
                "type": "userFillsByTime",
                "user": user,
                "startTime": s,
                "endTime": e,
                "aggregateByTime": False,
            },
            start_ms,
            end_ms,
            key=lambda r: (r.get("tid"), r.get("oid"), r.get("time"), r.get("sz")),
            page_limit=self.page_limit_fills,
        )

    def user_ledger(self, user: str, start_ms: int, end_ms: int) -> list[Record]:
        """Deposits, withdrawals, transfers and liquidations (no funding)."""
        return self._paginate(
            lambda s, e: {
                "type": "userNonFundingLedgerUpdates",
                "user": user,
                "startTime": s,
                "endTime": e,
            },
            start_ms,
            end_ms,
            key=lambda r: (r.get("hash"), r.get("time"), (r.get("delta") or {}).get("type")),
            page_limit=self.page_limit_events,
        )

    def user_funding(self, user: str, start_ms: int, end_ms: int) -> list[Record]:
        """Funding payments of ``user``."""
        return self._paginate(
            lambda s, e: {"type": "userFunding", "user": user, "startTime": s, "endTime": e},
            start_ms,
            end_ms,
            key=lambda r: (
                r.get("hash"),
                r.get("time"),
                (r.get("delta") or {}).get("coin"),
            ),
            page_limit=self.page_limit_events,
        )

    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list[Record]:
        """OHLCV candles (the API returns at most 5,000 per call, so we paginate)."""
        out: dict[int, Record] = {}
        cursor = start_ms
        for _ in range(1000):
            page = self.post(
                {
                    "type": "candleSnapshot",
                    "req": {"coin": coin, "interval": interval, "startTime": cursor, "endTime": end_ms},
                }
            ) or []
            new = [c for c in page if int(c["t"]) not in out]
            for c in new:
                out[int(c["t"])] = c
            if not new:
                break
            cursor = max(int(c["t"]) for c in page) + 1
            if cursor >= end_ms:
                break
        return [out[k] for k in sorted(out)]

    def clearinghouse_state(self, user: str) -> Record:
        """Current perp account snapshot (account value, open positions)."""
        return self.post({"type": "clearinghouseState", "user": user}) or {}

    def fetch_wallet(
        self, user: str, start_ms: int, end_ms: int, ledger_start_ms: int = 0
    ) -> dict[str, Any]:
        """Fills, ledger, funding and current state for one wallet.

        The ledger is fetched from ``ledger_start_ms`` (default: the beginning) because
        deposit history is short and is needed to reconstruct the balance.
        """
        return {
            "fills": self.user_fills(user, start_ms, end_ms),
            "ledger": self.user_ledger(user, ledger_start_ms, end_ms),
            "funding": self.user_funding(user, start_ms, end_ms),
            "state": self.clearinghouse_state(user),
            "meta": {"start_ms": start_ms, "end_ms": end_ms, "fetched_ms": int(time.time() * 1000)},
        }


def iter_wallets(
    client: HyperliquidInfoClient,
    users: Iterable[str],
    start_ms: int,
    end_ms: int,
    on_error: Callable[[str, Exception], None] | None = None,
):
    """Yield ``(user, data)`` for every wallet, skipping wallets that keep failing."""
    for user in users:
        try:
            yield user, client.fetch_wallet(user, start_ms, end_ms)
        except HyperliquidAPIError as exc:
            if on_error:
                on_error(user, exc)
