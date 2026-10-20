# Data sources

All data used here is public. Nothing requires an exchange agreement or an API key.

## Free path (default)

| Step | Source | What it gives | Limits |
| --- | --- | --- | --- |
| 1. Find wallets | Websocket `trades` channel (`wss://api.hyperliquid.xyz/ws`) | The two counterparties (`users`) of every trade | Samples wallets in proportion to how often they trade during the listening window |
| 2. Fills | `POST /info` `userFillsByTime` | Coin, price, size, side, `dir`, `startPosition`, `closedPnl`, fee, taker flag, liquidation info | At most 2,000 fills per call; only the 10,000 most recent fills per wallet |
| 3. Money in and out | `userNonFundingLedgerUpdates` | Deposits, withdrawals, transfers, vault flows, liquidations with account value | Time-windowed; fetched from the wallet's first event so the balance can be rebuilt |
| 4. Funding | `userFunding` | Funding payments per coin | Time-windowed |
| 5. Account snapshot | `clearinghouseState` | Current account value and open positions | Snapshot at fetch time only; used to anchor the balance curve |
| 6. Market context | `candleSnapshot` | Hourly candles for the market-shock marker | At most 5,000 candles per call |

Requests are throttled (`api.min_interval_s`, default 1 s) and retried with backoff on
HTTP 429 and 5xx. At three to four calls per wallet, 1,000 wallets take roughly an hour.

## Paid path (optional): S3 node archive

`s3://hl-mainnet-node-data/node_fills_by_block/` holds every fill on the exchange from
27 July 2025 onward, in the same format as the API. The bucket is **requester pays**:
the data is free, but whoever downloads it pays AWS transfer, around USD 0.09 per GB.
Community estimates put fills at under 1 GB per day compressed, so one month costs a few
dollars. Always run `hyperbehave s3-size --start ... --end ...` first and set an AWS
budget alert.

Use the archive when you need complete histories for wallets above the 10,000-fill API
cap, or the full population instead of a sample.

## References

- Hyperliquid docs, historical data: https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data
- Hyperliquid docs, info endpoint: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint
