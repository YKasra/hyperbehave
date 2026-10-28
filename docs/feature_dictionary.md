# Feature dictionary

Every marker is a computable definition tied to a DSM-5 gambling-disorder criterion or a
known behavioural-finance bias. The single source of truth is `MARKERS` in
`src/hyperbehave/markers.py`; this page is generated from it (`python scripts/make_feature_dictionary.py`).

Thresholds (minutes, hours, days) are set in `src/hyperbehave/default_config.yaml` and are
pre-registration candidates: keep defaults for the main analysis, vary them in sensitivity checks.

Core markers (used for the multi-marker high-risk screen) are marked with **core**.

| Marker | Criterion | Definition | Higher means |
| --- | --- | --- | --- |
| `loss_chasing_index` **core** | Chasing losses | P(new position within N min after a losing trip) minus the same after a winning trip | more gambling-like |
| `post_loss_size_ratio` **core** | Chasing losses | Median size of the next position after a loss / after a gain | more gambling-like |
| `streak_escalation_ratio` | Chasing losses, tolerance | Size of the position after a losing streak / size at the start of the streak | more gambling-like |
| `redeposit_after_liq_share` **core** | Chasing losses, loss of control | Share of liquidations followed by new money within N hours | more gambling-like |
| `size_trend_weekly` **core** | Tolerance | Weekly slope of log median position size | more gambling-like |
| `leverage_trend_weekly` | Tolerance | Weekly slope of log median effective leverage | more gambling-like |
| `median_eff_leverage` | Tolerance, risk taking | Median open notional / realised balance | more gambling-like |
| `sessions_per_active_day` **core** | Preoccupation | Trading sessions per active day | more gambling-like |
| `max_session_min` | Preoccupation, loss of control | Longest continuous session in minutes | more gambling-like |
| `rest_window_share` **core** | Loss of control | Share of opening trades inside the wallet's own quietest 5 h daily window (sleep proxy) | more gambling-like |
| `hour_week_entropy` | Preoccupation | Normalised entropy of activity over the 168 hours of the week | more gambling-like |
| `session_end_loss_streak` | Loss of control | Mean losing streak at which sessions end (stops only when forced) | more gambling-like |
| `doubling_down_share` **core** | Chasing losses | Share of positions that were increased while under water | more gambling-like |
| `disposition_ratio` | Behavioural bias (disposition effect) | Median holding time of losing trips / winning trips | more gambling-like |
| `house_money_ratio` | Behavioural bias (house money) | Median size of the next position after a gain / median position size | more gambling-like |
| `volatility_reactivity` | Impulsivity | New positions per hour after market shocks / baseline rate | more gambling-like |
| `relapse_count` | Unsuccessful efforts to stop | Breaks of N days or more followed by a return | more gambling-like |
| `relapse_size_ratio` | Unsuccessful efforts to stop, tolerance | Position size after returning from a break / before it | more gambling-like |
| `lottery_share` | Gambling-like asset choice | Share of traded notional outside the major coins | more gambling-like |
| `loss_to_capital` **core** | Harm, jeopardised finances | Net trading loss in the window / capital at risk (balance at window start + money added; 0 when profitable) | more gambling-like |
| `max_drawdown_rel` | Harm, jeopardised finances | Max drawdown of cumulative trading PnL in the window / capital at risk | more gambling-like |

## Activity covariates

Not markers, but needed as controls and for the frequency-only baseline:
`n_fills`, `n_orders`, `orders_per_day`, `n_trips`, `active_days`, `span_days`, `fills_per_active_day`,
`total_volume`, `median_trip_notional`, `win_rate`, `maker_share`, `n_liquidations`,
`median_redeposit_delay_h`, `net_trading_pnl`, `start_balance`, `window_inflow`, `window_outflow`,
`capital_at_risk`, `withdraw_deposit_ratio`, `max_drawdown_usd`, `median_session_min`, `no_rest`,
`balance_anchored`, `balance_consistent`.

## Units of analysis

- **Fill**: one execution reported by the exchange.
- **Round trip**: one position in one coin, from leaving zero until flat again (or a flip). This is the unit of a bet.
- **Order**: all fills sharing one order id; one market order sweeping several price levels is one order.
- **Active order**: a taker order that is not a liquidation, i.e. a decision taken at that moment. Sessions, the rest window and weekly entropy use active orders only.
- **Session**: active orders separated by less than `sessions.gap_min` minutes (default 30).
- **Balance curve**: realised balance after each cash event; rebuilt backwards from the live account value when the data runs up to the fetch time (`balance_anchored = True`), otherwise forwards from zero.

## Known limitations

- Time zone is unknown for public wallets, so night trading is measured against each wallet's own quietest 5 h window (a sleep proxy), not against clock time.
- One person can run several wallets and one wallet can be shared; Layer A treats each wallet as a unit.
- Positions open before the window start are flagged `left_censored` and excluded from holding-time markers.
- Effective leverage uses realised balance and ignores unrealised PnL between fills.
- Spot trades are excluded; the study covers perpetual futures only.
- Stop-loss and take-profit triggers look like manual taker orders in public data.
- Capital at risk is only as good as the public ledger. When the rebuilt balance goes clearly negative (`balance_consistent = 0`), money flows are missing and balance-based markers (leverage, harm) are set to missing for that wallet.
