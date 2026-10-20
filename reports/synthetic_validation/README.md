# Synthetic validation

Wallets: 95 (90 human archetypes, 5 bots).
Bots removed by the exclusion rules: 5 of 5.
Human wallets wrongly excluded: 0.

## Can each marker tell the programmed gambler apart?

AUC = probability that a random gambler scores higher than a random non-gambler (0.5 = no separation, 1.0 = perfect). Coverage = share of wallets where the marker could be computed. A note marks markers whose simulated direction comes from an assumption, not from a programmed gambling rule: only real labelled data can test those.

| marker | auc | coverage | note |
| --- | --- | --- | --- |
| disposition_ratio | 1.000 | 1.000 |  |
| rest_window_share | 1.000 | 1.000 |  |
| median_eff_leverage | 1.000 | 1.000 |  |
| lottery_share | 1.000 | 1.000 |  |
| max_drawdown_rel | 1.000 | 1.000 |  |
| doubling_down_share | 1.000 | 1.000 |  |
| loss_chasing_index | 0.998 | 1.000 |  |
| hour_week_entropy | 0.984 | 1.000 |  |
| sessions_per_active_day | 0.977 | 1.000 |  |
| max_session_min | 0.954 | 1.000 |  |
| post_loss_size_ratio | 0.933 | 1.000 |  |
| loss_to_inflow | 0.895 | 1.000 |  |
| size_trend_weekly | 0.812 | 1.000 |  |
| volatility_reactivity | 0.601 | 1.000 | no reaction to market shocks was programmed |
| relapse_count | 0.600 | 1.000 | breaks happen only after rare full ruin |
| leverage_trend_weekly | 0.500 | 1.000 | leverage is fixed per simulated trader |
| streak_escalation_ratio | 0.398 | 0.833 | simulated losing streaks mostly end in ruin and a smaller fresh deposit, which caps the next size |
| house_money_ratio | 0.066 | 1.000 | simulated gamblers reset size after a win by design |
| session_end_loss_streak | 0.000 | 1.000 | simulated gamblers pause after wins by design, so they rarely end a session on losses |
| redeposit_after_liq_share |  | 0.333 |  |
| relapse_size_ratio |  | 0.067 | breaks happen only after rare full ruin |

## Medians by archetype

| marker | disciplined | recreational | gambler |
| --- | --- | --- | --- |
| loss_chasing_index | 0.000 | 0.012 | 0.204 |
| post_loss_size_ratio | 0.996 | 0.984 | 1.362 |
| streak_escalation_ratio | 0.985 | 0.942 | 0.827 |
| redeposit_after_liq_share |  |  | 0.905 |
| size_trend_weekly | -0.003 | -0.013 | 0.052 |
| leverage_trend_weekly | -0.000 | -0.000 | 0.002 |
| median_eff_leverage | 0.219 | 0.642 | 4.817 |
| sessions_per_active_day | 2.444 | 4.344 | 5.667 |
| max_session_min | 27.914 | 29.329 | 53.963 |
| rest_window_share | 0.000 | 0.000 | 0.030 |
| hour_week_entropy | 0.609 | 0.804 | 0.876 |
| session_end_loss_streak | 0.572 | 0.512 | 0.169 |
| doubling_down_share | 0.000 | 0.000 | 0.242 |
| disposition_ratio | 0.746 | 1.000 | 3.990 |
| house_money_ratio | 1.003 | 1.008 | 0.948 |
| volatility_reactivity | 0.900 | 0.962 | 1.143 |
| relapse_count | 0.000 | 0.000 | 0.000 |
| relapse_size_ratio |  |  | 0.529 |
| lottery_share | 0.099 | 0.390 | 0.739 |
| loss_to_inflow | 0.028 | 0.172 | 0.858 |
| max_drawdown_rel | 0.064 | 0.316 | 1.113 |

![core markers](synthetic_core_markers.png)
