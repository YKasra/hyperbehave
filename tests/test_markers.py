import numpy as np
import pandas as pd

from hyperbehave import analysis as an
from hyperbehave import markers as mk
from hyperbehave.features import build_feature_table, wallet_features

MIN = 60_000


def _trips(rows):
    return pd.DataFrame(rows, columns=["open_time", "close_time", "net_pnl", "entry_notional",
                                       "duration_min", "n_adds", "n_adverse_adds", "left_censored"])


def test_loss_chasing_detects_fast_reentry_after_losses():
    rows, t = [], 0
    for k in range(20):
        loss = k % 2 == 0
        rows.append([t, t + 10 * MIN, -1.0 if loss else 1.0, 200.0 if not loss else 100.0, 10, 0, 0, False])
        t += 10 * MIN + (5 * MIN if loss else 300 * MIN)
    out = mk.loss_chasing(_trips(rows), window_min=60, min_events=5)
    assert out["loss_chasing_index"] > 0.9
    assert out["post_loss_size_ratio"] > 1.5


def test_loss_chasing_needs_enough_events():
    rows = [[0, MIN, -1.0, 1.0, 1, 0, 0, False], [2 * MIN, 3 * MIN, 1.0, 1.0, 1, 0, 0, False]]
    assert np.isnan(mk.loss_chasing(_trips(rows))["loss_chasing_index"])


def test_redeposit_after_liquidation():
    ledger = pd.DataFrame({"time": [3_600_000 * 5], "is_inflow": [True]})
    out = mk.redeposit_after_liquidation(np.array([0, 3_600_000 * 100]), ledger, window_h=24)
    assert out["redeposit_after_liq_share"] == 0.5
    assert out["median_redeposit_delay_h"] == 5.0


def test_rest_window_uses_opening_trades():
    hours = np.r_[np.repeat(np.arange(9, 18), 5)]
    fills = pd.DataFrame({"time": hours * 3_600_000, "dir": "Open Long"})
    out = mk.time_of_day(fills)
    assert out["rest_window_share"] == 0.0 and out["no_rest"] == 0.0


def test_auc():
    assert an.auc(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1])) == 1.0
    assert an.auc(np.array([1, 1]), np.array([0, 1])) == 0.5


def test_bots_excluded_and_humans_kept(synth, cfg):
    wallets, candles = synth
    extra = {a: {"archetype": d["meta"]["synthetic_archetype"]} for a, d in wallets.items()}
    ft = build_feature_table(wallets.items(), cfg, "salt", mk.market_shocks(candles), extra=extra)
    assert ft.loc[ft["archetype"] == "bot", "exclude"].all()
    assert not ft.loc[ft["archetype"] != "bot", "exclude"].any()
    assert "address" not in ft.columns and ft["wallet_id"].str.len().eq(16).all()


def test_markers_separate_programmed_gambler(synth, cfg):
    wallets, candles = synth
    extra = {a: {"archetype": d["meta"]["synthetic_archetype"]} for a, d in wallets.items()}
    ft = an.included(build_feature_table(wallets.items(), cfg, "salt", mk.market_shocks(candles), extra=extra))
    for marker in ("loss_chasing_index", "doubling_down_share", "median_eff_leverage",
                   "lottery_share", "disposition_ratio", "rest_window_share"):
        table = an.marker_auc_table(ft, "archetype", "gambler", [marker])
        assert table["auc"].iloc[0] > 0.8, marker


def test_empty_wallet_does_not_crash(cfg):
    row = wallet_features("0x" + "0" * 40, {"fills": [], "ledger": [], "funding": []}, cfg)
    assert row["exclude"]
