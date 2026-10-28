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


def _human_sweeper_fills(n_orders=1500):
    """A human placing a market order every ~3 minutes, each sweeping 3 resting orders."""
    from tests.conftest import fill

    rows, pos = [], 0.0
    for k in range(n_orders):
        t = k * 180_000 + (k % 7) * 1000
        side = "B" if pos <= 0 else "A"
        for j in range(3):
            f = fill(t, "BTC", side, 1, 100, pos, tid=k * 10 + j, d="Open Long" if side == "B" else "Close Long")
            f["oid"] = k
            rows.append(f)
            pos += 1 if side == "B" else -1
    return rows


def test_human_order_sweeps_not_flagged_as_bot(cfg):
    from hyperbehave.filters import bot_flags
    from hyperbehave.normalize import fills_to_frame

    flags = bot_flags(fills_to_frame(_human_sweeper_fills()), cfg)
    assert "machine-speed" not in flags["reason"]
    assert flags["median_order_gap_s"] > 100


def test_harm_uses_window_capital_and_detects_missing_flows():
    # Lifetime deposit of 10,000 two years ago; 1,000 of balance when the window opens;
    # a 500 loss inside the window. Loss must be measured against 1,000, not 10,000.
    bal = pd.DataFrame({"time": [0, 50, 100, 200], "delta": [10_000, -9_000, 0, -500],
                        "balance": [10_000, 1_000, 1_000, 500], "trading_pnl": [0, 0, 0, -500]})
    ledger = pd.DataFrame({"time": [0, 50], "flow": [10_000.0, -9_000.0]})
    out = mk.harm(bal, ledger, window_start=100)
    assert out["capital_at_risk"] == 1_000 and out["loss_to_capital"] == 0.5
    broken = bal.assign(balance=[10_000, 1_000, -2_000, -2_500])
    out = mk.harm(broken, ledger, window_start=100)
    assert out["balance_consistent"] == 0.0 and np.isnan(out["loss_to_capital"])


ADDR = "0x" + "a" * 40
OTHER = "0x" + "b" * 40


def _wallet(ledger, perp_value, spot_value, fills):
    end = 10_000_000
    return {"fills": fills, "ledger": ledger, "funding": [],
            "state": {"marginSummary": {"accountValue": str(perp_value)}, "assetPositions": []},
            "spot_state": {"balances": [{"coin": "USDC", "total": str(spot_value)}]},
            "meta": {"start_ms": 0, "end_ms": end, "fetched_ms": end}}


def _losing_trade(t0):
    from tests.conftest import fill

    return [fill(t0, "BTC", "B", 1, 100, 0, d="Open Long"),
            fill(t0 + 60_000, "BTC", "A", 1, 22, 1, pnl=-78.0, d="Close Long")]


def _send(t, user, dest, amount, src_dex, dst_dex):
    return {"time": t, "hash": f"0x{t}", "delta": {"type": "send", "user": user, "destination": dest,
            "token": "USDC", "amount": str(amount), "usdcValue": str(amount),
            "sourceDex": src_dex, "destinationDex": dst_dex}}


def test_standard_account_with_self_send_to_spot(cfg):
    # Deposit 5,000 to perp, lose 78, move 4,813 of own money from perp to spot.
    ledger = [{"time": 1_000, "delta": {"type": "deposit", "usdc": "5000"}},
              _send(5_000_000, ADDR, ADDR, 4813, "", "spot")]
    row = wallet_features(ADDR, _wallet(ledger, 109, 4813, _losing_trade(2_000_000)), cfg)
    assert row["balance_model"] == "perp+spot"
    assert np.isclose(row["capital_at_risk"], 5000) and np.isclose(row["loss_to_capital"], 78 / 5000)


def test_unified_account_spot_usdc_backs_perps(cfg):
    # Money arrives on the spot side; perp losses are paid from it; perp account shows 0.
    ledger = [_send(1_000, OTHER, ADDR, 5000, "", "spot")]
    row = wallet_features(ADDR, _wallet(ledger, 0, 4922, _losing_trade(2_000_000)), cfg)
    assert row["balance_model"] == "perp+spot"
    assert np.isclose(row["capital_at_risk"], 5000) and np.isclose(row["loss_to_capital"], 78 / 5000)


def test_send_between_two_users_perp_side():
    from hyperbehave.normalize import ledger_to_frame

    led = ledger_to_frame([_send(1, ADDR, OTHER, 40, "", ""), _send(2, ADDR, ADDR, 10, "", "spot")], ADDR)
    assert led["flow"].tolist() == [-40.0, -10.0]        # perp view: both leave perp
    assert led["flow_wallet"].tolist() == [-40.0, 0.0]   # wallet view: self move cancels
