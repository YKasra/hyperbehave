import numpy as np

from hyperbehave import markers as mk
from hyperbehave.features import build_feature_table
from hyperbehave.paper import bh, paper_report, wilson


def test_wilson_and_bh():
    lo, hi = wilson(5, 100)
    assert 0.01 < lo < 0.05 < hi < 0.12
    q = bh(np.array([0.01, 0.04, np.nan, 0.03]))
    assert np.isnan(q[2]) and np.allclose(q[[0, 3, 1]], [0.03, 0.04, 0.04])


def test_paper_report_runs_on_synthetic(synth, cfg, tmp_path):
    wallets, candles = synth
    ft = build_feature_table(wallets.items(), cfg, "salt", mk.market_shocks(candles))
    assert {"loss_chasing_p", "loss_chasing_index_h0", "lottery_share_h1"} <= set(ft.columns)
    path = paper_report(ft, tmp_path, n_perm=50)
    text = path.read_text(encoding="utf-8")
    assert "How common is each behaviour" in text and "reliable" in text
    for f in ("fig1_prevalence.png", "fig2_reliability.png", "fig3_cooccurrence.png",
              "fig4_cooccurrence_null.png", "fig5_leverage_drawdown.png", "table_prevalence.csv"):
        assert (tmp_path / f).exists(), f
    assert "wallet_id" not in text


def test_loss_chasing_permutation_detects_chaser():
    import pandas as pd

    rows, t = [], 0
    for k in range(40):
        loss = k % 2 == 0
        rows.append([t, t + 600_000, -1.0 if loss else 1.0, 100.0, 10, 0, 0, False])
        t += 600_000 + (300_000 if loss else 18_000_000)
    trips = pd.DataFrame(rows, columns=["open_time", "close_time", "net_pnl", "entry_notional",
                                        "duration_min", "n_adds", "n_adverse_adds", "left_censored"])
    out = mk.loss_chasing(trips, n_perm=200)
    assert out["loss_chasing_p"] < 0.01
