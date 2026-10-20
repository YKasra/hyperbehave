import numpy as np

from hyperbehave.normalize import fills_to_frame, ledger_to_frame, realised_balance_from_state
from hyperbehave.reconstruct import balance_curve, liquidation_times, round_trips, sessions
from tests.conftest import fill

MIN = 60_000
ADDR = "0x" + "a" * 40


def test_spot_fills_are_dropped():
    df = fills_to_frame([fill(1, "@107", "B", 1, 1, 0), fill(2, "PURR/USDC", "B", 1, 1, 0),
                         fill(3, "BTC", "B", 1, 1, 0)])
    assert df["coin"].tolist() == ["BTC"]


def test_round_trip_with_adverse_add_and_close():
    fills = [
        fill(0, "BTC", "B", 1, 100, 0, fee=0.1),              # open long 1 @ 100
        fill(10 * MIN, "BTC", "B", 1, 90, 1, fee=0.1),        # add while under water
        fill(30 * MIN, "BTC", "A", 2, 95, 2, pnl=0.0, fee=0.1),  # close flat at avg 95
    ]
    trips = round_trips(fills_to_frame(fills))
    assert len(trips) == 1
    t = trips.iloc[0]
    assert t.side == 1 and t.n_adds == 1 and t.n_adverse_adds == 1
    assert np.isclose(t.duration_min, 30)
    assert np.isclose(t.fees, 0.3)


def test_flip_closes_and_opens():
    fills = [
        fill(0, "ETH", "B", 2, 10, 0),
        fill(MIN, "ETH", "A", 5, 11, 2, pnl=2.0),   # long 2 -> short 3
        fill(2 * MIN, "ETH", "B", 3, 10, -3, pnl=3.0),
    ]
    trips = round_trips(fills_to_frame(fills))
    assert trips["side"].tolist() == [1, -1]
    assert trips["pnl"].tolist() == [2.0, 3.0]


def test_left_censored_position():
    fills = [fill(0, "SOL", "A", 4, 20, 4, pnl=-1.0)]  # window starts with an open long
    trips = round_trips(fills_to_frame(fills))
    assert len(trips) == 1 and bool(trips.iloc[0].left_censored)


def test_sessions_split_on_gap():
    fills = fills_to_frame([fill(t * MIN, "BTC", "B", 1, 1, 0, tid=t) for t in (0, 5, 10, 100, 101)])
    s = sessions(fills, gap_min=30)
    assert s["n_fills"].tolist() == [3, 2]


def test_ledger_flows_and_anchor():
    ledger = ledger_to_frame([
        {"time": 1, "delta": {"type": "deposit", "usdc": "100"}},
        {"time": 2, "delta": {"type": "withdraw", "usdc": "30", "fee": "1"}},
        {"time": 3, "delta": {"type": "internalTransfer", "usdc": "5", "destination": ADDR, "user": "0xb"}},
        {"time": 4, "delta": {"type": "liquidation", "accountValue": "12.5"}},
    ], ADDR)
    assert ledger["flow"].tolist() == [100.0, -30.0, 5.0, 0.0]
    fills = fills_to_frame([fill(5, "BTC", "A", 1, 1, 1, pnl=-10, fee=1)])
    fwd = balance_curve(fills, ledger, ledger.iloc[0:0])
    assert np.isclose(fwd["balance"].iloc[-1], 64.0)
    back = balance_curve(fills, ledger, ledger.iloc[0:0], anchor_balance=64.0)
    assert np.allclose(back["balance"], fwd["balance"])
    assert realised_balance_from_state({"marginSummary": {"accountValue": "110"},
                                        "assetPositions": [{"position": {"unrealizedPnl": "10"}}]}) == 100.0


def test_liquidation_events_are_merged():
    fills = fills_to_frame([fill(1000, "BTC", "A", 1, 1, 1, liq_user=ADDR)], ADDR)
    ledger = ledger_to_frame([{"time": 2000, "delta": {"type": "liquidation", "accountValue": "1"}}], ADDR)
    assert len(liquidation_times(fills, ledger)) == 1


def test_liquidation_of_counterparty_is_ignored():
    fills = fills_to_frame([fill(1000, "BTC", "B", 1, 1, 0, liq_user="0x" + "c" * 40)], ADDR)
    assert not fills["is_liquidation"].any()
