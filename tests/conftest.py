import pytest

from hyperbehave import load_config, synthetic


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synth():
    wallets, candles = synthetic.generate(n_per_archetype=6, days=45, n_bots=2, seed=11)
    return wallets, candles


def fill(t, coin, side, sz, px, start, pnl=0.0, fee=0.0, crossed=True, tid=None, liq_user=None, d=""):
    f = {"time": t, "coin": coin, "side": side, "sz": str(sz), "px": str(px),
         "startPosition": str(start), "closedPnl": str(pnl), "fee": str(fee),
         "crossed": crossed, "tid": tid if tid is not None else t, "oid": t, "dir": d}
    if liq_user:
        f["liquidation"] = {"liquidatedUser": liq_user, "markPx": str(px), "method": "market"}
    return f
