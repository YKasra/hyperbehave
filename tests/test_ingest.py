import json

from hyperbehave.ingest.api_client import HyperliquidInfoClient
from hyperbehave.ingest.harvest import extract_users
from hyperbehave.ingest.s3_archive import parse_fill_lines
from hyperbehave.storage import load_wallet, save_wallet, wallet_id


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, ""

    def json(self):
        return self._payload


class FakeSession:
    """Serves fills in pages of ``page`` records, like the real 2,000-fill cap."""

    def __init__(self, fills, page=3, fail_first=False):
        self.fills, self.page, self.calls, self.fail_first = fills, page, [], fail_first

    def post(self, url, json, timeout):
        self.calls.append(json)
        if self.fail_first and len(self.calls) == 1:
            return FakeResponse(None, 429)
        if json["type"] == "userFillsByTime":
            sel = [f for f in self.fills if json["startTime"] <= f["time"] <= json["endTime"]]
            return FakeResponse(sel[: self.page])
        return FakeResponse([])


def _client(session, page=3):
    return HyperliquidInfoClient(min_interval_s=0, session=session, sleep=lambda s: None,
                                 page_limit_fills=page, page_limit_events=page)


def test_pagination_collects_every_fill_including_shared_timestamps():
    times = [1, 2, 2, 2, 3, 4, 5, 6, 7]
    fills = [{"time": t, "tid": i, "oid": i, "sz": "1"} for i, t in enumerate(times)]
    session = FakeSession(fills, page=3)
    got = _client(session).user_fills("0xabc", 0, 100)
    assert sorted(f["tid"] for f in got) == list(range(len(fills)))
    assert len(session.calls) < 10


def test_retry_on_rate_limit():
    session = FakeSession([{"time": 1, "tid": 1, "oid": 1, "sz": "1"}], fail_first=True)
    assert len(_client(session).user_fills("0xabc", 0, 10)) == 1


def test_extract_users_from_trades_message():
    a, b = "0x" + "1" * 40, "0x" + "2" * 40
    msg = {"channel": "trades", "data": [{"coin": "BTC", "users": [a, b]}, {"users": ["bad"]}]}
    assert extract_users(msg) == [a, b]
    assert extract_users({"channel": "pong"}) == []


def test_parse_archive_shapes():
    lines = [json.dumps({"events": [["0xAA", {"coin": "BTC"}]]}), json.dumps(["0xBB", {"coin": "ETH"}]), "junk"]
    assert [u for u, _ in parse_fill_lines(lines)] == ["0xaa", "0xbb"]


def test_storage_roundtrip_and_pseudonym(tmp_path):
    data = {"fills": [{"time": 1}], "ledger": [], "funding": [], "state": {"x": 1}, "meta": {}}
    save_wallet(tmp_path, "0xABC", data)
    assert load_wallet(tmp_path, "0xabc")["fills"] == [{"time": 1}]
    assert wallet_id("0xABC", "s") == wallet_id("0xabc", "s") != wallet_id("0xabc", "t")


def test_harvest_counts_roundtrip(tmp_path):
    from collections import Counter

    from hyperbehave.ingest.harvest import load_counts, save_counts

    path = tmp_path / "a.csv"
    save_counts(Counter({"0x" + "1" * 40: 3}), path)
    c = load_counts(path)
    c.update(["0x" + "1" * 40, "0x" + "2" * 40])
    save_counts(c, path)
    assert load_counts(path) == Counter({"0x" + "1" * 40: 4, "0x" + "2" * 40: 1})
