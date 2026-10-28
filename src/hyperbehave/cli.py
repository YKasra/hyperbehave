"""Command line interface: ``hyperbehave <command> --help``."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from hyperbehave import load_config

DAY_MS = 86_400_000
DEFAULT_CANDLES = "data/raw/candles_BTC_1h.parquet"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _date_ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _window(args) -> tuple[int, int]:
    end = _date_ms(args.end) if args.end else _now_ms()
    start = _date_ms(args.start) if args.start else end - args.days * DAY_MS
    return start, end


# ----------------------------------------------------------------------------- commands
def cmd_harvest(args) -> None:
    from hyperbehave.ingest.harvest import harvest_addresses

    cfg = load_config(args.config)
    print(f"Listening for {args.minutes:g} min. Progress every 30 s; Ctrl+C stops and saves.")
    harvest_addresses(args.coins, args.minutes, args.out, cfg["api"]["ws_url"])


def cmd_fetch(args) -> None:
    from hyperbehave.ingest.api_client import HyperliquidAPIError, HyperliquidInfoClient
    from hyperbehave.storage import list_wallets, save_wallet

    cfg = load_config(args.config)
    with open(args.addresses, newline="", encoding="utf-8") as fh:
        addresses = [row["address"].lower() for row in csv.DictReader(fh)]
    if args.sample:
        import random

        rng = random.Random(args.seed)
        addresses = rng.sample(addresses, min(args.sample, len(addresses)))
    elif args.limit:
        addresses = addresses[: args.limit]
    done = set(list_wallets(args.out)) if args.resume else set()
    todo = [a for a in addresses if a not in done]
    start, end = _window(args)
    client = HyperliquidInfoClient(cfg["api"]["base_url"], cfg["api"]["min_interval_s"], cfg["api"]["max_retries"])
    print(f"Fetching {len(todo)} wallets ({len(done)} already on disk). Ctrl+C is safe; rerun with --resume.")
    t0 = time.monotonic()
    for i, addr in enumerate(todo, 1):
        if not args.end:
            # Without a fixed end, use "now" for each wallet so its data runs right up to
            # the account snapshot taken in the same call (needed to anchor the balance).
            end = _now_ms()
            if not args.start:
                start = end - args.days * DAY_MS
        try:
            data = client.fetch_wallet(addr, start, end)
            save_wallet(args.out, addr, data)
            note = f"{len(data['fills'])} fills, {len(data['ledger'])} ledger events"
        except HyperliquidAPIError as exc:
            note = f"skipped: {exc}"
        per = (time.monotonic() - t0) / i
        print(f"  {i}/{len(todo)} {addr[:10]}... {note} (about {per * (len(todo) - i) / 60:.0f} min left)")


def cmd_candles(args) -> None:
    from hyperbehave.ingest.api_client import HyperliquidInfoClient

    cfg = load_config(args.config)
    start, end = _window(args)
    client = HyperliquidInfoClient(cfg["api"]["base_url"], cfg["api"]["min_interval_s"])
    rows = client.candles(args.coin, "1h", start, end)
    df = pd.DataFrame({"t": [int(r["t"]) for r in rows], "c": [float(r["c"]) for r in rows]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"{len(df)} hourly candles -> {args.out}")


def cmd_features(args) -> None:
    from hyperbehave.features import build_feature_table
    from hyperbehave.markers import market_shocks
    from hyperbehave.storage import get_salt, iter_wallets

    cfg = load_config(args.config)
    shocks = None
    candles = args.candles
    if candles is None and Path(DEFAULT_CANDLES).exists():
        candles = DEFAULT_CANDLES
    if candles:
        shocks = market_shocks(pd.read_parquet(candles), cfg["markers"]["volatility"]["shock_quantile"])
        print(f"Using market candles from {candles}")
    else:
        print("No candles found: volatility_reactivity will be empty. Run 'hyperbehave candles' first.")
    table = build_feature_table(iter_wallets(args.raw), cfg, get_salt(args.private), shocks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    kept = int((~table["exclude"].astype(bool)).sum()) if len(table) else 0
    print(f"{len(table)} wallets ({kept} included) -> {args.out}")


def cmd_synthetic(args) -> None:
    from hyperbehave import synthetic
    from hyperbehave.storage import save_wallet

    wallets, candles = synthetic.generate(args.n, args.days, args.bots, args.seed)
    out = Path(args.out)
    for addr, data in wallets.items():
        save_wallet(out / "wallets", addr, data)
    candles.to_parquet(out / "candles_BTC_1h.parquet", index=False)
    print(f"{len(wallets)} synthetic wallets -> {out}")


def cmd_validate(args) -> None:
    from hyperbehave import synthetic
    from hyperbehave.features import build_feature_table
    from hyperbehave.markers import market_shocks
    from hyperbehave.report import validation_report

    cfg = load_config(args.config)
    wallets, candles = synthetic.generate(args.n, args.days, args.bots, args.seed)
    shocks = market_shocks(candles, cfg["markers"]["volatility"]["shock_quantile"])
    extra = {a: {"archetype": d["meta"]["synthetic_archetype"]} for a, d in wallets.items()}
    table = build_feature_table(wallets.items(), cfg, "synthetic", shocks, extra=extra)
    path = validation_report(table, args.out)
    table.to_parquet(Path(args.out) / "synthetic_features.parquet", index=False)
    print(f"Validation report -> {path}")


def cmd_describe(args) -> None:
    from hyperbehave.report import layer_a_report

    table = pd.read_parquet(args.features)
    path = layer_a_report(table, args.out, with_profiles=not args.no_profiles)
    print(f"Layer A report -> {path}")


def cmd_inspect(args) -> None:
    """Aggregate counts of event types in the raw data (no addresses), for checking
    that every ledger and fill type is parsed."""
    from collections import Counter

    from hyperbehave.storage import iter_wallets

    ledger_types, dirs, coins = Counter(), Counter(), Counter()
    send_dex, n = Counter(), 0
    for _, raw in iter_wallets(args.raw):
        n += 1
        for rec in raw.get("ledger", []):
            d = rec.get("delta") or {}
            ledger_types[d.get("type", "?")] += 1
            if d.get("type") == "send":
                send_dex[f"{d.get('sourceDex', '?')!r} -> {d.get('destinationDex', '?')!r} {d.get('token', '?')}"] += 1
        for f in raw.get("fills", []):
            dirs[f.get("dir", "?")] += 1
            c = str(f.get("coin", ""))
            coins["spot" if c.startswith("@") or "/" in c else ("builder perp" if ":" in c else "perp")] += 1
    print(f"wallets: {n}")
    for title, counter in (("ledger event types", ledger_types), ("send routes", send_dex),
                           ("fill directions", dirs), ("market kinds", coins)):
        print(f"\n{title}:")
        for k, v in counter.most_common(25):
            print(f"  {v:>8}  {k}")


def cmd_s3_size(args) -> None:
    from hyperbehave.ingest.s3_archive import size_range

    s = date.fromisoformat(args.start)
    e = date.fromisoformat(args.end)
    print(size_range(s, e))


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hyperbehave", description=__doc__)
    p.add_argument("--config", help="YAML file overriding the default configuration")
    sub = p.add_subparsers(dest="command", required=True)

    def window(sp):
        sp.add_argument("--start", help="YYYY-MM-DD (UTC); default: --days before --end")
        sp.add_argument("--end", help="YYYY-MM-DD (UTC); default: now")
        sp.add_argument("--days", type=int, default=90)

    sp = sub.add_parser("harvest", help="collect active wallet addresses from the public trades stream")
    sp.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL", "HYPE", "DOGE", "XRP"])
    sp.add_argument("--minutes", type=float, default=60)
    sp.add_argument("--out", default="data/raw/addresses.csv")
    sp.set_defaults(func=cmd_harvest)

    sp = sub.add_parser("fetch", help="download fills, ledger, funding and state per wallet")
    sp.add_argument("--addresses", default="data/raw/addresses.csv")
    sp.add_argument("--out", default="data/raw/wallets")
    sp.add_argument("--limit", type=int, default=0, help="first N addresses (the busiest, often bots)")
    sp.add_argument("--sample", type=int, default=0, help="random N addresses (recommended)")
    sp.add_argument("--seed", type=int, default=42, help="random seed for --sample")
    sp.add_argument("--resume", action="store_true", help="skip wallets already on disk")
    window(sp)
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("candles", help="hourly candles for the market-shock marker")
    sp.add_argument("--coin", default="BTC")
    sp.add_argument("--out", default=DEFAULT_CANDLES)
    window(sp)
    sp.set_defaults(func=cmd_candles)

    sp = sub.add_parser("features", help="compute the behavioural markers for every wallet")
    sp.add_argument("--raw", default="data/raw/wallets")
    sp.add_argument("--candles", default=None, help=f"default: {DEFAULT_CANDLES} if it exists")
    sp.add_argument("--private", default="data/private", help="folder holding the secret salt")
    sp.add_argument("--out", default="data/processed/features.parquet")
    sp.set_defaults(func=cmd_features)

    sp = sub.add_parser("synthetic", help="write synthetic wallets in API format")
    sp.add_argument("--n", type=int, default=30, help="wallets per archetype")
    sp.add_argument("--days", type=int, default=90)
    sp.add_argument("--bots", type=int, default=5)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--out", default="data/synthetic")
    sp.set_defaults(func=cmd_synthetic)

    sp = sub.add_parser("validate", help="check that markers recover the programmed archetypes")
    sp.add_argument("--n", type=int, default=30)
    sp.add_argument("--days", type=int, default=90)
    sp.add_argument("--bots", type=int, default=5)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--out", default="reports/synthetic_validation")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("describe", help="Layer A population report from a feature table")
    sp.add_argument("--features", default="data/processed/features.parquet")
    sp.add_argument("--out", default="reports/layer_a")
    sp.add_argument("--no-profiles", action="store_true")
    sp.set_defaults(func=cmd_describe)

    sp = sub.add_parser("inspect", help="count event types in the raw data (no addresses)")
    sp.add_argument("--raw", default="data/raw/wallets")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("s3-size", help="size and cost of an S3 archive date range (requester pays)")
    sp.add_argument("--start", required=True)
    sp.add_argument("--end", required=True)
    sp.set_defaults(func=cmd_s3_size)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
