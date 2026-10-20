"""Optional: full population history from the Hyperliquid S3 node archive.

The archive bucket ``s3://hl-mainnet-node-data`` is public but **requester pays**: the
person downloading pays AWS data transfer (roughly USD 0.09 per GB). Fills from
27 July 2025 onward live under ``node_fills_by_block/hourly/YYYYMMDD/H.lz4``.

Always size a date range before downloading it::

    hyperbehave s3-size --start 2026-09-01 --end 2026-09-07

This module shells out to the AWS CLI (``aws``) and the ``lz4`` command line tool so it
needs no extra Python packages. Configure credentials once with ``aws configure``.

File formats differ between archive generations, so :func:`parse_fill_lines` accepts
the shapes seen so far and skips anything it does not recognise. Check the first file
you download by eye before running a large job.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

BUCKET = "hl-mainnet-node-data"
PREFIX = "node_fills_by_block/hourly"
USD_PER_GB = 0.09


def _days(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def size_range(start: date, end: date) -> dict[str, float]:
    """Total compressed size and an egress cost estimate for a date range."""
    total = 0
    for d in _days(start, end):
        cmd = [
            "aws", "s3", "ls", f"s3://{BUCKET}/{PREFIX}/{d:%Y%m%d}/",
            "--request-payer", "requester", "--summarize",
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if "Total Size:" in line:
                total += int(line.split(":")[1].strip())
    gb = total / 1024**3
    return {"bytes": total, "gb": round(gb, 3), "est_usd": round(gb * USD_PER_GB, 2)}


def download_range(start: date, end: date, dest: str | Path) -> list[Path]:
    """Download and decompress hourly fill files into ``dest/YYYYMMDD/H.jsonl``."""
    dest = Path(dest)
    files: list[Path] = []
    for d in _days(start, end):
        day_dir = dest / f"{d:%Y%m%d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "aws", "s3", "cp", f"s3://{BUCKET}/{PREFIX}/{d:%Y%m%d}/", str(day_dir),
                "--recursive", "--request-payer", "requester",
            ],
            check=True,
        )
        for lz in sorted(day_dir.glob("*.lz4")):
            out = lz.with_suffix(".jsonl")
            subprocess.run(["lz4", "-d", "-f", "--rm", str(lz), str(out)], check=True)
            files.append(out)
    return files


def parse_fill_lines(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(user, fill)`` pairs from archive lines.

    Recognised shapes:

    * ``{"events": [[user, fill], ...], ...}`` (one block per line)
    * ``[user, fill]`` (one fill per line, older ``node_fills``)
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("events"), list):
            for ev in obj["events"]:
                if isinstance(ev, list) and len(ev) == 2 and isinstance(ev[1], dict):
                    yield str(ev[0]).lower(), ev[1]
        elif isinstance(obj, list) and len(obj) == 2 and isinstance(obj[1], dict):
            yield str(obj[0]).lower(), obj[1]
