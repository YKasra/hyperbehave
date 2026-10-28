"""Local storage of raw wallet data and pseudonymous wallet IDs.

Raw data lives under ``data/raw/wallets/<address>/{fills,ledger,funding}.json.gz`` and
never leaves your machine (``data/`` is git-ignored). Every table that leaves the raw
layer identifies wallets only by a salted hash, so results can be shared without
publishing addresses.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Any

LIST_PARTS = ("fills", "ledger", "funding")
DICT_PARTS = ("state", "spot_state", "meta")


def save_wallet(root: str | Path, address: str, data: dict[str, Any]) -> Path:
    folder = Path(root) / address.lower()
    folder.mkdir(parents=True, exist_ok=True)
    for part in LIST_PARTS + DICT_PARTS:
        default: Any = [] if part in LIST_PARTS else {}
        with gzip.open(folder / f"{part}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(data.get(part, default), fh)
    return folder


def load_wallet(root: str | Path, address: str) -> dict[str, Any]:
    folder = Path(root) / address.lower()
    out: dict[str, Any] = {}
    for part in LIST_PARTS + DICT_PARTS:
        path = folder / f"{part}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                out[part] = json.load(fh)
        else:
            out[part] = [] if part in LIST_PARTS else {}
    return out


def list_wallets(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "fills.json.gz").exists())


def iter_wallets(root: str | Path) -> Iterator[tuple[str, dict[str, list[dict[str, Any]]]]]:
    for address in list_wallets(root):
        yield address, load_wallet(root, address)


# --------------------------------------------------------------------- pseudonymisation
def get_salt(private_dir: str | Path = "data/private") -> str:
    """Return the project salt, creating it once if needed.

    The salt can also come from the ``HYPERBEHAVE_SALT`` environment variable. Keep it
    secret: anyone with the salt can re-link wallet IDs to addresses.
    """
    env = os.environ.get("HYPERBEHAVE_SALT")
    if env:
        return env
    path = Path(private_dir) / "salt.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(32)
    path.write_text(salt, encoding="utf-8")
    return salt


def wallet_id(address: str, salt: str) -> str:
    """Stable 16-hex-character pseudonym for an address."""
    return hashlib.sha256((salt + address.lower()).encode("utf-8")).hexdigest()[:16]
