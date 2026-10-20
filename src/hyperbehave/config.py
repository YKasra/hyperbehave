"""Configuration loading.

The package ships a default configuration (``default_config.yaml``). A user file can
override any subset of keys; values are merged recursively.
"""

from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the default config, optionally overridden by a YAML file."""
    text = resources.files("hyperbehave").joinpath("default_config.yaml").read_text("utf-8")
    cfg = yaml.safe_load(text)
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, user)
    return cfg
