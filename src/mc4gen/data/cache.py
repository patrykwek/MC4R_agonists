"""Filesystem cache under ``~/.mc4gen/cache/``."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

from mc4gen._logging import get_logger

log = get_logger(__name__)

_ROOT = Path(os.environ.get("MC4GEN_CACHE_ROOT", Path.home() / ".mc4gen" / "cache"))


def cache_root() -> Path:
    _ROOT.mkdir(parents=True, exist_ok=True)
    return _ROOT


def subcache(name: str) -> Path:
    path = cache_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def json_get(scope: str, key: str) -> Any | None:
    path = subcache(scope) / f"{key}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def json_put(scope: str, key: str, value: Any) -> None:
    path = subcache(scope) / f"{key}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, default=str)


def pickle_get(scope: str, key: str) -> Any | None:
    path = subcache(scope) / f"{key}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def pickle_put(scope: str, key: str, value: Any) -> None:
    path = subcache(scope) / f"{key}.pkl"
    with path.open("wb") as fh:
        pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)


def path_for(scope: str, key: str, suffix: str = "") -> Path:
    return subcache(scope) / f"{key}{suffix}"
