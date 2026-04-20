"""Package-level logging.

Usage: ``log = get_logger(__name__)``. Set level via env ``MC4GEN_LOG_LEVEL``.
No module in the package should call :func:`print`.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("MC4GEN_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
    root = logging.getLogger("mc4gen")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the mc4gen root."""
    _configure_root()
    if not name.startswith("mc4gen"):
        name = f"mc4gen.{name}"
    return logging.getLogger(name)
