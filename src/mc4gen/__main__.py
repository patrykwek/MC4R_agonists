"""Package CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from mc4gen import __version__
from mc4gen._logging import get_logger
from mc4gen.pipeline.prioritize import run_prioritization
from mc4gen.pipeline.run_reinvent import run_reinvent_stage

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(prog="mc4gen")
    parser.add_argument("--version", action="version", version=f"mc4gen {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Execute a REINVENT 4 TOML configuration.")
    p_run.add_argument("config", type=Path)
    p_run.add_argument("--output", type=Path, default=Path("runs/latest"))

    p_pri = sub.add_parser("prioritize", help="Execute the 11-stage prioritization funnel.")
    p_pri.add_argument("smiles_csv", type=Path)
    p_pri.add_argument("--top", type=int, default=10)

    args = parser.parse_args()
    if args.cmd == "run":
        run_reinvent_stage(args.config, output_dir=args.output)
    elif args.cmd == "prioritize":
        run_prioritization(args.smiles_csv, max_candidates=args.top)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
