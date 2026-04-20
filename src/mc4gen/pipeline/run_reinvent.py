"""Execute REINVENT 4 TOML configurations as subprocesses."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ReinventRunResult:
    config: Path
    output_dir: Path
    csv_path: Path
    log_path: Path
    return_code: int


def _reinvent_executable() -> str:
    for candidate in ("reinvent", "reinvent4"):
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError("REINVENT 4 CLI not found on $PATH; install the 'reinvent' PyPI package.")


def run_reinvent_stage(
    config_path: Path,
    *,
    output_dir: Path,
    logfile: Path | None = None,
) -> ReinventRunResult:
    """Run REINVENT 4 against the provided TOML and return paths to its outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "molecules.csv"
    log_path = logfile or (output_dir / "reinvent.log")
    exe = _reinvent_executable()
    cmd = [
        exe,
        "-f",
        "toml",
        "-l",
        str(log_path),
        str(config_path),
    ]
    log.info("Executing REINVENT: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("REINVENT failed (code %d):\n%s", proc.returncode, proc.stderr[-2000:])
    return ReinventRunResult(
        config=config_path,
        output_dir=output_dir,
        csv_path=csv_path,
        log_path=log_path,
        return_code=proc.returncode,
    )


def collect_outputs(run_dirs: list[Path]) -> pd.DataFrame:
    """Concatenate ``molecules.csv`` outputs from multiple REINVENT runs."""
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        csv = run_dir / "molecules.csv"
        if not csv.exists():
            log.warning("Missing output %s", csv)
            continue
        df = pd.read_csv(csv)
        df["run"] = run_dir.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
