"""gmx_MMPBSA rescoring wrapper (open-source MM-GBSA)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MMPBSAResult:
    complex_prmtop: Path
    summary_csv: Path
    delta_g_kcal: float
    stderr: str


_DEFAULT_IN = """\
&general
  startframe=1, endframe=200, interval=10, verbose=2,
/
&gb
  igb=8, saltcon=0.150,
/
&decomp
  idecomp=0,
/
"""


def _require(exe: str) -> str:
    path = shutil.which(exe)
    if path is None:
        raise RuntimeError(f"{exe} not found on $PATH (install gmx_MMPBSA).")
    return path


def write_default_input(path: Path) -> Path:
    path.write_text(_DEFAULT_IN, encoding="utf-8")
    return path


def rescore(
    complex_prmtop: Path,
    complex_traj: Path,
    ligand_mask: str,
    receptor_mask: str,
    out_dir: Path,
) -> MMPBSAResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = _require("gmx_MMPBSA")
    mmpbsa_in = write_default_input(out_dir / "mmpbsa.in")
    summary_csv = out_dir / "FINAL_RESULTS_MMPBSA.csv"
    cmd = [
        exe,
        "MPI",
        "-O",
        "-i",
        str(mmpbsa_in),
        "-cs",
        str(complex_prmtop),
        "-ct",
        str(complex_traj),
        "-lm",
        ligand_mask,
        "-rm",
        receptor_mask,
        "-eo",
        str(out_dir / "ENERGIES.csv"),
        "-deo",
        str(out_dir / "DECOMP.csv"),
        "-o",
        str(summary_csv),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    delta_g = _parse_delta_g(summary_csv)
    return MMPBSAResult(
        complex_prmtop=complex_prmtop,
        summary_csv=summary_csv,
        delta_g_kcal=delta_g,
        stderr=proc.stderr,
    )


def _parse_delta_g(csv_path: Path) -> float:
    if not csv_path.exists():
        return float("nan")
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("DELTA TOTAL"):
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return float("nan")
    return float("nan")
