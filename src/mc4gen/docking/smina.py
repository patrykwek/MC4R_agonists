"""smina (Vina fork) wrapper for cross-validation of scores."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mc4gen._logging import get_logger
from mc4gen.data import cache, structures
from mc4gen.docking.prep import prepare_ligand

log = get_logger(__name__)

_AFFINITY_PATTERN = re.compile(r"Affinity:\s*(-?\d+\.\d+)")


@dataclass(frozen=True, slots=True)
class SminaResult:
    smiles: str
    pdb_id: str
    score: float
    out_path: Path


def _smina_executable() -> str:
    exe = shutil.which("smina") or shutil.which("smina.static")
    if exe is None:
        raise RuntimeError("smina not found on $PATH")
    return exe


def dock(
    smiles: str,
    pdb_id: str,
    *,
    exhaustiveness: int = 16,
    scoring: str = "vinardo",
    seed: int = 42,
) -> SminaResult | None:
    receptor = structures.prepare_receptor(pdb_id)
    center, size = structures.DEFAULT_GRID_BOX[pdb_id]
    work_dir = cache.subcache(f"smina/{pdb_id.lower()}")
    prep = prepare_ligand(smiles, work_dir)
    if prep is None:
        return None

    out_path = work_dir / f"{prep.pdbqt_path.stem}_out.pdbqt"
    cmd = [
        _smina_executable(),
        "--receptor",
        str(receptor),
        "--ligand",
        str(prep.pdbqt_path),
        "--out",
        str(out_path),
        "--center_x",
        f"{center[0]}",
        "--center_y",
        f"{center[1]}",
        "--center_z",
        f"{center[2]}",
        "--size_x",
        f"{size[0]}",
        "--size_y",
        f"{size[1]}",
        "--size_z",
        f"{size[2]}",
        "--exhaustiveness",
        str(exhaustiveness),
        "--scoring",
        scoring,
        "--seed",
        str(seed),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as err:
        log.error("smina failed: %s", err.stderr)
        return None
    match = _AFFINITY_PATTERN.search(proc.stdout)
    if match is None:
        return None
    score = float(match.group(1))
    return SminaResult(smiles=smiles, pdb_id=pdb_id, score=score, out_path=out_path)
