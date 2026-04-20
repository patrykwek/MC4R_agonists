"""Receptor preparation: PDB -> PDBQT via ADFR's prepare_receptor4."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from mc4gen._logging import get_logger

log = get_logger(__name__)


def _prepare_receptor_executable() -> str:
    """Locate ``prepare_receptor`` / ``prepare_receptor4.py`` on ``$PATH``."""
    for candidate in ("prepare_receptor", "prepare_receptor4.py", "prepare_receptor4"):
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError(
        "ADFR's prepare_receptor not found on $PATH. Install ADFR suite "
        "(https://ccsb.scripps.edu/adfr/) and ensure the binary is available."
    )


def to_pdbqt(pdb_in: Path, pdbqt_out: Path) -> Path:
    """Convert a prepared PDB to PDBQT."""
    exe = _prepare_receptor_executable()
    pdbqt_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-r", str(pdb_in), "-o", str(pdbqt_out), "-A", "checkhydrogens"]
    log.info("prepare_receptor: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"prepare_receptor failed: {err.stderr}") from err
    if not pdbqt_out.exists():
        raise RuntimeError(f"prepare_receptor produced no output at {pdbqt_out}")
    return pdbqt_out


def assign_propka_protonation(pdb_in: Path, *, ph: float = 7.4) -> dict[str, float]:
    """Return residue -> pKa mapping from PROPKA at the requested pH."""
    import propka.run as propka_run

    params = propka_run.single(str(pdb_in), optargs=["--quiet"])
    result: dict[str, float] = {}
    for group in params.conformations["AVR"].groups:
        key = f"{group.residue_type} {group.atom.res_num}{group.atom.chain_id}"
        result[key] = float(group.pka_value)
    return result
