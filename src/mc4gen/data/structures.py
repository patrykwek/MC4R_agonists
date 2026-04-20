"""Download, prepare and cache the five MC4R cryo-EM structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

from mc4gen._logging import get_logger
from mc4gen.data import cache

log = get_logger(__name__)

PDB_BASE_URL = "https://files.rcsb.org/download"


@dataclass(frozen=True, slots=True)
class MC4RStructure:
    pdb_id: str
    state: str
    ligand: str
    resolution_a: float
    has_calcium: bool
    notes: str


STRUCTURES: dict[str, MC4RStructure] = {
    "7AUE": MC4RStructure("7AUE", "active", "setmelanotide", 2.9, True, "Gs-coupled reference."),
    "7PIU": MC4RStructure("7PIU", "active", "setmelanotide", 2.6, True, "Higher-resolution; Ca2+ visible."),
    "7PIV": MC4RStructure("7PIV", "active", "NDP-alpha-MSH", 2.86, True, "Linear peptide mode."),
    "6W25": MC4RStructure("6W25", "inactive", "SHU9119", 2.75, False, "Antagonist template."),
    "7F53": MC4RStructure("7F53", "active", "bremelanotide", 3.0, True, "Secondary validation."),
}

DEFAULT_GRID_BOX: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # (center_xyz, size_xyz) in angstroms, coarsely centered on orthosteric pocket.
    "7AUE": ((0.5, -1.2, 3.8), (24.0, 24.0, 24.0)),
    "7PIU": ((0.6, -1.0, 3.9), (22.0, 22.0, 22.0)),
    "7PIV": ((0.3, -1.5, 3.5), (24.0, 24.0, 24.0)),
    "6W25": ((0.0, -1.0, 3.2), (24.0, 24.0, 24.0)),
    "7F53": ((0.4, -1.2, 3.6), (24.0, 24.0, 24.0)),
}


def download_pdb(pdb_id: str, *, force: bool = False) -> Path:
    """Fetch a PDB file from RCSB into the structure cache."""
    key = f"{pdb_id.lower()}.pdb"
    path = cache.path_for("structures", pdb_id.lower(), ".pdb")
    if path.exists() and not force:
        return path
    url = f"{PDB_BASE_URL}/{pdb_id.upper()}.pdb"
    log.info("Downloading %s from %s", pdb_id, url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return path


def prepare_receptor(pdb_id: str, *, ph: float = 7.4) -> Path:
    """Run pdbfixer + propka + PDBQT generation; return PDBQT path."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    pdb_path = download_pdb(pdb_id)
    out_dir = cache.subcache("receptors")
    fixed_pdb = out_dir / f"{pdb_id.lower()}_fixed.pdb"
    pdbqt = out_dir / f"{pdb_id.lower()}.pdbqt"
    if pdbqt.exists():
        return pdbqt

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=ph)
    with fixed_pdb.open("w", encoding="utf-8") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

    # Hand off to the dedicated receptor-prep module for PDBQT conversion.
    from mc4gen.docking.receptor_prep import to_pdbqt

    to_pdbqt(fixed_pdb, pdbqt)
    return pdbqt


def all_structures() -> list[MC4RStructure]:
    return list(STRUCTURES.values())


def active_structures() -> list[MC4RStructure]:
    return [s for s in STRUCTURES.values() if s.state == "active"]
