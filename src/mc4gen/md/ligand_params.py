"""Ligand parameterisation via ACPYPE + antechamber (GAFF2)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mc4gen._logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LigandTopology:
    name: str
    prmtop: Path
    inpcrd: Path
    mol2: Path
    frcmod: Path


def _require(exe: str) -> str:
    path = shutil.which(exe)
    if path is None:
        raise RuntimeError(f"{exe} not found on $PATH (install AmberTools / ACPYPE).")
    return path


def parameterize(smiles_mol_block: str, out_dir: Path, *, name: str, charge: int = 0) -> LigandTopology:
    out_dir.mkdir(parents=True, exist_ok=True)
    mol_path = out_dir / f"{name}.mol"
    mol_path.write_text(smiles_mol_block, encoding="utf-8")

    antechamber = _require("antechamber")
    parmchk = _require("parmchk2")
    tleap = _require("tleap")

    mol2_path = out_dir / f"{name}.mol2"
    subprocess.run(
        [
            antechamber,
            "-i",
            str(mol_path),
            "-fi",
            "mdl",
            "-o",
            str(mol2_path),
            "-fo",
            "mol2",
            "-c",
            "bcc",
            "-s",
            "2",
            "-nc",
            str(charge),
            "-at",
            "gaff2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    frcmod = out_dir / f"{name}.frcmod"
    subprocess.run(
        [parmchk, "-i", str(mol2_path), "-f", "mol2", "-o", str(frcmod), "-s", "gaff2"],
        check=True,
        capture_output=True,
        text=True,
    )

    leap_in = out_dir / "leap.in"
    prmtop = out_dir / f"{name}.prmtop"
    inpcrd = out_dir / f"{name}.inpcrd"
    leap_in.write_text(
        "source leaprc.gaff2\n"
        f"lig = loadmol2 {mol2_path}\n"
        f"loadamberparams {frcmod}\n"
        f"saveamberparm lig {prmtop} {inpcrd}\n"
        "quit\n",
        encoding="utf-8",
    )
    subprocess.run([tleap, "-f", str(leap_in)], check=True, capture_output=True, text=True)
    return LigandTopology(name=name, prmtop=prmtop, inpcrd=inpcrd, mol2=mol2_path, frcmod=frcmod)
