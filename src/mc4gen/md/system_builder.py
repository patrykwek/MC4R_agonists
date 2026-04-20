"""Membrane system builder (packmol-memgen driver)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MembraneSystem:
    name: str
    prmtop: Path
    inpcrd: Path
    built_pdb: Path


def _require(exe: str) -> str:
    path = shutil.which(exe)
    if path is None:
        raise RuntimeError(f"{exe} not found on $PATH (install AmberTools).")
    return path


def build_popc_system(
    receptor_pdb: Path,
    ligand_mol2: Path,
    out_dir: Path,
    *,
    name: str,
    padding_xy: float = 20.0,
    padding_z: float = 15.0,
    lipid: str = "POPC",
) -> MembraneSystem:
    out_dir.mkdir(parents=True, exist_ok=True)
    packmol_memgen = _require("packmol-memgen")
    cmd = [
        packmol_memgen,
        "--pdb",
        str(receptor_pdb),
        "--lipids",
        lipid,
        "--salt",
        "--salt_c",
        "Na+",
        "--saltcon",
        "0.15",
        "--dist_wat",
        f"{padding_z}",
        "--dist",
        f"{padding_xy}",
        "--preserve",
        "--overwrite",
        "--output",
        str(out_dir / f"{name}_built.pdb"),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    built_pdb = out_dir / f"{name}_built.pdb"

    tleap = _require("tleap")
    prmtop = out_dir / f"{name}.prmtop"
    inpcrd = out_dir / f"{name}.inpcrd"
    leap_in = out_dir / "build.leap"
    leap_in.write_text(
        "source leaprc.protein.ff14SB\n"
        "source leaprc.lipid17\n"
        "source leaprc.water.tip3p\n"
        "source leaprc.gaff2\n"
        f"lig = loadmol2 {ligand_mol2}\n"
        f"sys = loadpdb {built_pdb}\n"
        f"saveamberparm sys {prmtop} {inpcrd}\n"
        "quit\n",
        encoding="utf-8",
    )
    subprocess.run([tleap, "-f", str(leap_in)], check=True, capture_output=True, text=True)
    return MembraneSystem(name=name, prmtop=prmtop, inpcrd=inpcrd, built_pdb=built_pdb)
