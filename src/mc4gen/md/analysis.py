"""MD trajectory analysis: RMSD, RMSF, contacts, Ca2+ persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class StabilityReport:
    ligand_rmsd_mean: float
    ligand_rmsd_std: float
    pocket_backbone_rmsd_mean: float
    ca_contact_fraction: float
    n_frames: int


@dataclass(frozen=True, slots=True)
class ContactSummary:
    residue: str
    fraction_frames: float


def _load_universe(prmtop: Path, trajectory: Path):
    import MDAnalysis as mda

    return mda.Universe(str(prmtop), str(trajectory))


def ligand_rmsd(prmtop: Path, trajectory: Path, ligand_resname: str = "LIG") -> NDArray[np.float32]:
    from MDAnalysis.analysis.rms import RMSD

    universe = _load_universe(prmtop, trajectory)
    selection = f"resname {ligand_resname} and not name H*"
    rmsd = RMSD(universe, universe, select=selection, ref_frame=0)
    rmsd.run()
    return np.asarray(rmsd.rmsd[:, 2], dtype=np.float32)


def pocket_backbone_rmsd(
    prmtop: Path,
    trajectory: Path,
    pocket_residues: Sequence[int],
) -> NDArray[np.float32]:
    from MDAnalysis.analysis.rms import RMSD

    universe = _load_universe(prmtop, trajectory)
    resids = "+".join(str(r) for r in pocket_residues)
    selection = f"backbone and resid {resids}"
    rmsd = RMSD(universe, universe, select=selection, ref_frame=0)
    rmsd.run()
    return np.asarray(rmsd.rmsd[:, 2], dtype=np.float32)


def calcium_contact_fraction(
    prmtop: Path,
    trajectory: Path,
    *,
    ligand_resname: str = "LIG",
    ion_selection: str = "resname CA and element Ca",
    cutoff: float = 3.0,
) -> float:
    universe = _load_universe(prmtop, trajectory)
    lig = universe.select_atoms(f"resname {ligand_resname} and (element O or element N)")
    ion = universe.select_atoms(ion_selection)
    if len(lig) == 0 or len(ion) == 0:
        return 0.0
    contact_frames = 0
    n_frames = 0
    for _ in universe.trajectory:
        diff = lig.positions[:, None, :] - ion.positions[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        if float(distances.min()) <= cutoff:
            contact_frames += 1
        n_frames += 1
    return contact_frames / n_frames if n_frames else 0.0


def summarize(
    prmtop: Path,
    trajectories: Iterable[Path],
    *,
    pocket_residues: Sequence[int],
    ligand_resname: str = "LIG",
) -> StabilityReport:
    lig_rmsds: list[NDArray[np.float32]] = []
    pocket_rmsds: list[NDArray[np.float32]] = []
    ca_fracs: list[float] = []
    for traj in trajectories:
        lig_rmsds.append(ligand_rmsd(prmtop, traj, ligand_resname))
        pocket_rmsds.append(pocket_backbone_rmsd(prmtop, traj, pocket_residues))
        ca_fracs.append(calcium_contact_fraction(prmtop, traj, ligand_resname=ligand_resname))
    if not lig_rmsds:
        return StabilityReport(0.0, 0.0, 0.0, 0.0, 0)
    lig = np.concatenate(lig_rmsds)
    pocket = np.concatenate(pocket_rmsds)
    return StabilityReport(
        ligand_rmsd_mean=float(lig.mean()),
        ligand_rmsd_std=float(lig.std()),
        pocket_backbone_rmsd_mean=float(pocket.mean()),
        ca_contact_fraction=float(np.mean(ca_fracs)),
        n_frames=int(lig.size),
    )
