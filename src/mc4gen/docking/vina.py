"""AutoDock Vina 1.2 wrapper (Python bindings)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from mc4gen._logging import get_logger
from mc4gen.data import cache, structures
from mc4gen.docking.prep import prepare_ligand

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DockingResult:
    smiles: str
    pdb_id: str
    score: float
    pose_pdbqt: str
    constraints_satisfied: dict[str, bool] = field(default_factory=dict)
    interactions: dict[str, object] = field(default_factory=dict)


def _vina_engine(
    receptor_pdbqt: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    exhaustiveness: int,
    n_poses: int,
    seed: int,
):
    from vina import Vina

    vina = Vina(sf_name="vina", cpu=0, seed=seed)
    vina.set_receptor(str(receptor_pdbqt))
    vina.compute_vina_maps(center=list(center), box_size=list(size))
    vina.set_exhaustiveness(exhaustiveness)
    vina.set_n_poses(n_poses)
    return vina


def _cache_key(smiles: str, pdb_id: str, seed: int) -> str:
    return cache.hash_key("vina", smiles, pdb_id, str(seed))


def dock_smiles(
    smiles: str,
    pdb_id: str,
    *,
    exhaustiveness: int = 16,
    n_poses: int = 10,
    seed: int = 42,
    use_cache: bool = True,
) -> DockingResult | None:
    """Dock a single SMILES; result is cached by ``(smiles, pdb_id, seed)``."""
    key = _cache_key(smiles, pdb_id, seed)
    if use_cache:
        hit = cache.json_get("vina", key)
        if hit is not None:
            return DockingResult(**hit)

    receptor = structures.prepare_receptor(pdb_id)
    center, size = structures.DEFAULT_GRID_BOX[pdb_id]
    work_dir = cache.subcache(f"ligands/{pdb_id.lower()}")
    prep = prepare_ligand(smiles, work_dir)
    if prep is None:
        return None

    vina = _vina_engine(receptor, center, size, exhaustiveness=exhaustiveness, n_poses=n_poses, seed=seed)
    vina.set_ligand_from_file(str(prep.pdbqt_path))
    vina.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    scores = vina.energies()
    if scores is None or len(scores) == 0:
        return None
    top_score = float(scores[0][0])
    pose_str = vina.poses(n_poses=1)

    result = DockingResult(
        smiles=smiles,
        pdb_id=pdb_id,
        score=top_score,
        pose_pdbqt=pose_str,
    )
    if use_cache:
        cache.json_put(
            "vina",
            key,
            {
                "smiles": result.smiles,
                "pdb_id": result.pdb_id,
                "score": result.score,
                "pose_pdbqt": result.pose_pdbqt,
                "constraints_satisfied": result.constraints_satisfied,
                "interactions": result.interactions,
            },
        )
    return result


def dock_many(
    smiles_iter: Iterable[str],
    pdb_ids: Sequence[str],
    *,
    exhaustiveness: int = 16,
    n_poses: int = 10,
) -> list[dict[str, DockingResult]]:
    results: list[dict[str, DockingResult]] = []
    for smi in smiles_iter:
        per_structure: dict[str, DockingResult] = {}
        for pdb in pdb_ids:
            res = dock_smiles(smi, pdb, exhaustiveness=exhaustiveness, n_poses=n_poses)
            if res is not None:
                per_structure[pdb] = res
        results.append(per_structure)
    return results


def ensemble_score(per_structure: dict[str, DockingResult]) -> float:
    """Return the median Vina score across structures (lower is better)."""
    if not per_structure:
        return math.inf
    scores = sorted(r.score for r in per_structure.values())
    mid = len(scores) // 2
    if len(scores) % 2 == 1:
        return scores[mid]
    return 0.5 * (scores[mid - 1] + scores[mid])
