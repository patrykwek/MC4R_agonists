"""REINVENT 4 structure-based scoring component using AutoDock Vina."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from mc4gen._logging import get_logger
from mc4gen.docking.vina import DockingResult, dock_smiles, ensemble_score
from mc4gen.reinvent_plugins.dtos import ComponentResults

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MC4RDockingParameters:
    pdb_ids: tuple[str, ...] = ("7PIU", "7AUE", "7PIV", "6W25", "7F53")
    target_kcal: float = -9.0
    sigmoid_slope: float = 0.5
    exhaustiveness: int = 16
    n_poses: int = 10
    require_ca_coordination: bool = False
    require_glu100_hbond: bool = False
    seed: int = 42
    minimum_structures: int = 2
    aggregation: str = "median"


def _softscale(score: float, target: float, slope: float) -> float:
    """Sigmoid reward: higher (less negative) Vina => lower reward.

    Uses ``1 / (1 + exp((score - target) / slope))`` so that ``score <= target``
    yields reward > 0.5.
    """
    try:
        return 1.0 / (1.0 + math.exp((score - target) / slope))
    except OverflowError:
        return 0.0 if score > target else 1.0


@dataclass
class MC4RDockingVina:
    params: MC4RDockingParameters = field(default_factory=MC4RDockingParameters)
    name: str = "mc4r_docking_vina"

    @classmethod
    def from_parameters(cls, parameters: dict[str, object]) -> "MC4RDockingVina":
        pdb_ids = tuple(parameters.get("pdb_ids", ("7PIU", "7AUE", "7PIV", "6W25", "7F53")))
        return cls(
            params=MC4RDockingParameters(
                pdb_ids=pdb_ids,
                target_kcal=float(parameters.get("target_kcal", -9.0)),
                sigmoid_slope=float(parameters.get("sigmoid_slope", 0.5)),
                exhaustiveness=int(parameters.get("exhaustiveness", 16)),
                n_poses=int(parameters.get("n_poses", 10)),
                require_ca_coordination=bool(parameters.get("require_ca_coordination", False)),
                require_glu100_hbond=bool(parameters.get("require_glu100_hbond", False)),
                seed=int(parameters.get("seed", 42)),
                minimum_structures=int(parameters.get("minimum_structures", 2)),
                aggregation=str(parameters.get("aggregation", "median")),
            )
        )

    def _aggregate(self, per_structure: dict[str, DockingResult]) -> float:
        if len(per_structure) < self.params.minimum_structures:
            return math.inf
        if self.params.aggregation == "min":
            return min(r.score for r in per_structure.values())
        return ensemble_score(per_structure)

    def __call__(self, smiles_iter: Sequence[str]) -> ComponentResults:
        scores: list[float] = []
        meta: list[dict[str, object]] = []
        for smi in smiles_iter:
            per_struct: dict[str, DockingResult] = {}
            for pdb in self.params.pdb_ids:
                res = dock_smiles(
                    smi,
                    pdb,
                    exhaustiveness=self.params.exhaustiveness,
                    n_poses=self.params.n_poses,
                    seed=self.params.seed,
                )
                if res is not None:
                    per_struct[pdb] = res
            aggregated = self._aggregate(per_struct)
            reward = 0.0 if math.isinf(aggregated) else _softscale(
                aggregated, self.params.target_kcal, self.params.sigmoid_slope
            )
            scores.append(reward)
            meta.append(
                {
                    "aggregated_kcal": aggregated if math.isfinite(aggregated) else None,
                    "per_structure": {k: v.score for k, v in per_struct.items()},
                    "n_structures_scored": len(per_struct),
                }
            )
        return ComponentResults(scores=np.asarray(scores, dtype=np.float32), metadata=tuple(meta))
