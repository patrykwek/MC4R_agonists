"""Predict whether a molecule's docked pose coordinates the orthosteric Ca2+."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from mc4gen.docking.constraints import CA_COORDINATION, GLU100_HBOND, DistanceConstraint
from mc4gen.docking.vina import dock_smiles
from mc4gen.reinvent_plugins.dtos import ComponentResults


@dataclass(frozen=True, slots=True)
class CalciumCoordinationParameters:
    pdb_id: str = "7PIU"
    require_both: bool = False


def _parse_pdbqt_coordinates(pdbqt_text: str) -> np.ndarray:
    coords: list[list[float]] = []
    for line in io.StringIO(pdbqt_text):
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append([x, y, z])
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def _first_model(pdbqt_text: str) -> str:
    parts = re.split(r"^ENDMDL.*$", pdbqt_text, flags=re.MULTILINE)
    return parts[0]


@dataclass
class CalciumCoordinationPredictor:
    params: CalciumCoordinationParameters = field(default_factory=CalciumCoordinationParameters)
    ca_constraint: DistanceConstraint = field(default=CA_COORDINATION)
    glu100_constraint: DistanceConstraint = field(default=GLU100_HBOND)
    name: str = "calcium_coordination"

    @classmethod
    def from_parameters(cls, parameters: dict[str, object]) -> "CalciumCoordinationPredictor":
        return cls(
            params=CalciumCoordinationParameters(
                pdb_id=str(parameters.get("pdb_id", "7PIU")),
                require_both=bool(parameters.get("require_both", False)),
            )
        )

    def _score_one(self, smi: str) -> tuple[float, dict[str, object]]:
        result = dock_smiles(smi, self.params.pdb_id)
        if result is None:
            return 0.0, {"reason": "docking_failed"}
        coords = _parse_pdbqt_coordinates(_first_model(result.pose_pdbqt))
        ca_penalty, ca_ok = self.ca_constraint.penalty(coords)
        glu_penalty, glu_ok = self.glu100_constraint.penalty(coords)
        if self.params.require_both:
            reward = 1.0 if (ca_ok and glu_ok) else 0.0
        else:
            reward = 1.0 if ca_ok else 0.0
        return reward, {
            "ca_ok": ca_ok,
            "glu100_ok": glu_ok,
            "ca_penalty": ca_penalty,
            "glu100_penalty": glu_penalty,
        }

    def __call__(self, smiles_iter: Sequence[str]) -> ComponentResults:
        scores: list[float] = []
        meta: list[dict[str, object]] = []
        for smi in smiles_iter:
            reward, details = self._score_one(smi)
            scores.append(reward)
            meta.append(details)
        return ComponentResults(
            scores=np.asarray(scores, dtype=np.float32), metadata=tuple(meta)
        )
