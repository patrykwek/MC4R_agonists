"""SMARTS-based chemotype-novelty scoring component."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mc4gen.chemotypes import Chemotype, load_chemotypes, match_any
from mc4gen.reinvent_plugins.dtos import ComponentResults


@dataclass(frozen=True, slots=True)
class ChemotypeNoveltyParameters:
    yaml_path: Path | None = None
    penalty_hit: float = 0.0
    reward_miss: float = 1.0


@dataclass
class ChemotypeNoveltyFilter:
    params: ChemotypeNoveltyParameters = field(default_factory=ChemotypeNoveltyParameters)
    chemotypes: list[Chemotype] = field(init=False)
    name: str = "chemotype_novelty"

    def __post_init__(self) -> None:
        self.chemotypes = load_chemotypes(self.params.yaml_path)

    @classmethod
    def from_parameters(cls, parameters: dict[str, object]) -> "ChemotypeNoveltyFilter":
        yaml_path = parameters.get("yaml_path")
        return cls(
            params=ChemotypeNoveltyParameters(
                yaml_path=Path(str(yaml_path)) if yaml_path else None,
                penalty_hit=float(parameters.get("penalty_hit", 0.0)),
                reward_miss=float(parameters.get("reward_miss", 1.0)),
            )
        )

    def __call__(self, smiles_iter: Sequence[str]) -> ComponentResults:
        rewards: list[float] = []
        meta: list[dict[str, object]] = []
        for smi in smiles_iter:
            hits = match_any(smi, self.chemotypes)
            if hits:
                rewards.append(self.params.penalty_hit)
            else:
                rewards.append(self.params.reward_miss)
            meta.append({"matched_chemotypes": hits})
        return ComponentResults(
            scores=np.asarray(rewards, dtype=np.float32), metadata=tuple(meta)
        )
