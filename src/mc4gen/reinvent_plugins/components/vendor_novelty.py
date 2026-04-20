"""Penalize molecules highly similar to cached vendor libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from mc4gen.data.vendors import load_vendor_pool, max_tanimoto_to_vendor
from mc4gen.reinvent_plugins.dtos import ComponentResults


@dataclass(frozen=True, slots=True)
class VendorNoveltyParameters:
    max_tanimoto: float = 0.6
    library_size: int = 200_000


@dataclass
class VendorNoveltyFilter:
    params: VendorNoveltyParameters = field(default_factory=VendorNoveltyParameters)
    pool: np.ndarray = field(init=False)
    name: str = "vendor_novelty"

    def __post_init__(self) -> None:
        self.pool = load_vendor_pool(max_entries=self.params.library_size)

    @classmethod
    def from_parameters(cls, parameters: dict[str, object]) -> "VendorNoveltyFilter":
        return cls(
            params=VendorNoveltyParameters(
                max_tanimoto=float(parameters.get("max_tanimoto", 0.6)),
                library_size=int(parameters.get("library_size", 200_000)),
            )
        )

    def __call__(self, smiles_iter: Sequence[str]) -> ComponentResults:
        scores: list[float] = []
        meta: list[dict[str, object]] = []
        for smi in smiles_iter:
            sim = max_tanimoto_to_vendor(smi, self.pool)
            reward = 1.0 if sim <= self.params.max_tanimoto else max(0.0, 1.0 - sim)
            scores.append(reward)
            meta.append({"max_tanimoto_to_vendor": sim})
        return ComponentResults(
            scores=np.asarray(scores, dtype=np.float32), metadata=tuple(meta)
        )
