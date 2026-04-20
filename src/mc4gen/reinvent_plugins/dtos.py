"""Data-transfer objects for the REINVENT 4 scoring components.

REINVENT 4 scoring components receive a list of SMILES and must return an
object with a ``scores`` attribute (a ``numpy`` array or list of floats) and an
optional ``metadata`` attribute (arbitrary dict per molecule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ComponentResults:
    scores: NDArray[np.float32]
    metadata: Sequence[Mapping[str, object]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.metadata and len(self.metadata) != self.scores.shape[0]:
            raise ValueError(
                f"metadata length {len(self.metadata)} != scores length {self.scores.shape[0]}"
            )

    @classmethod
    def from_floats(
        cls,
        values: Sequence[float],
        metadata: Sequence[Mapping[str, object]] | None = None,
    ) -> "ComponentResults":
        arr = np.asarray(values, dtype=np.float32)
        return cls(scores=arr, metadata=tuple(metadata) if metadata else ())


@dataclass(frozen=True, slots=True)
class Parameters:
    name: str
    weight: float = 1.0
