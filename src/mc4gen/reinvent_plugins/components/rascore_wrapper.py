"""RAscore synthetic-accessibility wrapper (Thakkar et al. 2021)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mc4gen._logging import get_logger
from mc4gen.reinvent_plugins.dtos import ComponentResults

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RAScoreParameters:
    model_type: str = "DNN"
    weights_path: Path | None = None
    threshold: float = 0.8


@dataclass
class RAScoreComponent:
    params: RAScoreParameters = field(default_factory=RAScoreParameters)
    name: str = "ra_score"
    _scorer: object | None = field(init=False, default=None)

    def _load_scorer(self) -> object:
        if self._scorer is not None:
            return self._scorer
        from RAscore.RAscore_NN import RAScorerNN
        from RAscore.RAscore_XGB import RAScorerXGB

        if self.params.model_type.upper().startswith("X"):
            scorer: object = (
                RAScorerXGB(str(self.params.weights_path))
                if self.params.weights_path
                else RAScorerXGB()
            )
        else:
            scorer = (
                RAScorerNN(str(self.params.weights_path))
                if self.params.weights_path
                else RAScorerNN()
            )
        self._scorer = scorer
        return scorer

    @classmethod
    def from_parameters(cls, parameters: dict[str, object]) -> "RAScoreComponent":
        weights = parameters.get("weights_path")
        return cls(
            params=RAScoreParameters(
                model_type=str(parameters.get("model_type", "DNN")),
                weights_path=Path(str(weights)) if weights else None,
                threshold=float(parameters.get("threshold", 0.8)),
            )
        )

    def __call__(self, smiles_iter: Sequence[str]) -> ComponentResults:
        scorer = self._load_scorer()
        scores: list[float] = []
        meta: list[dict[str, object]] = []
        for smi in smiles_iter:
            try:
                value = float(scorer.predict(smi))  # type: ignore[attr-defined]
            except Exception as err:  # pragma: no cover - external lib failure
                log.warning("RAscore failed for %s: %s", smi[:40], err)
                value = 0.0
            scores.append(value)
            meta.append({"ra_score": value, "passes_threshold": value >= self.params.threshold})
        return ComponentResults(
            scores=np.asarray(scores, dtype=np.float32), metadata=tuple(meta)
        )
