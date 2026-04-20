"""MC4R vs. MC1/3/5R selectivity scoring component."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mc4gen._logging import get_logger
from mc4gen.qsar.panel import MelanocortinPanel
from mc4gen.reinvent_plugins.dtos import ComponentResults

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SelectivityParameters:
    panel_path: Path
    counter_targets: tuple[str, ...] = ("MC1R", "MC3R", "MC5R")
    min_pki_mc4r: float = 7.0
    min_selectivity_mc1r: float = 2.0
    min_selectivity_mc3r: float = 0.0
    min_selectivity_mc5r: float = 0.0
    weight_mc4r_potency: float = 0.5
    weight_selectivity: float = 0.5


def _sigmoid(x: float, center: float, slope: float = 0.75) -> float:
    import math

    try:
        return 1.0 / (1.0 + math.exp(-(x - center) / slope))
    except OverflowError:
        return 1.0 if x > center else 0.0


@dataclass
class MelanocortinSelectivityPanel:
    params: SelectivityParameters
    panel: MelanocortinPanel = field(init=False)
    name: str = "melanocortin_selectivity"

    def __post_init__(self) -> None:
        self.panel = MelanocortinPanel.load(self.params.panel_path)

    @classmethod
    def from_parameters(cls, parameters: dict[str, object]) -> "MelanocortinSelectivityPanel":
        return cls(
            params=SelectivityParameters(
                panel_path=Path(str(parameters["panel_path"])),
                counter_targets=tuple(
                    parameters.get("counter_targets", ("MC1R", "MC3R", "MC5R"))
                ),
                min_pki_mc4r=float(parameters.get("min_pki_mc4r", 7.0)),
                min_selectivity_mc1r=float(parameters.get("min_selectivity_mc1r", 2.0)),
                min_selectivity_mc3r=float(parameters.get("min_selectivity_mc3r", 0.0)),
                min_selectivity_mc5r=float(parameters.get("min_selectivity_mc5r", 0.0)),
                weight_mc4r_potency=float(parameters.get("weight_mc4r_potency", 0.5)),
                weight_selectivity=float(parameters.get("weight_selectivity", 0.5)),
            )
        )

    def __call__(self, smiles_iter: Sequence[str]) -> ComponentResults:
        rewards: list[float] = []
        meta: list[dict[str, object]] = []
        thresholds = {
            "MC1R": self.params.min_selectivity_mc1r,
            "MC3R": self.params.min_selectivity_mc3r,
            "MC5R": self.params.min_selectivity_mc5r,
        }
        for smi in smiles_iter:
            pred = self.panel.predict(smi)
            if pred is None or "MC4R" not in pred.pki:
                rewards.append(0.0)
                meta.append({"reason": "prediction_unavailable"})
                continue
            mc4r = pred.pki["MC4R"]
            potency_term = _sigmoid(mc4r, self.params.min_pki_mc4r)
            sel_terms: list[float] = []
            for counter in self.params.counter_targets:
                if counter not in pred.pki:
                    continue
                delta = mc4r - pred.pki[counter]
                sel_terms.append(_sigmoid(delta, thresholds[counter]))
            sel_mean = float(np.mean(sel_terms)) if sel_terms else 0.0
            reward = (
                self.params.weight_mc4r_potency * potency_term
                + self.params.weight_selectivity * sel_mean
            )
            rewards.append(float(reward))
            meta.append(
                {
                    "pki": dict(pred.pki),
                    "uncertainty": dict(pred.uncertainty),
                    "in_domain": dict(pred.in_domain),
                    "selectivity": pred.selectivity(),
                }
            )
        return ComponentResults(
            scores=np.asarray(rewards, dtype=np.float32), metadata=tuple(meta)
        )
