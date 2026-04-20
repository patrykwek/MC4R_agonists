r"""Formal estimands for the mc4gen pipeline.

Each estimand is a frozen dataclass with a LaTeX definition and the machine
operationalization that realizes it in code.

.. math::

   \hat{pK}_i^{\mathrm{MC4R}}(x) = f_\theta(x)

   S_{\mathrm{MC4R/MC}i}(x) = \hat{pK}_i^{\mathrm{MC4R}}(x) - \hat{pK}_i^{\mathrm{MC}i}(x)

   D(x) = \prod_k S_k(x)^{w_k}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Estimand:
    name: str
    latex: str
    unit: str
    operationalization: str


@dataclass(frozen=True, slots=True)
class DesirabilityWeights:
    r"""Geometric-mean weights; normalized so that ``sum(weights)=1``."""

    predicted_pki_mc4r: float = 0.30
    selectivity_mc1r: float = 0.20
    selectivity_mc3r: float = 0.05
    selectivity_mc5r: float = 0.05
    vina_score: float = 0.20
    ca_coordination: float = 0.05
    ra_score: float = 0.05
    novelty: float = 0.05
    qed: float = 0.05

    def as_dict(self) -> Mapping[str, float]:
        return {
            "predicted_pki_mc4r": self.predicted_pki_mc4r,
            "selectivity_mc1r": self.selectivity_mc1r,
            "selectivity_mc3r": self.selectivity_mc3r,
            "selectivity_mc5r": self.selectivity_mc5r,
            "vina_score": self.vina_score,
            "ca_coordination": self.ca_coordination,
            "ra_score": self.ra_score,
            "novelty": self.novelty,
            "qed": self.qed,
        }

    def validate(self) -> None:
        total = sum(self.as_dict().values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"Desirability weights must sum to 1.0 (got {total:.4f})")


PREDICTED_PKI_MC4R = Estimand(
    name="predicted_pki_mc4r",
    latex=r"\hat{pK}_i^{\mathrm{MC4R}}(x)",
    unit="log-molar",
    operationalization="5-fold CV ensemble of RF/LGBM/SVR/Ridge/MLP on ChEMBL MC4R data.",
)

SELECTIVITY_MC4R_MC1R = Estimand(
    name="selectivity_mc4r_mc1r",
    latex=r"S_{\mathrm{MC4R/MC1R}}(x)",
    unit="log-molar (difference)",
    operationalization="pKi_MC4R - pKi_MC1R using two independent QSAR ensembles.",
)

SELECTIVITY_MC4R_MC3R = Estimand(
    name="selectivity_mc4r_mc3r",
    latex=r"S_{\mathrm{MC4R/MC3R}}(x)",
    unit="log-molar (difference)",
    operationalization="pKi_MC4R - pKi_MC3R.",
)

SELECTIVITY_MC4R_MC5R = Estimand(
    name="selectivity_mc4r_mc5r",
    latex=r"S_{\mathrm{MC4R/MC5R}}(x)",
    unit="log-molar (difference)",
    operationalization="pKi_MC4R - pKi_MC5R.",
)

VINA_SCORE_ENSEMBLE = Estimand(
    name="vina_score_ensemble",
    latex=r"V_{\mathrm{ens}}(x) = \mathrm{median}_{j \in J} V_j(x)",
    unit="kcal/mol",
    operationalization="AutoDock Vina score against each of 5 MC4R structures; median taken.",
)

DESIRABILITY = Estimand(
    name="desirability",
    latex=r"D(x) = \prod_k S_k(x)^{w_k}",
    unit="dimensionless in [0, 1]",
    operationalization="Bickerton-style weighted geometric mean of rescaled component scores.",
)

ALL_ESTIMANDS: tuple[Estimand, ...] = (
    PREDICTED_PKI_MC4R,
    SELECTIVITY_MC4R_MC1R,
    SELECTIVITY_MC4R_MC3R,
    SELECTIVITY_MC4R_MC5R,
    VINA_SCORE_ENSEMBLE,
    DESIRABILITY,
)
