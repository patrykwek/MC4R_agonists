"""Machine-readable pipeline assumptions.

Each :class:`Assumption` carries testable implications and untestable components
(flagged). The manuscript auto-generates a limitations section from this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Assumption:
    key: str
    statement: str
    testable_implications: tuple[str, ...]
    untestable_components: tuple[str, ...] = field(default_factory=tuple)
    mitigations: tuple[str, ...] = field(default_factory=tuple)


VINA_AFFINITY_PROXY = Assumption(
    key="vina_affinity_proxy",
    statement=(
        "AutoDock Vina docking scores correlate with experimental binding affinity "
        "sufficiently to be useful as a reinforcement-learning reward signal."
    ),
    testable_implications=(
        "Spearman rho between Vina score and experimental pKi on held-out MC4R actives > 0.3.",
        "Self-redock of setmelanotide into 7PIU yields heavy-atom RMSD <= 2.5 A.",
        "Cross-dock of NDP-alpha-MSH into 7AUE yields heavy-atom RMSD <= 3.5 A.",
    ),
    untestable_components=(
        "Absolute kcal/mol values cannot be interpreted as binding free energies.",
    ),
    mitigations=(
        "Ensemble over 5 independent cryo-EM structures, take the median.",
        "Report Spearman rho prominently in the manuscript.",
    ),
)

QSAR_APPLICABILITY_DOMAIN = Assumption(
    key="qsar_applicability_domain",
    statement=(
        "Predicted pKi values for molecules inside the applicability domain are more reliable "
        "than those outside."
    ),
    testable_implications=(
        "Scaffold-split hold-out R^2 >= 0.5 for MC4R.",
        "AD-gated predictions have lower RMSE than non-gated predictions on the same split.",
    ),
    untestable_components=(
        "The 'true' applicability domain for a de novo generated molecule is never fully known.",
    ),
    mitigations=(
        "Report leverage h* = 3p/n and Tanimoto-to-nearest-neighbor cutoffs.",
        "Exclude OOD candidates from headline claims.",
    ),
)

SELECTIVITY_INDEPENDENCE = Assumption(
    key="selectivity_independence",
    statement=(
        "Errors in the MC1R, MC3R, MC4R, and MC5R QSAR ensembles are approximately independent, "
        "so the variance of pKi_MC4R - pKi_MCi decomposes as the sum of component variances."
    ),
    testable_implications=(
        "Residual correlations between panel members on a shared held-out compound set are < 0.3.",
    ),
    untestable_components=(
        "No dataset spans all four receptors densely enough to fully validate independence.",
    ),
    mitigations=(
        "Flag the untestable component in the manuscript; report residual correlations we can observe.",
    ),
)

POSE_TO_AGONISM = Assumption(
    key="pose_to_agonism",
    statement=(
        "A molecule with a low-energy docked pose that recapitulates the setmelanotide orthosteric "
        "contacts (Ca2+ coordination, Glu100 H-bond, hydrophobic cluster) is more likely to be a "
        "functional agonist than a random scaffold."
    ),
    testable_implications=(),
    untestable_components=(
        "Functional agonism requires cell-based cAMP or beta-arrestin assays, which are out of scope.",
    ),
    mitigations=(
        "State prominently that the project produces computational hypotheses, not validated drugs.",
    ),
)

BIAS_UNPREDICTED = Assumption(
    key="bias_unpredicted",
    statement=(
        "Setmelanotide's Gq/11 vs. Gs signaling bias is a functional property not captured by any "
        "structure-based scoring term in this pipeline."
    ),
    testable_implications=(),
    untestable_components=(
        "Biased agonism requires orthogonal signaling assays.",
    ),
    mitigations=(
        "Do not claim to predict bias anywhere in manuscript or Streamlit app.",
    ),
)

ALL_ASSUMPTIONS: tuple[Assumption, ...] = (
    VINA_AFFINITY_PROXY,
    QSAR_APPLICABILITY_DOMAIN,
    SELECTIVITY_INDEPENDENCE,
    POSE_TO_AGONISM,
    BIAS_UNPREDICTED,
)
