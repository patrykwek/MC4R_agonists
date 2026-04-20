"""Auto-generate a natural-language rationale per candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mc4gen.pipeline.prioritize import Candidate


@dataclass(frozen=True, slots=True)
class Rationale:
    candidate: Candidate
    summary: str
    bullets: tuple[str, ...]


_SUMMARY = (
    "Candidate {short} scored {vina:.2f} kcal/mol (ensemble median) with QSAR-predicted "
    "pKi_MC4R = {pki:.2f} and MC4R/MC1R selectivity = {sel:.2f} log units. "
    "Chemotype-novelty = {novel:.2f}, nearest vendor Tanimoto = {vendor:.2f}, RAscore = {ra:.2f}."
)


def _short(smiles: str, length: int = 28) -> str:
    return smiles if len(smiles) <= length else smiles[: length - 3] + "..."


def generate(
    candidate: Candidate,
    *,
    md_metrics: Mapping[str, float] | None = None,
) -> Rationale:
    bullets: list[str] = [
        f"Ensemble Vina score over 5 MC4R structures: {candidate.vina_score:.2f} kcal/mol.",
        f"QSAR pKi(MC4R) = {candidate.predicted_pki_mc4r:.2f}; "
        f"MC1R/MC3R/MC5R selectivities = "
        f"{candidate.selectivity_mc1r:.2f}/{candidate.selectivity_mc3r:.2f}/"
        f"{candidate.selectivity_mc5r:.2f}.",
        f"RAscore = {candidate.ra_score:.2f} (synthetic accessibility).",
        f"Vendor Tanimoto = {candidate.vendor_tanimoto:.2f} (nearest ZINC20/Enamine REAL).",
        f"Butina cluster id = {candidate.cluster_id}.",
    ]
    if md_metrics:
        bullets.append(
            "MD stability: ligand RMSD = "
            f"{md_metrics.get('ligand_rmsd_mean', float('nan')):.2f} Å; "
            f"Ca2+ contact fraction = {md_metrics.get('ca_contact_fraction', float('nan')):.2f}."
        )
    summary = _SUMMARY.format(
        short=_short(candidate.smiles),
        vina=candidate.vina_score,
        pki=candidate.predicted_pki_mc4r,
        sel=candidate.selectivity_mc1r,
        novel=candidate.chemotype_novelty,
        vendor=candidate.vendor_tanimoto,
        ra=candidate.ra_score,
    )
    return Rationale(candidate=candidate, summary=summary, bullets=tuple(bullets))
