"""PLIP + ProLIF wrappers for post-dock interaction fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class InteractionFingerprint:
    hbond_donor_residues: tuple[str, ...]
    hbond_acceptor_residues: tuple[str, ...]
    hydrophobic_residues: tuple[str, ...]
    pi_stacking_residues: tuple[str, ...]
    cation_pi_residues: tuple[str, ...]
    salt_bridge_residues: tuple[str, ...]
    metal_coordinated: bool

    def flat(self) -> dict[str, object]:
        return {
            "hbd": list(self.hbond_donor_residues),
            "hba": list(self.hbond_acceptor_residues),
            "hydrophobic": list(self.hydrophobic_residues),
            "pi_stack": list(self.pi_stacking_residues),
            "cation_pi": list(self.cation_pi_residues),
            "salt_bridge": list(self.salt_bridge_residues),
            "metal": self.metal_coordinated,
        }


def _extract(report: object, attribute: str) -> tuple[str, ...]:
    group = getattr(report, attribute, None)
    if group is None:
        return ()
    residues: list[str] = []
    for inter in group:
        res_name = getattr(inter, "restype", getattr(inter, "resname", ""))
        res_num = getattr(inter, "resnr", "")
        chain = getattr(inter, "reschain", "")
        residues.append(f"{res_name}{res_num}{chain}")
    return tuple(residues)


def analyze_plip(complex_pdb: Path) -> InteractionFingerprint:
    """Run PLIP on a protein-ligand complex PDB file."""
    from plip.structure.preparation import PDBComplex

    complex_ = PDBComplex()
    complex_.load_pdb(str(complex_pdb))
    complex_.analyze()

    reports = list(complex_.interaction_sets.values())
    if not reports:
        return InteractionFingerprint((), (), (), (), (), (), False)
    report = reports[0]
    return InteractionFingerprint(
        hbond_donor_residues=_extract(report, "hbonds_ldon"),
        hbond_acceptor_residues=_extract(report, "hbonds_pdon"),
        hydrophobic_residues=_extract(report, "hydrophobic_contacts"),
        pi_stacking_residues=_extract(report, "pistacking"),
        cation_pi_residues=_extract(report, "pication_laro"),
        salt_bridge_residues=_extract(report, "saltbridge_lneg"),
        metal_coordinated=bool(getattr(report, "metal_complexes", [])),
    )


def tanimoto_fingerprint(a: InteractionFingerprint, b: InteractionFingerprint) -> float:
    """Tanimoto over interaction-residue sets."""
    sets_a = set(a.hbond_donor_residues + a.hbond_acceptor_residues + a.hydrophobic_residues)
    sets_b = set(b.hbond_donor_residues + b.hbond_acceptor_residues + b.hydrophobic_residues)
    if not sets_a and not sets_b:
        return 1.0
    inter = len(sets_a & sets_b)
    union = len(sets_a | sets_b)
    return inter / union if union else 0.0


def residue_coverage(ifps: Iterable[InteractionFingerprint]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ifp in ifps:
        for res in set(
            ifp.hbond_donor_residues
            + ifp.hbond_acceptor_residues
            + ifp.hydrophobic_residues
            + ifp.pi_stacking_residues
        ):
            counts[res] = counts.get(res, 0) + 1
    return counts
