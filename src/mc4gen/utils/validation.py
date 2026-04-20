"""Lipinski, Veber, and MOSES-style drug-likeness checks."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


@dataclass(frozen=True, slots=True)
class DrugLikenessReport:
    mw: float
    logp: float
    h_donors: int
    h_acceptors: int
    rotatable: int
    tpsa: float
    rings: int
    violations: tuple[str, ...]

    @property
    def passes_lipinski(self) -> bool:
        return len(
            tuple(v for v in self.violations if v.startswith("Lipinski"))
        ) <= 1

    @property
    def passes_veber(self) -> bool:
        return not any(v.startswith("Veber") for v in self.violations)


def evaluate(smiles: str) -> DrugLikenessReport | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    rings = Lipinski.RingCount(mol)
    violations: list[str] = []
    if mw > 500:
        violations.append("Lipinski:MW>500")
    if logp > 5:
        violations.append("Lipinski:logP>5")
    if hbd > 5:
        violations.append("Lipinski:HBD>5")
    if hba > 10:
        violations.append("Lipinski:HBA>10")
    if rot > 10:
        violations.append("Veber:RotBonds>10")
    if tpsa > 140:
        violations.append("Veber:TPSA>140")
    return DrugLikenessReport(
        mw=mw,
        logp=logp,
        h_donors=hbd,
        h_acceptors=hba,
        rotatable=rot,
        tpsa=tpsa,
        rings=rings,
        violations=tuple(violations),
    )


def passes_drug_likeness(smiles: str, *, max_violations: int = 1) -> bool:
    report = evaluate(smiles)
    if report is None:
        return False
    return len(report.violations) <= max_violations
