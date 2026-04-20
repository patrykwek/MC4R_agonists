"""Ligand preparation: protonation (dimorphite-dl) + conformer + PDBQT (meeko)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mc4gen._logging import get_logger
from mc4gen.utils.smiles import canonicalize, embed_conformer

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LigandPrepResult:
    smiles: str
    pdbqt_path: Path
    protonation_smiles: str
    rdkit_mol_block: str


def enumerate_protonations(smiles: str, *, ph: float = 7.4) -> list[str]:
    """Enumerate microstates at ``ph`` using dimorphite-dl."""
    from dimorphite_dl import DimorphiteDL

    dm = DimorphiteDL(
        min_ph=ph - 0.5,
        max_ph=ph + 0.5,
        max_variants=4,
        label_states=False,
        pka_precision=1.0,
    )
    states = dm.protonate(smiles)
    cleaned: list[str] = []
    for s in states:
        canon = canonicalize(s)
        if canon and canon not in cleaned:
            cleaned.append(canon)
    if not cleaned:
        canon = canonicalize(smiles)
        if canon is not None:
            cleaned.append(canon)
    return cleaned


def prepare_ligand(smiles: str, out_dir: Path, *, ph: float = 7.4) -> LigandPrepResult | None:
    """Return a PDBQT-ready ligand for the lowest-index protonation state."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem

    out_dir.mkdir(parents=True, exist_ok=True)
    states = enumerate_protonations(smiles, ph=ph)
    if not states:
        return None
    prot_smi = states[0]
    mol = embed_conformer(prot_smi)
    if mol is None:
        return None

    prep = MoleculePreparation()
    prep.prepare(mol)
    pdbqt_strings = PDBQTWriterLegacy.write_string(prep.setup)
    pdbqt_text = pdbqt_strings[0] if isinstance(pdbqt_strings, list) else pdbqt_strings

    safe = "".join(c for c in smiles if c.isalnum())[:24] or "lig"
    pdbqt_path = out_dir / f"{safe}.pdbqt"
    pdbqt_path.write_text(pdbqt_text, encoding="utf-8")
    return LigandPrepResult(
        smiles=smiles,
        pdbqt_path=pdbqt_path,
        protonation_smiles=prot_smi,
        rdkit_mol_block=Chem.MolToMolBlock(mol),
    )
