"""SMILES canonicalization, stereo enumeration, and sanity checks."""

from __future__ import annotations

from typing import Iterable, Iterator

from rdkit import Chem
from rdkit.Chem import AllChem, EnumerateStereoisomers


def canonicalize(smiles: str, *, isomeric: bool = True) -> str | None:
    """Return canonical SMILES or ``None`` if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def canonicalize_batch(smiles: Iterable[str]) -> list[str | None]:
    return [canonicalize(s) for s in smiles]


def enumerate_stereoisomers(smiles: str, *, max_isomers: int = 16) -> list[str]:
    """Enumerate up to ``max_isomers`` stereoisomers via RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    opts = EnumerateStereoisomers.StereoEnumerationOptions(
        maxIsomers=max_isomers, onlyUnassigned=True, unique=True
    )
    isomers = EnumerateStereoisomers.EnumerateStereoisomers(mol, options=opts)
    return [Chem.MolToSmiles(m, isomericSmiles=True) for m in isomers]


def embed_conformer(smiles: str, *, seed: int = 0xC0FFEE) -> Chem.Mol | None:
    """Embed a single 3D conformer using ETKDGv3 and MMFF94 minimization."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:  # pragma: no cover - field-force failures are rare
        return None
    return mol


def iter_valid(smiles: Iterable[str]) -> Iterator[str]:
    for s in smiles:
        canon = canonicalize(s)
        if canon is not None:
            yield canon


def is_valid(smiles: str) -> bool:
    return canonicalize(smiles) is not None
