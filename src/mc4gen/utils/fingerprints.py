"""Fingerprint utilities (Morgan / ECFP, RDKit 2D descriptors)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_MORGAN_COUNT_GEN = GetMorganGenerator(radius=2, fpSize=2048, countSimulation=False)
_MORGAN_BIT_GEN = GetMorganGenerator(radius=2, fpSize=2048)
_ECFP4_GEN = GetMorganGenerator(radius=2, fpSize=2048)


def morgan_count(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> NDArray[np.int32]:
    """Return a count Morgan fingerprint of dimension ``n_bits``."""
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetCountFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_bit(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> NDArray[np.uint8]:
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def ecfp4(smiles: str) -> NDArray[np.int32] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_count(mol, radius=2, n_bits=2048)


def tanimoto(a: NDArray[np.int32], b: NDArray[np.int32]) -> float:
    """Tanimoto similarity on count fingerprints (Rogers-Tanimoto for counts)."""
    num = float(np.minimum(a, b).sum())
    den = float(np.maximum(a, b).sum())
    return num / den if den > 0 else 0.0


def bulk_tanimoto(query: NDArray[np.int32], pool: NDArray[np.int32]) -> NDArray[np.float32]:
    """Compute Tanimoto of ``query`` against ``pool`` (shape ``(n, d)``)."""
    num = np.minimum(query[None, :], pool).sum(axis=1).astype(np.float32)
    den = np.maximum(query[None, :], pool).sum(axis=1).astype(np.float32)
    return np.where(den > 0, num / den, 0.0)


_DESC_NAMES: tuple[str, ...] = tuple(d[0] for d in Descriptors.descList)


def rdkit_descriptors_2d(mol: Chem.Mol) -> NDArray[np.float32]:
    """Return the ~208-dim RDKit 2D descriptor vector."""
    values = [fn(mol) for _, fn in Descriptors.descList]
    return np.asarray(values, dtype=np.float32)


def descriptor_names() -> tuple[str, ...]:
    return _DESC_NAMES


def featurize_batch(
    smiles_iter: Iterable[str],
    *,
    include_descriptors: bool = True,
) -> NDArray[np.float32]:
    """Concatenated ECFP4 (2048) + 2D descriptors (208) -> 2256-dim feature matrix."""
    rows: list[NDArray[np.float32]] = []
    for s in smiles_iter:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = morgan_count(mol, radius=2, n_bits=2048).astype(np.float32)
        if include_descriptors:
            desc = rdkit_descriptors_2d(mol)
            rows.append(np.concatenate([fp, desc]))
        else:
            rows.append(fp)
    return np.stack(rows) if rows else np.zeros((0, 2048), dtype=np.float32)
