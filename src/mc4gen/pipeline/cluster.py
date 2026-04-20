"""Butina clustering on Morgan-2 fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina


@dataclass(frozen=True, slots=True)
class ClusterResult:
    members: tuple[int, ...]
    representative: int


def cluster_smiles(
    smiles: Sequence[str],
    *,
    cutoff: float = 0.4,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[ClusterResult]:
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(gen.GetFingerprint(mol))
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    if not valid_indices:
        return []

    dists: list[float] = []
    for i, idx_i in enumerate(valid_indices):
        for idx_j in valid_indices[:i]:
            sim = _tanimoto(fps[idx_i], fps[idx_j])
            dists.append(1.0 - sim)
    clusters = Butina.ClusterData(dists, len(valid_indices), cutoff, isDistData=True)
    out: list[ClusterResult] = []
    for cluster in clusters:
        members = tuple(valid_indices[m] for m in cluster)
        out.append(ClusterResult(members=members, representative=members[0]))
    return out


def _tanimoto(a, b) -> float:
    from rdkit import DataStructs

    return float(DataStructs.TanimotoSimilarity(a, b))


def cluster_dataframe(smiles: Sequence[str], *, cutoff: float = 0.4) -> NDArray[np.int64]:
    """Return a per-molecule cluster index (``-1`` for invalid)."""
    results = cluster_smiles(smiles, cutoff=cutoff)
    labels = np.full(len(smiles), -1, dtype=np.int64)
    for cluster_id, cluster in enumerate(results):
        for idx in cluster.members:
            labels[idx] = cluster_id
    return labels
