"""Featurization for the QSAR panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from mc4gen.utils.fingerprints import morgan_count, rdkit_descriptors_2d


@dataclass(frozen=True, slots=True)
class FeaturizerConfig:
    radius: int = 2
    n_bits: int = 2048
    include_descriptors: bool = True


def featurize(smiles: str, config: FeaturizerConfig | None = None) -> NDArray[np.float32] | None:
    cfg = config or FeaturizerConfig()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = morgan_count(mol, radius=cfg.radius, n_bits=cfg.n_bits).astype(np.float32)
    if cfg.include_descriptors:
        desc = rdkit_descriptors_2d(mol)
        return np.concatenate([fp, np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)])
    return fp


def featurize_batch(
    smiles: Iterable[str],
    config: FeaturizerConfig | None = None,
) -> tuple[NDArray[np.float32], list[int]]:
    cfg = config or FeaturizerConfig()
    vecs: list[NDArray[np.float32]] = []
    valid: list[int] = []
    for i, s in enumerate(smiles):
        v = featurize(s, cfg)
        if v is None:
            continue
        vecs.append(v)
        valid.append(i)
    if not vecs:
        dim = cfg.n_bits + (208 if cfg.include_descriptors else 0)
        return np.zeros((0, dim), dtype=np.float32), []
    return np.stack(vecs), valid
