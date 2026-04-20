r"""Applicability domain gating.

Two complementary tests are applied per query:

1. **Leverage** :math:`h^* = 3p/n` from the training feature matrix.
2. **Tanimoto distance** to the nearest training neighbour must be below
   the 95th percentile of training self-distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mc4gen.utils.fingerprints import bulk_tanimoto


@dataclass(frozen=True, slots=True)
class DomainModel:
    hat_matrix_diag: NDArray[np.float32]
    leverage_cutoff: float
    tanimoto_self_cutoff: float
    train_fps: NDArray[np.int32]

    def in_domain(self, query_fp: NDArray[np.int32], query_feat: NDArray[np.float32]) -> bool:
        sims = bulk_tanimoto(query_fp, self.train_fps)
        if sims.size == 0:
            return False
        nearest = float(sims.max())
        if nearest < self.tanimoto_self_cutoff:
            return False
        leverage = float(query_feat @ query_feat.T) / max(self.hat_matrix_diag.shape[0], 1)
        return leverage <= self.leverage_cutoff

    def coverage(
        self,
        query_fps: NDArray[np.int32],
        query_feats: NDArray[np.float32],
    ) -> float:
        if query_fps.shape[0] == 0:
            return 0.0
        in_dom = sum(
            int(self.in_domain(fp, feat))
            for fp, feat in zip(query_fps, query_feats, strict=True)
        )
        return in_dom / query_fps.shape[0]


def fit(
    train_feats: NDArray[np.float32],
    train_fps: NDArray[np.int32],
) -> DomainModel:
    """Fit an applicability-domain model from training features + fingerprints."""
    n = train_feats.shape[0]
    p = train_feats.shape[1]
    leverage_cutoff = 3.0 * p / max(n, 1)
    diag = np.einsum("ij,ij->i", train_feats, train_feats).astype(np.float32)

    self_sims: list[float] = []
    for i in range(min(n, 512)):
        sims = bulk_tanimoto(train_fps[i], np.delete(train_fps, i, axis=0))
        if sims.size == 0:
            continue
        self_sims.append(float(sims.max()))
    tanimoto_cutoff = float(np.quantile(self_sims, 0.05)) if self_sims else 0.0
    return DomainModel(
        hat_matrix_diag=diag,
        leverage_cutoff=leverage_cutoff,
        tanimoto_self_cutoff=tanimoto_cutoff,
        train_fps=train_fps,
    )
