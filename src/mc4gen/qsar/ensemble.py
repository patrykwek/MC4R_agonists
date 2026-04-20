r"""Per-receptor ensemble regressor.

Each ensemble has :math:`K = 5` cross-validation folds; the per-query prediction
is the mean across folds, and the uncertainty :math:`\hat\sigma` is the fold
standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.stats
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from mc4gen._logging import get_logger
from mc4gen.qsar.applicability_domain import DomainModel, fit as fit_domain
from mc4gen.qsar.models import DEFAULT_SPECS, ModelSpec, Regressor, build

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FoldResult:
    spec: ModelSpec
    fold_index: int
    rmse: float
    r2: float
    spearman: float


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    ensemble: "Ensemble"
    per_fold: tuple[FoldResult, ...]
    holdout_rmse: float
    holdout_r2: float
    holdout_spearman: float


@dataclass
class Ensemble:
    models: list[Regressor]
    specs: list[ModelSpec]
    domain: DomainModel

    def predict(self, X: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        preds = np.stack([np.asarray(m.predict(X), dtype=np.float32) for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0, ddof=1)


def _score_fold(
    spec: ModelSpec,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    X_val: NDArray[np.float32],
    y_val: NDArray[np.float32],
    fold_index: int,
) -> tuple[Regressor, FoldResult]:
    model = build(spec)
    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_val), dtype=np.float32)
    rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
    ss_res = float(np.sum((y_val - preds) ** 2))
    ss_tot = float(np.sum((y_val - y_val.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    spearman = float(scipy.stats.spearmanr(preds, y_val).statistic)
    return model, FoldResult(spec=spec, fold_index=fold_index, rmse=rmse, r2=r2, spearman=spearman)


def train_ensemble(
    X: NDArray[np.float32],
    y: NDArray[np.float32],
    fingerprints: NDArray[np.int32],
    *,
    specs: Sequence[ModelSpec] | None = None,
    n_splits: int = 5,
    random_state: int = 0,
) -> EnsembleResult:
    """Train a K-fold ensemble with one model of each spec per fold."""
    use_specs = list(specs) if specs is not None else list(DEFAULT_SPECS)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_models: list[Regressor] = []
    all_specs: list[ModelSpec] = []
    fold_results: list[FoldResult] = []

    for fold_index, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        for spec in use_specs:
            model, fr = _score_fold(spec, X_train, y_train, X_val, y_val, fold_index)
            all_models.append(model)
            all_specs.append(spec)
            fold_results.append(fr)

    holdout_rmse = float(np.mean([fr.rmse for fr in fold_results]))
    holdout_r2 = float(np.mean([fr.r2 for fr in fold_results]))
    holdout_spearman = float(np.mean([fr.spearman for fr in fold_results]))

    domain = fit_domain(X, fingerprints)
    ensemble = Ensemble(models=all_models, specs=all_specs, domain=domain)
    return EnsembleResult(
        ensemble=ensemble,
        per_fold=tuple(fold_results),
        holdout_rmse=holdout_rmse,
        holdout_r2=holdout_r2,
        holdout_spearman=holdout_spearman,
    )
