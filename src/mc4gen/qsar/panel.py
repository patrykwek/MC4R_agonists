"""Unified MC1/3/4/5R QSAR panel: training, persistence, prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from mc4gen._logging import get_logger
from mc4gen.data.chembl import ChEMBLRecord, load_panel, panel_to_dataframe
from mc4gen.qsar.ensemble import Ensemble, EnsembleResult, train_ensemble
from mc4gen.qsar.features import FeaturizerConfig, featurize_batch
from mc4gen.utils.fingerprints import morgan_count

log = get_logger(__name__)

RECEPTORS: tuple[str, ...] = ("MC1R", "MC3R", "MC4R", "MC5R")


@dataclass(frozen=True, slots=True)
class PanelPrediction:
    smiles: str
    pki: Mapping[str, float]
    uncertainty: Mapping[str, float]
    in_domain: Mapping[str, bool]

    def selectivity(self, primary: str = "MC4R") -> Mapping[str, float]:
        return {other: self.pki[primary] - self.pki[other] for other in self.pki if other != primary}


@dataclass
class TrainingArtifacts:
    config: FeaturizerConfig
    per_receptor: dict[str, EnsembleResult]
    holdout_summary: pd.DataFrame


def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def _scaffold_split(
    records: Sequence[ChEMBLRecord],
    *,
    holdout_fraction: float = 0.15,
    random_state: int = 0,
) -> tuple[list[ChEMBLRecord], list[ChEMBLRecord]]:
    by_scaffold: dict[str, list[ChEMBLRecord]] = {}
    for r in records:
        by_scaffold.setdefault(_scaffold(r.smiles), []).append(r)
    scaffolds = sorted(by_scaffold, key=lambda s: len(by_scaffold[s]))
    rng = np.random.default_rng(random_state)
    rng.shuffle(scaffolds)
    n_total = len(records)
    target_holdout = int(n_total * holdout_fraction)
    holdout: list[ChEMBLRecord] = []
    train: list[ChEMBLRecord] = []
    for scaf in scaffolds:
        bucket = by_scaffold[scaf]
        if len(holdout) + len(bucket) <= target_holdout:
            holdout.extend(bucket)
        else:
            train.extend(bucket)
    return train, holdout


@dataclass
class MelanocortinPanel:
    ensembles: dict[str, Ensemble]
    feature_config: FeaturizerConfig

    def predict(self, smiles: str) -> PanelPrediction | None:
        X, valid = featurize_batch([smiles], self.feature_config)
        if not valid:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = morgan_count(mol)
        pki: dict[str, float] = {}
        unc: dict[str, float] = {}
        in_dom: dict[str, bool] = {}
        for receptor, ens in self.ensembles.items():
            mean, std = ens.predict(X)
            pki[receptor] = float(mean[0])
            unc[receptor] = float(std[0])
            in_dom[receptor] = ens.domain.in_domain(fp, X[0])
        return PanelPrediction(smiles=smiles, pki=pki, uncertainty=unc, in_domain=in_dom)

    def predict_batch(self, smiles: Sequence[str]) -> list[PanelPrediction | None]:
        return [self.predict(s) for s in smiles]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "ensembles": self.ensembles,
                "feature_config": self.feature_config,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "MelanocortinPanel":
        blob = joblib.load(path)
        return cls(ensembles=blob["ensembles"], feature_config=blob["feature_config"])


def train_panel(
    *,
    feature_config: FeaturizerConfig | None = None,
    save_to: Path | None = None,
) -> tuple[MelanocortinPanel, TrainingArtifacts]:
    cfg = feature_config or FeaturizerConfig()
    raw = load_panel()
    summary_rows: list[dict[str, float | str]] = []
    ensembles: dict[str, Ensemble] = {}
    per_receptor: dict[str, EnsembleResult] = {}
    for receptor in RECEPTORS:
        records = raw.get(receptor, [])
        if len(records) < 50:
            log.warning("Skipping %s: only %d records", receptor, len(records))
            continue
        train, holdout = _scaffold_split(records)
        X_train, valid_train = featurize_batch([r.smiles for r in train], cfg)
        y_train = np.asarray([train[i].pvalue for i in valid_train], dtype=np.float32)
        fps_train = np.stack(
            [morgan_count(Chem.MolFromSmiles(train[i].smiles)) for i in valid_train]
        )
        result = train_ensemble(X_train, y_train, fps_train)
        ensembles[receptor] = result.ensemble
        per_receptor[receptor] = result

        X_h, valid_h = featurize_batch([r.smiles for r in holdout], cfg)
        y_h = np.asarray([holdout[i].pvalue for i in valid_h], dtype=np.float32)
        if X_h.shape[0] > 0:
            preds, _ = result.ensemble.predict(X_h)
            rmse_h = float(np.sqrt(np.mean((preds - y_h) ** 2)))
            ss_res = float(np.sum((y_h - preds) ** 2))
            ss_tot = float(np.sum((y_h - y_h.mean()) ** 2)) or 1.0
            r2_h = 1.0 - ss_res / ss_tot
        else:
            rmse_h, r2_h = float("nan"), float("nan")
        summary_rows.append(
            {
                "receptor": receptor,
                "n_train": len(train),
                "n_holdout": len(holdout),
                "cv_rmse": result.holdout_rmse,
                "cv_r2": result.holdout_r2,
                "cv_spearman": result.holdout_spearman,
                "holdout_rmse": rmse_h,
                "holdout_r2": r2_h,
            }
        )

    panel = MelanocortinPanel(ensembles=ensembles, feature_config=cfg)
    if save_to is not None:
        panel.save(save_to)
    artifacts = TrainingArtifacts(
        config=cfg,
        per_receptor=per_receptor,
        holdout_summary=pd.DataFrame(summary_rows),
    )
    return panel, artifacts


def descriptive_panel_frame() -> pd.DataFrame:
    raw = load_panel()
    return panel_to_dataframe(raw)
