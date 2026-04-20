"""Eleven-stage prioritization funnel.

Stage definitions and thresholds come from ``AGENTS.md``. Each stage applies a
filter and logs how many molecules survive; the final output is a list of
cluster-representative candidates ready for MM-GBSA / MD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from mc4gen._logging import get_logger
from mc4gen.chemotypes import load_chemotypes, match_any
from mc4gen.data.vendors import load_vendor_pool, max_tanimoto_to_vendor
from mc4gen.docking.vina import dock_smiles, ensemble_score
from mc4gen.pipeline.cluster import cluster_dataframe
from mc4gen.qsar.panel import MelanocortinPanel
from mc4gen.utils.smiles import canonicalize
from mc4gen.utils.validation import evaluate

log = get_logger(__name__)

SETMELANOTIDE_SMILES = (
    "CC(C)C[C@@H](C(=O)N[C@H](C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc2ccc(cc2)O)"
    "C(=O)N[C@@H](CCCCN)C(=O)N)C)NC(=O)[C@H](Cc3c[nH]c4c3cccc4)NC(=O)CNC(=O)"
    "[C@H](Cc5ccc(cc5)O)NC(=O)[C@H](C)N"
)

_MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True, slots=True)
class PrioritizationConfig:
    top_candidates: int = 10
    pharma_similarity_cutoff: float = 0.85
    chemotype_novelty_min: float = 0.5
    vendor_tanimoto_max: float = 0.6
    vina_cutoff_kcal: float = -9.0
    min_active_structures: int = 2
    qsar_mc4r_min: float = 7.0
    qsar_selectivity_min: float = 2.0
    rascore_min: float = 0.8
    cluster_cutoff: float = 0.4
    structures: tuple[str, ...] = ("7PIU", "7AUE", "7PIV", "6W25", "7F53")


@dataclass(frozen=True, slots=True)
class Candidate:
    smiles: str
    vina_score: float
    predicted_pki_mc4r: float
    selectivity_mc1r: float
    selectivity_mc3r: float
    selectivity_mc5r: float
    ra_score: float
    chemotype_novelty: float
    vendor_tanimoto: float
    cluster_id: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrioritizationTrace:
    stage_counts: dict[str, int]


def _pharmacophore_similarity(smi: str, reference: str = SETMELANOTIDE_SMILES) -> float:
    ref_mol = Chem.MolFromSmiles(reference)
    mol = Chem.MolFromSmiles(smi)
    if ref_mol is None or mol is None:
        return 0.0
    ref_fp = _MORGAN_GEN.GetFingerprint(ref_mol)
    fp = _MORGAN_GEN.GetFingerprint(mol)
    return float(DataStructs.TanimotoSimilarity(ref_fp, fp))


def _ra_score(smi: str) -> float:
    try:
        from RAscore.RAscore_NN import RAScorerNN

        scorer = RAScorerNN()
        return float(scorer.predict(smi))
    except Exception:  # pragma: no cover - RAscore optional
        return float("nan")


def _vina_active(per_structure: dict[str, float], cutoff: float, minimum: int) -> bool:
    active = sum(1 for s in per_structure.values() if s <= cutoff)
    return active >= minimum


def run_prioritization(
    pool: pd.DataFrame | Sequence[str] | Path,
    *,
    panel_path: Path | None = None,
    vendor_pool: NDArray[np.int32] | None = None,
    config: PrioritizationConfig | None = None,
    max_candidates: int | None = None,
) -> tuple[list[Candidate], PrioritizationTrace]:
    cfg = config or PrioritizationConfig()
    if max_candidates is not None:
        cfg = PrioritizationConfig(**{**cfg.__dict__, "top_candidates": max_candidates})

    if isinstance(pool, Path):
        df = pd.read_csv(pool)
    elif isinstance(pool, pd.DataFrame):
        df = pool.copy()
    else:
        df = pd.DataFrame({"SMILES": list(pool)})

    if "SMILES" not in df.columns and "smiles" in df.columns:
        df = df.rename(columns={"smiles": "SMILES"})

    trace: dict[str, int] = {"stage_1_raw": len(df)}

    # Stage 2: validity + uniqueness
    df["canonical"] = df["SMILES"].map(lambda s: canonicalize(str(s)))
    df = df.dropna(subset=["canonical"]).drop_duplicates(subset=["canonical"])
    trace["stage_2_valid_unique"] = len(df)

    # Stage 3: pharmacophore similarity to setmelanotide
    df["pharma_sim"] = df["canonical"].map(_pharmacophore_similarity)
    df = df[df["pharma_sim"] >= cfg.pharma_similarity_cutoff]
    trace["stage_3_pharma"] = len(df)

    # Stage 4: Lipinski / drug-likeness
    reports = df["canonical"].map(evaluate)
    mask = reports.map(lambda r: r is not None and len(r.violations) <= 1)
    df = df[mask]
    trace["stage_4_drug_like"] = len(df)

    # Stage 5: chemotype novelty and vendor novelty
    chemotypes = load_chemotypes()
    df["matched_chemotypes"] = df["canonical"].map(lambda s: tuple(match_any(s, chemotypes)))
    df["chemotype_novelty"] = df["matched_chemotypes"].map(lambda hits: 0.0 if hits else 1.0)
    df = df[df["chemotype_novelty"] >= cfg.chemotype_novelty_min]

    pool_fps = vendor_pool if vendor_pool is not None else load_vendor_pool()
    df["vendor_tanimoto"] = df["canonical"].map(
        lambda s: max_tanimoto_to_vendor(str(s), pool_fps)
    )
    df = df[df["vendor_tanimoto"] <= cfg.vendor_tanimoto_max]
    trace["stage_5_novelty"] = len(df)

    # Stage 6: docking ensemble
    vina_rows: list[dict[str, object]] = []
    for smi in df["canonical"]:
        per_structure: dict[str, float] = {}
        for pdb in cfg.structures:
            res = dock_smiles(str(smi), pdb)
            if res is not None:
                per_structure[pdb] = res.score
        vina_rows.append(
            {
                "per_structure": per_structure,
                "vina_score": ensemble_score(
                    {k: type("_R", (), {"score": v}) for k, v in per_structure.items()}  # type: ignore[misc]
                ),
                "active_count": sum(1 for s in per_structure.values() if s <= cfg.vina_cutoff_kcal),
            }
        )
    vina_df = pd.DataFrame(vina_rows)
    df = df.reset_index(drop=True)
    df = pd.concat([df, vina_df], axis=1)
    df = df[df["active_count"] >= cfg.min_active_structures]
    trace["stage_6_docking"] = len(df)

    # Stage 7: QSAR panel gates
    panel: MelanocortinPanel | None = None
    if panel_path is not None and panel_path.exists():
        panel = MelanocortinPanel.load(panel_path)
    mc4r_vals: list[float] = []
    sel_mc1r: list[float] = []
    sel_mc3r: list[float] = []
    sel_mc5r: list[float] = []
    for smi in df["canonical"]:
        if panel is None:
            mc4r_vals.append(float("nan"))
            sel_mc1r.append(float("nan"))
            sel_mc3r.append(float("nan"))
            sel_mc5r.append(float("nan"))
            continue
        pred = panel.predict(str(smi))
        if pred is None:
            mc4r_vals.append(float("nan"))
            sel_mc1r.append(float("nan"))
            sel_mc3r.append(float("nan"))
            sel_mc5r.append(float("nan"))
            continue
        mc4r_vals.append(pred.pki.get("MC4R", float("nan")))
        sel_mc1r.append(pred.pki.get("MC4R", 0.0) - pred.pki.get("MC1R", 0.0))
        sel_mc3r.append(pred.pki.get("MC4R", 0.0) - pred.pki.get("MC3R", 0.0))
        sel_mc5r.append(pred.pki.get("MC4R", 0.0) - pred.pki.get("MC5R", 0.0))
    df["pki_mc4r"] = mc4r_vals
    df["sel_mc1r"] = sel_mc1r
    df["sel_mc3r"] = sel_mc3r
    df["sel_mc5r"] = sel_mc5r
    if panel is not None:
        df = df[
            (df["pki_mc4r"] >= cfg.qsar_mc4r_min)
            & (df["sel_mc1r"] >= cfg.qsar_selectivity_min)
        ]
    trace["stage_7_qsar"] = len(df)

    # Stage 8: RAscore
    df["ra_score"] = df["canonical"].map(_ra_score)
    df = df[(df["ra_score"].isna()) | (df["ra_score"] >= cfg.rascore_min)]
    trace["stage_8_rascore"] = len(df)

    # Stage 9: Butina cluster representatives
    labels = cluster_dataframe(list(df["canonical"]), cutoff=cfg.cluster_cutoff)
    df["cluster_id"] = labels
    df = df.sort_values("vina_score").drop_duplicates(subset=["cluster_id"])
    trace["stage_9_cluster"] = len(df)

    # Stage 10/11 (MMPBSA + MD) are downstream modules: return the top-K here.
    df = df.sort_values(["vina_score", "pki_mc4r"], ascending=[True, False])
    df = df.head(cfg.top_candidates)

    candidates: list[Candidate] = []
    for _, row in df.iterrows():
        candidates.append(
            Candidate(
                smiles=str(row["canonical"]),
                vina_score=float(row.get("vina_score", float("nan"))),
                predicted_pki_mc4r=float(row.get("pki_mc4r", float("nan"))),
                selectivity_mc1r=float(row.get("sel_mc1r", float("nan"))),
                selectivity_mc3r=float(row.get("sel_mc3r", float("nan"))),
                selectivity_mc5r=float(row.get("sel_mc5r", float("nan"))),
                ra_score=float(row.get("ra_score", float("nan"))),
                chemotype_novelty=float(row.get("chemotype_novelty", 0.0)),
                vendor_tanimoto=float(row.get("vendor_tanimoto", 0.0)),
                cluster_id=int(row.get("cluster_id", -1)),
                metadata={"per_structure": row.get("per_structure", {})},
            )
        )
    trace["stage_10_topK"] = len(candidates)
    return candidates, PrioritizationTrace(stage_counts=trace)
