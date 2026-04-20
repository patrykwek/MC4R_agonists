"""ChEMBL 34 loaders for the four melanocortin receptors.

Uses the ChEMBL web-services REST API (CC BY-SA 3.0, no key required). All
responses are cached under ``~/.mc4gen/cache/chembl/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests

from mc4gen._logging import get_logger
from mc4gen.data import cache
from mc4gen.utils.smiles import canonicalize

log = get_logger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

TARGET_IDS: dict[str, str] = {
    "MC1R": "CHEMBL4607",
    "MC3R": "CHEMBL4608",
    "MC4R": "CHEMBL3227",
    "MC5R": "CHEMBL4630",
}

ALLOWED_RELATIONS: frozenset[str] = frozenset({"=", "~"})
ALLOWED_ACTIVITY_TYPES: frozenset[str] = frozenset({"Ki", "IC50", "EC50", "Kd"})


@dataclass(frozen=True, slots=True)
class ChEMBLRecord:
    smiles: str
    receptor: str
    activity_type: str
    value_nm: float
    pvalue: float
    confidence_score: int
    chembl_id: str


def _to_pvalue(value_nm: float) -> float:
    if value_nm <= 0:
        return math.nan
    return -math.log10(value_nm * 1e-9)


def _request_page(url: str, params: dict[str, str | int]) -> dict[str, object]:
    response = requests.get(url, params=params, timeout=60, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def fetch_activities(target: str, *, limit: int | None = None) -> pd.DataFrame:
    """Fetch activity rows for one receptor, cached on disk."""
    if target not in TARGET_IDS:
        raise ValueError(f"Unknown target {target!r}; choose from {sorted(TARGET_IDS)}")
    ckey = cache.hash_key("chembl.activities", target, str(limit))
    cached = cache.json_get("chembl", ckey)
    if cached is not None:
        return pd.DataFrame(cached)

    target_id = TARGET_IDS[target]
    url = f"{CHEMBL_API}/activity.json"
    params: dict[str, str | int] = {
        "target_chembl_id": target_id,
        "standard_type__in": ",".join(sorted(ALLOWED_ACTIVITY_TYPES)),
        "limit": 1000,
        "offset": 0,
    }
    records: list[dict[str, object]] = []
    while True:
        payload = _request_page(url, params)
        page = payload.get("activities", [])
        records.extend(page)
        total = int(payload.get("page_meta", {}).get("total_count", len(records)))
        offset = int(params["offset"]) + int(params["limit"])
        if offset >= total or (limit and len(records) >= limit):
            break
        params["offset"] = offset
    df = pd.DataFrame(records)
    cache.json_put("chembl", ckey, df.to_dict(orient="records"))
    return df


def normalize_activities(df: pd.DataFrame, receptor: str) -> list[ChEMBLRecord]:
    """Clean ChEMBL raw activities into pKi records ready for QSAR training."""
    out: list[ChEMBLRecord] = []
    for row in df.itertuples(index=False):
        smiles_raw = getattr(row, "canonical_smiles", None)
        value = getattr(row, "standard_value", None)
        unit = getattr(row, "standard_units", None)
        rel = getattr(row, "standard_relation", None)
        atype = getattr(row, "standard_type", None)
        conf = getattr(row, "confidence_score", None)
        chembl_id = getattr(row, "molecule_chembl_id", None)
        if smiles_raw is None or value is None or unit != "nM":
            continue
        if rel not in ALLOWED_RELATIONS or atype not in ALLOWED_ACTIVITY_TYPES:
            continue
        if conf is None or int(conf) < 8:
            continue
        canon = canonicalize(str(smiles_raw))
        if canon is None:
            continue
        try:
            vnm = float(value)
        except (TypeError, ValueError):
            continue
        pval = _to_pvalue(vnm)
        if math.isnan(pval):
            continue
        out.append(
            ChEMBLRecord(
                smiles=canon,
                receptor=receptor,
                activity_type=str(atype),
                value_nm=vnm,
                pvalue=pval,
                confidence_score=int(conf),
                chembl_id=str(chembl_id) if chembl_id else "",
            )
        )
    return out


def deduplicate(records: Iterable[ChEMBLRecord]) -> list[ChEMBLRecord]:
    """Deduplicate by (smiles, receptor); collapse replicates by median pKi."""
    groups: dict[tuple[str, str], list[ChEMBLRecord]] = {}
    for r in records:
        groups.setdefault((r.smiles, r.receptor), []).append(r)
    deduped: list[ChEMBLRecord] = []
    for (smiles, receptor), group in groups.items():
        pvals = sorted(r.pvalue for r in group)
        median_p = pvals[len(pvals) // 2]
        first = group[0]
        deduped.append(
            ChEMBLRecord(
                smiles=smiles,
                receptor=receptor,
                activity_type=first.activity_type,
                value_nm=10 ** (-median_p) * 1e9,
                pvalue=median_p,
                confidence_score=max(r.confidence_score for r in group),
                chembl_id=first.chembl_id,
            )
        )
    return deduped


def load_panel() -> dict[str, list[ChEMBLRecord]]:
    """Fetch, normalize, and deduplicate all four receptors."""
    out: dict[str, list[ChEMBLRecord]] = {}
    for receptor in TARGET_IDS:
        raw = fetch_activities(receptor)
        clean = normalize_activities(raw, receptor)
        out[receptor] = deduplicate(clean)
        log.info("ChEMBL %s: %d unique actives", receptor, len(out[receptor]))
    return out


def panel_to_dataframe(panel: dict[str, list[ChEMBLRecord]]) -> pd.DataFrame:
    rows = [
        {
            "smiles": r.smiles,
            "receptor": r.receptor,
            "pvalue": r.pvalue,
            "activity_type": r.activity_type,
            "confidence_score": r.confidence_score,
            "chembl_id": r.chembl_id,
        }
        for recs in panel.values()
        for r in recs
    ]
    return pd.DataFrame(rows)
