"""Vendor library caches for novelty assessment.

Supported: ZINC20 (CC BY 4.0), Enamine REAL Space (Enamine open license),
MolPort (free REST search tier). No API keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import requests
from numpy.typing import NDArray
from rdkit import Chem

from mc4gen._logging import get_logger
from mc4gen.data import cache
from mc4gen.utils.fingerprints import bulk_tanimoto, morgan_count

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VendorEntry:
    smiles: str
    vendor: str
    catalog_id: str


ZINC_URL = "https://files.docking.org/2D/{tranche}/{tranche}.smi"
ENAMINE_REAL_SAMPLE = "https://enamine.net/compound-collections/real-compounds/real-sample.smi"
MOLPORT_SEARCH = "https://api.molport.com/api/chemical-search/search/ssimilarity"


def _load_smi(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            smiles = line.split()[0]
            if smiles.startswith("#"):
                continue
            yield smiles


def cache_zinc_tranche(tranche: str) -> Path:
    """Download a ZINC20 SMILES tranche and return the cached path."""
    path = cache.path_for("vendor/zinc", tranche, ".smi")
    if path.exists():
        return path
    url = ZINC_URL.format(tranche=tranche)
    log.info("Downloading ZINC20 tranche %s", tranche)
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    return path


def cache_enamine_real(sample_only: bool = True) -> Path:
    path = cache.path_for("vendor/enamine", "sample", ".smi")
    if path.exists():
        return path
    if not sample_only:
        raise RuntimeError("Full Enamine REAL download is multi-hundred GB; use sample_only=True.")
    log.info("Downloading Enamine REAL Space sample")
    r = requests.get(ENAMINE_REAL_SAMPLE, timeout=600)
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    return path


def molport_similarity(smiles: str, *, similarity: float = 0.7) -> list[VendorEntry]:
    """Query MolPort's free similarity-search endpoint."""
    payload = {
        "User Name": "",
        "Authentication Code": "",
        "Structure": smiles,
        "Search Type": 2,
        "Maximum Search Time": 60000,
        "Similarity Index": similarity,
        "Maximum Results": 100,
        "Chemical Similarity Index Type": "tanimoto",
    }
    try:
        r = requests.post(MOLPORT_SEARCH, json=payload, timeout=90)
        r.raise_for_status()
        body = r.json()
    except requests.RequestException as err:
        log.warning("MolPort search failed for %s: %s", smiles[:40], err)
        return []
    items = body.get("data", {}).get("molecules", [])
    return [
        VendorEntry(smiles=item.get("smiles", ""), vendor="MolPort", catalog_id=str(item.get("id", "")))
        for item in items
        if item.get("smiles")
    ]


def fingerprint_library(smiles_iter: Iterable[str]) -> NDArray[np.int32]:
    fps: list[NDArray[np.int32]] = []
    for s in smiles_iter:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fps.append(morgan_count(mol))
    if not fps:
        return np.zeros((0, 2048), dtype=np.int32)
    return np.stack(fps)


def load_vendor_pool(max_entries: int = 200_000) -> NDArray[np.int32]:
    """Return a fingerprint matrix for the cached vendor libraries."""
    cached = cache.subcache("vendor")
    pool: list[NDArray[np.int32]] = []
    for smi_path in sorted(cached.rglob("*.smi")):
        for smi in _load_smi(smi_path):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            pool.append(morgan_count(mol))
            if len(pool) >= max_entries:
                break
        if len(pool) >= max_entries:
            break
    if not pool:
        return np.zeros((0, 2048), dtype=np.int32)
    return np.stack(pool)


def max_tanimoto_to_vendor(smiles: str, pool: NDArray[np.int32]) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or pool.shape[0] == 0:
        return 0.0
    query = morgan_count(mol)
    sims = bulk_tanimoto(query, pool)
    return float(sims.max())
