"""Load and match the MC4R chemotype SMARTS library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rdkit import Chem

from mc4gen._logging import get_logger

log = get_logger(__name__)

_DEFAULT_YAML = Path(__file__).parent / "mc4r_chemotypes.yaml"


@dataclass(frozen=True, slots=True)
class Chemotype:
    name: str
    smarts: str
    description: str
    reference: str

    def pattern(self) -> Chem.Mol:
        mol = Chem.MolFromSmarts(self.smarts)
        if mol is None:
            raise ValueError(f"Invalid SMARTS for chemotype {self.name!r}: {self.smarts!r}")
        return mol


def _parse_yaml(text: str) -> list[dict[str, str]]:
    """Tiny YAML subset sufficient for this flat list-of-dicts format."""
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.startswith("- "):
            if current is not None:
                entries.append(current)
            current = {}
            line = line[2:]
            if ":" in line:
                k, _, v = line.partition(":")
                current[k.strip()] = v.strip().strip('"').strip("'")
        else:
            if current is None:
                continue
            k, _, v = line.partition(":")
            current[k.strip()] = v.strip().strip('"').strip("'")
    if current is not None:
        entries.append(current)
    return entries


def load_chemotypes(path: Path | None = None) -> list[Chemotype]:
    path = path or _DEFAULT_YAML
    text = path.read_text(encoding="utf-8")
    raw = _parse_yaml(text)
    out: list[Chemotype] = []
    for entry in raw:
        try:
            ct = Chemotype(
                name=entry["name"],
                smarts=entry["smarts"],
                description=entry.get("description", ""),
                reference=entry.get("reference", ""),
            )
        except KeyError as err:
            log.warning("Skipping malformed chemotype entry: missing %s", err)
            continue
        out.append(ct)
    return out


def match_any(smiles: str, chemotypes: Iterable[Chemotype]) -> list[str]:
    """Return names of chemotypes whose SMARTS matches ``smiles``."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    hits: list[str] = []
    for ct in chemotypes:
        if mol.HasSubstructMatch(ct.pattern()):
            hits.append(ct.name)
    return hits


def novelty_score(smiles: str, chemotypes: Iterable[Chemotype]) -> float:
    """Return 1.0 if no chemotype matches, else 0.0."""
    return 0.0 if match_any(smiles, chemotypes) else 1.0
