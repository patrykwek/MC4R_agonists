"""Retrosynthesis (AiZynthFinder) + vendor lookup for the final 10 candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.data.vendors import molport_similarity

log = get_logger(__name__)

INPUT_CSV = Path("artifacts/top10_candidates.csv")
OUT_CSV = Path("artifacts/top10_synthesis.csv")


def _aizynth_for(smi: str) -> dict[str, object]:
    from aizynthfinder.aizynthfinder import AiZynthFinder

    finder = AiZynthFinder(configfile="configs/aizynth.yaml")
    finder.stock.select("zinc")
    finder.expansion_policy.select("uspto")
    finder.filter_policy.select("uspto")
    finder.target_smiles = smi
    finder.tree_search()
    finder.build_routes()
    routes = finder.routes
    return {
        "n_routes": len(routes),
        "is_solved": bool(finder.tree.is_solved),
        "top_route_score": float(routes[0].score) if routes else float("nan"),
    }


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    out: list[dict[str, object]] = []
    for _, row in df.iterrows():
        smi = str(row["smiles"])
        vendor_hits = molport_similarity(smi)
        try:
            retro = _aizynth_for(smi)
        except Exception as err:  # pragma: no cover - external dep failure
            log.warning("AiZynth failed for %s: %s", smi[:40], err)
            retro = {"n_routes": 0, "is_solved": False, "top_route_score": float("nan")}
        out.append(
            {
                "smiles": smi,
                "n_vendor_hits": len(vendor_hits),
                **retro,
            }
        )
    pd.DataFrame(out).to_csv(OUT_CSV, index=False)
    log.info("Wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
