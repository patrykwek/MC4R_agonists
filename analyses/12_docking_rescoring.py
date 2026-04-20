"""smina cross-validation of the Vina scores for the top candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.docking import smina

log = get_logger(__name__)

INPUT_CSV = Path("artifacts/top10_candidates.csv")
OUT_CSV = Path("artifacts/top10_smina.csv")


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    rows: list[dict[str, object]] = []
    for _, candidate in df.iterrows():
        smi = str(candidate["smiles"])
        for pdb in ("7PIU", "7AUE", "7PIV"):
            result = smina.dock(smi, pdb)
            rows.append(
                {
                    "smiles": smi,
                    "pdb_id": pdb,
                    "smina_score": result.score if result is not None else float("nan"),
                    "smina_out": str(result.out_path) if result is not None else "",
                }
            )
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    log.info("Wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
