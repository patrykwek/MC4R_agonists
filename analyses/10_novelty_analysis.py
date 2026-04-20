"""Chemotype and vendor-library novelty of the generated pool."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.chemotypes import load_chemotypes, match_any
from mc4gen.data.vendors import load_vendor_pool, max_tanimoto_to_vendor
from mc4gen.reporting.tables import dataframe_to_booktabs

log = get_logger(__name__)

INPUT_CSV = Path("runs/stage_2_7piu/molecules.csv")
TABLE_PATH = Path("artifacts/tables/novelty.tex")


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    chemotypes = load_chemotypes()
    vendor_pool = load_vendor_pool()

    df["matched_chemotypes"] = df["SMILES"].astype(str).map(
        lambda s: ";".join(match_any(s, chemotypes))
    )
    df["max_tanimoto_vendor"] = df["SMILES"].astype(str).map(
        lambda s: max_tanimoto_to_vendor(s, vendor_pool)
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "fraction_matching_chemotype",
                "fraction_vendor_sim_>0.6",
                "median_vendor_sim",
            ],
            "value": [
                float((df["matched_chemotypes"] != "").mean()),
                float((df["max_tanimoto_vendor"] > 0.6).mean()),
                float(df["max_tanimoto_vendor"].median()),
            ],
        }
    )
    table = dataframe_to_booktabs(summary, caption="Novelty metrics.", label="novelty")
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table.render(), encoding="utf-8")
    log.info("Wrote %s", TABLE_PATH)


if __name__ == "__main__":
    main()
