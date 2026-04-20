"""Descriptive statistics for the ChEMBL MC1/3/4/5R panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.data.chembl import load_panel, panel_to_dataframe
from mc4gen.reporting.plots import score_distribution
from mc4gen.reporting.tables import dataframe_to_booktabs

log = get_logger(__name__)

OUT_DIR = Path("artifacts/tables")
FIG_DIR = Path("artifacts/figures")


def main() -> None:
    panel = load_panel()
    frame = panel_to_dataframe(panel)
    summary = (
        frame.groupby("receptor")
        .agg(
            n_compounds=("smiles", "nunique"),
            median_pki=("pvalue", "median"),
            p5=("pvalue", lambda s: float(s.quantile(0.05))),
            p95=("pvalue", lambda s: float(s.quantile(0.95))),
        )
        .reset_index()
    )
    table = dataframe_to_booktabs(
        summary,
        caption="ChEMBL 34 melanocortin panel descriptive statistics.",
        label="chembl_panel",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "chembl_panel.tex").write_text(table.render(), encoding="utf-8")

    arrays: dict[str, pd.Series] = {
        receptor: frame.loc[frame.receptor == receptor, "pvalue"].to_numpy()
        for receptor in ("MC1R", "MC3R", "MC4R", "MC5R")
    }
    score_distribution(arrays, FIG_DIR / "chembl_pki_distributions.png")
    log.info("Wrote %s", OUT_DIR / "chembl_panel.tex")


if __name__ == "__main__":
    main()
