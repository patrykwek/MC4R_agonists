"""QSAR-panel selectivity distributions over generated molecules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.qsar.panel import MelanocortinPanel
from mc4gen.reporting.plots import score_distribution

log = get_logger(__name__)

PANEL_PATH = Path("artifacts/qsar_panel.joblib")
INPUT_CSV = Path("runs/stage_2_7piu/molecules.csv")
OUT_FIG = Path("artifacts/figures/selectivity_distributions.png")


def main() -> None:
    panel = MelanocortinPanel.load(PANEL_PATH)
    df = pd.read_csv(INPUT_CSV)
    preds = panel.predict_batch(df["SMILES"].astype(str).tolist())

    mc4r = np.asarray([p.pki.get("MC4R", np.nan) if p else np.nan for p in preds], dtype=np.float32)
    mc1r = np.asarray([p.pki.get("MC1R", np.nan) if p else np.nan for p in preds], dtype=np.float32)
    mc3r = np.asarray([p.pki.get("MC3R", np.nan) if p else np.nan for p in preds], dtype=np.float32)
    mc5r = np.asarray([p.pki.get("MC5R", np.nan) if p else np.nan for p in preds], dtype=np.float32)

    distributions = {
        "pKi(MC4R)": mc4r,
        "MC4R-MC1R": mc4r - mc1r,
        "MC4R-MC3R": mc4r - mc3r,
        "MC4R-MC5R": mc4r - mc5r,
    }
    score_distribution(distributions, OUT_FIG)
    log.info("Wrote %s", OUT_FIG)


if __name__ == "__main__":
    main()
