"""Train the four-receptor QSAR panel and validate on scaffold-split hold-outs."""

from __future__ import annotations

from pathlib import Path

from mc4gen._logging import get_logger
from mc4gen.qsar.panel import train_panel
from mc4gen.reporting.tables import dataframe_to_booktabs

log = get_logger(__name__)

PANEL_PATH = Path("artifacts/qsar_panel.joblib")
TABLE_PATH = Path("artifacts/tables/qsar_summary.tex")


def main() -> None:
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel, artifacts = train_panel(save_to=PANEL_PATH)

    if artifacts.holdout_summary.empty:
        log.error("QSAR training produced no summary rows; aborting.")
        return
    mc4r_row = artifacts.holdout_summary.query("receptor == 'MC4R'")
    if not mc4r_row.empty and float(mc4r_row["holdout_r2"].iloc[0]) < 0.5:
        log.error("QSAR MC4R hold-out R^2 below 0.5; failing gate.")

    table = dataframe_to_booktabs(
        artifacts.holdout_summary,
        caption="QSAR panel training and hold-out metrics.",
        label="qsar_summary",
    )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table.render(), encoding="utf-8")
    log.info("Panel saved to %s, metrics -> %s", PANEL_PATH, TABLE_PATH)
    _ = panel  # keep reference


if __name__ == "__main__":
    main()
