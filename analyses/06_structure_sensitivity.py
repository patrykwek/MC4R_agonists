"""Compare run outputs across the five MC4R structures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.reporting.tables import dataframe_to_booktabs

log = get_logger(__name__)

STRUCTURES = ("7AUE", "7PIU", "7PIV", "6W25", "7F53")
TABLE_PATH = Path("artifacts/tables/structure_sensitivity.tex")


def _summarize(run_dir: Path) -> dict[str, float]:
    csv = run_dir / "molecules.csv"
    if not csv.exists():
        return {"mean_score": float("nan"), "n": 0, "mean_vina": float("nan")}
    df = pd.read_csv(csv)
    return {
        "mean_score": float(df.get("score", pd.Series(dtype=float)).mean()),
        "n": int(len(df)),
        "mean_vina": float(df.get("mc4r_docking_vina_raw", pd.Series(dtype=float)).mean()),
    }


def main() -> None:
    rows: list[dict[str, float | str]] = []
    for pdb in STRUCTURES:
        stats = _summarize(Path(f"runs/stage_2_rl_{pdb.lower()}"))
        rows.append({"structure": pdb, **stats})
    df = pd.DataFrame(rows)
    table = dataframe_to_booktabs(
        df,
        caption="RL outcomes across MC4R structures.",
        label="structure_sensitivity",
    )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table.render(), encoding="utf-8")
    log.info("Wrote %s", TABLE_PATH)


if __name__ == "__main__":
    main()
