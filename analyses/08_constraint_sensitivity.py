"""Compare strict vs. relaxed Ca2+/Glu100 constraints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.reporting.tables import dataframe_to_booktabs

log = get_logger(__name__)

VARIANTS = {"strict": "runs/stage_3_strict", "relaxed": "runs/stage_3_relaxed"}
TABLE_PATH = Path("artifacts/tables/constraint_sensitivity.tex")


def main() -> None:
    rows: list[dict[str, float | str]] = []
    for name, path_str in VARIANTS.items():
        csv = Path(path_str) / "molecules.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        rows.append(
            {
                "variant": name,
                "n": int(len(df)),
                "mean_score": float(df.get("score", pd.Series(dtype=float)).mean()),
                "ca_satisfied_fraction": float(
                    df.get("calcium_coordination", pd.Series(dtype=float)).mean()
                ),
            }
        )
    table = dataframe_to_booktabs(
        pd.DataFrame(rows),
        caption="Strict vs. relaxed orthosteric constraints.",
        label="constraint_sensitivity",
    )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table.render(), encoding="utf-8")
    log.info("Wrote %s", TABLE_PATH)


if __name__ == "__main__":
    main()
