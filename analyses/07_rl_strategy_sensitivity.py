"""Compare REINVENT vs. AHC vs. Augmented Memory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.reporting.tables import dataframe_to_booktabs

log = get_logger(__name__)

STRATEGIES = {
    "reinvent": "runs/stage_2_7piu",
    "ahc": "runs/ahc",
    "augmem": "runs/augmem",
}
TABLE_PATH = Path("artifacts/tables/rl_strategy_sensitivity.tex")


def main() -> None:
    rows: list[dict[str, float | str]] = []
    for name, path_str in STRATEGIES.items():
        csv = Path(path_str) / "molecules.csv"
        if not csv.exists():
            rows.append({"strategy": name, "n": 0, "mean_score": float("nan")})
            continue
        df = pd.read_csv(csv)
        rows.append(
            {
                "strategy": name,
                "n": int(len(df)),
                "mean_score": float(df.get("score", pd.Series(dtype=float)).mean()),
                "unique_scaffolds": int(df["canonical_scaffold"].nunique())
                if "canonical_scaffold" in df
                else 0,
            }
        )
    df_out = pd.DataFrame(rows)
    table = dataframe_to_booktabs(df_out, caption="RL strategy comparison.", label="rl_sensitivity")
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table.render(), encoding="utf-8")
    log.info("Wrote %s", TABLE_PATH)


if __name__ == "__main__":
    main()
