"""Run the 11-stage prioritization funnel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.pipeline.prioritize import run_prioritization
from mc4gen.pipeline.run_reinvent import collect_outputs
from mc4gen.reporting.plots import funnel

log = get_logger(__name__)

RUN_DIRS = [Path(f"runs/{name}") for name in ("stage_2_rl_7piu", "stage_2_rl_7aue", "stage_2_rl_7piv")]
OUT_CSV = Path("artifacts/top10_candidates.csv")
FIG_PATH = Path("artifacts/figures/prioritization_funnel.png")


def main() -> None:
    pool = collect_outputs(RUN_DIRS)
    if pool.empty:
        log.error("No molecules found across runs.")
        return
    candidates, trace = run_prioritization(
        pool,
        panel_path=Path("artifacts/qsar_panel.joblib"),
        max_candidates=10,
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([c.__dict__ for c in candidates]).to_csv(OUT_CSV, index=False)
    funnel(trace.stage_counts, FIG_PATH)
    log.info("Wrote %s and %s", OUT_CSV, FIG_PATH)


if __name__ == "__main__":
    main()
