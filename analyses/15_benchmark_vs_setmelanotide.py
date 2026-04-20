"""Side-by-side benchmark of the final 10 candidates against setmelanotide."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.docking.vina import dock_smiles, ensemble_score
from mc4gen.pipeline.prioritize import SETMELANOTIDE_SMILES
from mc4gen.qsar.panel import MelanocortinPanel
from mc4gen.reporting.tables import dataframe_to_booktabs
from mc4gen.utils.validation import evaluate

log = get_logger(__name__)

TOP_CSV = Path("artifacts/top10_candidates.csv")
OUT_TABLE = Path("artifacts/tables/setmelanotide_benchmark.tex")
STRUCTURES = ("7PIU", "7AUE", "7PIV", "6W25", "7F53")


def _dock_ensemble(smi: str) -> float:
    per_struct: dict[str, float] = {}
    for pdb in STRUCTURES:
        res = dock_smiles(smi, pdb)
        if res is not None:
            per_struct[pdb] = res.score
    if not per_struct:
        return float("nan")
    return ensemble_score(
        {k: type("_R", (), {"score": v}) for k, v in per_struct.items()}  # type: ignore[misc]
    )


def main() -> None:
    panel = MelanocortinPanel.load(Path("artifacts/qsar_panel.joblib"))
    candidates = pd.read_csv(TOP_CSV)
    rows: list[dict[str, object]] = []

    reference_report = evaluate(SETMELANOTIDE_SMILES)
    rows.append(
        {
            "compound": "setmelanotide",
            "pki_mc4r": float("nan"),
            "selectivity_mc1r": float("nan"),
            "vina_ensemble": _dock_ensemble(SETMELANOTIDE_SMILES),
            "mw": reference_report.mw if reference_report else float("nan"),
            "rule_of_5_violations": len(reference_report.violations) if reference_report else -1,
        }
    )
    for i, row in candidates.iterrows():
        smi = str(row["smiles"])
        pred = panel.predict(smi)
        report = evaluate(smi)
        rows.append(
            {
                "compound": f"candidate_{i + 1}",
                "pki_mc4r": pred.pki.get("MC4R", float("nan")) if pred else float("nan"),
                "selectivity_mc1r": (
                    pred.pki.get("MC4R", 0.0) - pred.pki.get("MC1R", 0.0) if pred else float("nan")
                ),
                "vina_ensemble": row.get("vina_score", float("nan")),
                "mw": report.mw if report else float("nan"),
                "rule_of_5_violations": len(report.violations) if report else -1,
            }
        )
    table = dataframe_to_booktabs(
        pd.DataFrame(rows),
        caption="Benchmark of mc4gen candidates against setmelanotide.",
        label="setmelanotide_benchmark",
    )
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text(table.render(), encoding="utf-8")
    log.info("Wrote %s", OUT_TABLE)


if __name__ == "__main__":
    main()
