"""11-stage prioritization funnel."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from mc4gen.pipeline.prioritize import (
    PrioritizationConfig,
    _pharmacophore_similarity,
    run_prioritization,
)


class TestPharmacophoreSimilarity:
    def test_self_similarity_one(self) -> None:
        from mc4gen.pipeline.prioritize import SETMELANOTIDE_SMILES

        assert _pharmacophore_similarity(SETMELANOTIDE_SMILES) == 1.0

    def test_bad_smiles_zero(self) -> None:
        assert _pharmacophore_similarity("not_a_smiles") == 0.0


class TestFunnelRuns:
    def test_empty_pool(self) -> None:
        df = pd.DataFrame({"SMILES": []})
        with (
            patch("mc4gen.pipeline.prioritize.load_vendor_pool") as pool,
            patch("mc4gen.pipeline.prioritize.dock_smiles") as dock,
        ):
            pool.return_value = np.zeros((0, 2048), dtype=np.int32)
            dock.return_value = None
            candidates, trace = run_prioritization(df)
            assert candidates == []
            assert trace.stage_counts["stage_1_raw"] == 0


class TestConfigOverride:
    def test_max_candidates_overrides(self) -> None:
        cfg = PrioritizationConfig(top_candidates=5)
        assert cfg.top_candidates == 5
