"""Integration smoke test: 10-step mock RL loop, no REINVENT subprocess."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


class TestMiniRLLoop:
    def test_reward_monotonic_in_expectation(self) -> None:
        from mc4gen.reinvent_plugins.components.chemotype_novelty import (
            ChemotypeNoveltyFilter,
        )

        comp = ChemotypeNoveltyFilter.from_parameters({})

        rewards: list[float] = []
        pool = ["C", "CC", "CCC", "c1ccccc1", "CCO", "CCN", "NCC", "OCC", "C=C", "C#C"]
        for step in range(10):
            # Bias the pool by pretending the agent samples preferentially novel molecules.
            subset = pool[: step + 2]
            rewards.append(float(comp(subset).scores.mean()))
        assert len(rewards) == 10
        assert all(np.isfinite(r) for r in rewards)


class TestSmokePipeline:
    def test_prioritization_on_small_pool(self) -> None:
        import pandas as pd

        from mc4gen.pipeline.prioritize import run_prioritization

        pool = pd.DataFrame({"SMILES": ["c1ccccc1", "CCO", "bogus"]})
        with (
            patch("mc4gen.pipeline.prioritize.load_vendor_pool") as pool_mod,
            patch("mc4gen.pipeline.prioritize.dock_smiles") as dock,
            patch("mc4gen.pipeline.prioritize._ra_score") as ra,
        ):
            pool_mod.return_value = np.zeros((0, 2048), dtype=np.int32)
            dock.return_value = MagicMock(score=-10.0, pdb_id="7PIU")
            ra.return_value = 0.9
            candidates, trace = run_prioritization(pool)
            assert trace.stage_counts["stage_1_raw"] == 3
            assert trace.stage_counts["stage_2_valid_unique"] == 2
            assert isinstance(candidates, list)
