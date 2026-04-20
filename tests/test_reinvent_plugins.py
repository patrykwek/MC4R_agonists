"""REINVENT 4 plugin correctness + registration."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEntryPointRegistration:
    def test_all_components_registered(self) -> None:
        eps = importlib_metadata.entry_points(group="reinvent.plugins.components")
        names = {ep.name for ep in eps}
        expected = {
            "mc4r_docking_vina",
            "melanocortin_selectivity",
            "chemotype_novelty",
            "calcium_coordination",
            "vendor_novelty",
            "rascore_component",
        }
        missing = expected - names
        assert not missing, f"Missing entry points: {missing}"


class TestChemotypeNovelty:
    def test_scores_one_for_miss(self) -> None:
        from mc4gen.reinvent_plugins.components.chemotype_novelty import (
            ChemotypeNoveltyFilter,
        )

        comp = ChemotypeNoveltyFilter.from_parameters({})
        result = comp(["C"])
        assert result.scores.shape == (1,)
        assert result.scores[0] == pytest.approx(1.0)

    def test_scores_zero_for_hit(self) -> None:
        from mc4gen.reinvent_plugins.components.chemotype_novelty import (
            ChemotypeNoveltyFilter,
        )

        comp = ChemotypeNoveltyFilter.from_parameters({})
        # simple THIQ
        result = comp(["c1ccc2CCNCc2c1"])
        assert result.scores[0] == pytest.approx(0.0)


class TestMC4RDockingVina:
    def test_infinite_returns_zero_reward(self) -> None:
        from mc4gen.reinvent_plugins.components.mc4r_docking_vina import MC4RDockingVina

        comp = MC4RDockingVina.from_parameters({"minimum_structures": 5})
        with patch("mc4gen.reinvent_plugins.components.mc4r_docking_vina.dock_smiles") as ds:
            ds.return_value = None
            result = comp(["CCO"])
            assert result.scores[0] == 0.0

    def test_aggregation_median(self) -> None:
        from mc4gen.reinvent_plugins.components.mc4r_docking_vina import MC4RDockingVina

        comp = MC4RDockingVina.from_parameters(
            {"pdb_ids": ["7PIU", "7AUE", "7PIV"], "minimum_structures": 1}
        )
        poses = iter(
            [
                MagicMock(score=-7.0),
                MagicMock(score=-9.0),
                MagicMock(score=-11.0),
            ]
        )
        with patch(
            "mc4gen.reinvent_plugins.components.mc4r_docking_vina.dock_smiles",
            side_effect=lambda *a, **kw: next(poses),
        ):
            result = comp(["CCO"])
            assert result.metadata[0]["aggregated_kcal"] == pytest.approx(-9.0)


class TestIntegrationSmokeTenSteps:
    def test_10_step_loop_completes(self) -> None:
        from mc4gen.reinvent_plugins.components.chemotype_novelty import (
            ChemotypeNoveltyFilter,
        )

        comp = ChemotypeNoveltyFilter.from_parameters({})
        generated = ["C", "CC", "CCC", "CCCC", "c1ccccc1", "CCO", "CCN", "CNC", "CC(C)C", "O=CO"]
        scores = []
        for _ in range(10):
            scores.append(comp(generated).scores.mean())
        assert len(scores) == 10
        assert np.all(np.asarray(scores) >= 0.0)
