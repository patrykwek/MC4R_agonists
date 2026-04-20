"""Selectivity index computation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mc4gen.reinvent_plugins.components.melanocortin_selectivity import (
    MelanocortinSelectivityPanel,
)


class TestSelectivityReward:
    def test_empty_prediction_is_zero(self, tmp_path, monkeypatch) -> None:
        from mc4gen.qsar import panel as panel_mod

        fake_panel = MagicMock()
        fake_panel.predict.return_value = None
        monkeypatch.setattr(panel_mod.MelanocortinPanel, "load", lambda _: fake_panel)
        comp = MelanocortinSelectivityPanel.from_parameters({"panel_path": tmp_path / "p.joblib"})
        result = comp(["CCO"])
        assert result.scores[0] == 0.0
        assert result.metadata[0]["reason"] == "prediction_unavailable"

    def test_selectivity_rewards_positive_delta(self, tmp_path, monkeypatch) -> None:
        from mc4gen.qsar import panel as panel_mod
        from mc4gen.qsar.panel import PanelPrediction

        fake = MagicMock()
        fake.predict.return_value = PanelPrediction(
            smiles="CCO",
            pki={"MC4R": 8.0, "MC1R": 5.0, "MC3R": 5.5, "MC5R": 5.5},
            uncertainty={"MC4R": 0.1, "MC1R": 0.2, "MC3R": 0.2, "MC5R": 0.2},
            in_domain={"MC4R": True, "MC1R": True, "MC3R": True, "MC5R": True},
        )
        monkeypatch.setattr(panel_mod.MelanocortinPanel, "load", lambda _: fake)
        comp = MelanocortinSelectivityPanel.from_parameters({"panel_path": tmp_path / "p.joblib"})
        result = comp(["CCO"])
        assert result.scores[0] > 0.5


def test_sigmoid_bounds() -> None:
    from mc4gen.reinvent_plugins.components.melanocortin_selectivity import _sigmoid

    assert _sigmoid(-1000.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert _sigmoid(1000.0, 0.0) == pytest.approx(1.0, abs=1e-6)
