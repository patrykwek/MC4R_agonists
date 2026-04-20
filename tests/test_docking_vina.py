"""AutoDock Vina wrapper tests (mocked; no GPU / external binaries)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mc4gen.docking.vina import DockingResult, ensemble_score


def _make(score: float, pdb_id: str = "7PIU") -> DockingResult:
    return DockingResult(smiles="CCO", pdb_id=pdb_id, score=score, pose_pdbqt="")


class TestEnsembleScore:
    def test_single_structure(self) -> None:
        assert ensemble_score({"7PIU": _make(-8.0)}) == pytest.approx(-8.0)

    def test_median_two(self) -> None:
        per = {"7PIU": _make(-9.0), "7AUE": _make(-7.0)}
        assert ensemble_score(per) == pytest.approx(-8.0)

    def test_median_three(self) -> None:
        per = {"a": _make(-7.0), "b": _make(-9.0), "c": _make(-11.0)}
        assert ensemble_score(per) == pytest.approx(-9.0)

    def test_empty_returns_inf(self) -> None:
        import math

        assert math.isinf(ensemble_score({}))


class TestCacheKeying:
    def test_cache_key_includes_all_args(self) -> None:
        from mc4gen.docking.vina import _cache_key

        k1 = _cache_key("CCO", "7PIU", 1)
        k2 = _cache_key("CCO", "7PIU", 2)
        assert k1 != k2


class TestDockSmilesMocked:
    def test_returns_cached(self, tmp_cache_root) -> None:
        with patch("mc4gen.docking.vina.cache") as cache_mod:
            cache_mod.json_get.return_value = {
                "smiles": "CCO",
                "pdb_id": "7PIU",
                "score": -8.5,
                "pose_pdbqt": "",
                "constraints_satisfied": {},
                "interactions": {},
            }
            cache_mod.hash_key.return_value = "abc"
            from mc4gen.docking.vina import dock_smiles

            result = dock_smiles("CCO", "7PIU")
            assert result is not None
            assert result.score == pytest.approx(-8.5)
