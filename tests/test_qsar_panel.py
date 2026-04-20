"""QSAR panel training & prediction tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mc4gen.qsar.features import FeaturizerConfig, featurize, featurize_batch
from mc4gen.utils.smiles import canonicalize


class TestFeaturizer:
    def test_roundtrip_valid_smiles(self, aspirin_smiles: str) -> None:
        vec = featurize(aspirin_smiles)
        assert vec is not None
        assert vec.shape == (2048 + 208,)

    def test_invalid_returns_none(self) -> None:
        assert featurize("not_a_smiles") is None

    def test_batch_skips_invalid(self, aspirin_smiles: str, ibuprofen_smiles: str) -> None:
        X, valid = featurize_batch([aspirin_smiles, "bogus", ibuprofen_smiles])
        assert X.shape[0] == 2
        assert valid == [0, 2]

    def test_configurable_bits(self, aspirin_smiles: str) -> None:
        cfg = FeaturizerConfig(n_bits=1024, include_descriptors=False)
        vec = featurize(aspirin_smiles, cfg)
        assert vec is not None
        assert vec.shape == (1024,)


class TestPanelShape:
    def test_receptors_listed(self) -> None:
        from mc4gen.qsar.panel import RECEPTORS

        assert set(RECEPTORS) == {"MC1R", "MC3R", "MC4R", "MC5R"}


@given(st.sampled_from(["CCO", "c1ccccc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"]))
@settings(deadline=None, max_examples=10)
def test_canonicalize_idempotent(smi: str) -> None:
    first = canonicalize(smi)
    assert first is not None
    second = canonicalize(first)
    assert first == second


def test_panel_load_missing_file(tmp_path: Path) -> None:
    from mc4gen.qsar.panel import MelanocortinPanel

    with pytest.raises(Exception):
        MelanocortinPanel.load(tmp_path / "missing.joblib")
