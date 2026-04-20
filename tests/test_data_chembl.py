"""ChEMBL data layer tests (no live HTTP)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from mc4gen.data import chembl


class TestNormalization:
    def test_unknown_target_raises(self) -> None:
        with pytest.raises(ValueError):
            chembl.fetch_activities("UNKNOWN")

    def test_to_pvalue(self) -> None:
        assert chembl._to_pvalue(1.0) == pytest.approx(9.0)
        assert math.isnan(chembl._to_pvalue(0.0))

    def test_normalize_filters(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "canonical_smiles": "CCO",
                    "standard_value": 10.0,
                    "standard_units": "nM",
                    "standard_relation": "=",
                    "standard_type": "Ki",
                    "confidence_score": 9,
                    "molecule_chembl_id": "X1",
                },
                {
                    "canonical_smiles": "CCO",
                    "standard_value": 10.0,
                    "standard_units": "mM",
                    "standard_relation": "=",
                    "standard_type": "Ki",
                    "confidence_score": 9,
                    "molecule_chembl_id": "X2",
                },
                {
                    "canonical_smiles": "CCO",
                    "standard_value": 10.0,
                    "standard_units": "nM",
                    "standard_relation": ">",
                    "standard_type": "Ki",
                    "confidence_score": 9,
                    "molecule_chembl_id": "X3",
                },
            ]
        )
        out = chembl.normalize_activities(raw, "MC4R")
        assert len(out) == 1
        assert out[0].receptor == "MC4R"

    def test_deduplicate_collapses_replicates(self) -> None:
        records = [
            chembl.ChEMBLRecord("CCO", "MC4R", "Ki", 10.0, 8.0, 9, "X"),
            chembl.ChEMBLRecord("CCO", "MC4R", "Ki", 100.0, 7.0, 9, "Y"),
        ]
        out = chembl.deduplicate(records)
        assert len(out) == 1
        assert out[0].pvalue == pytest.approx(7.0)
