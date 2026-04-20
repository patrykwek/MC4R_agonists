"""SMARTS chemotype matching."""

from __future__ import annotations

from mc4gen.chemotypes import load_chemotypes, match_any


class TestChemotypes:
    def test_load_non_empty(self) -> None:
        cts = load_chemotypes()
        assert len(cts) >= 6
        assert all(ct.name for ct in cts)

    def test_thiq_matches(self) -> None:
        cts = load_chemotypes()
        hits = match_any("c1ccc2CCNCc2c1", cts)
        assert "tetrahydroisoquinoline_core" in hits

    def test_ethanol_no_match(self) -> None:
        cts = load_chemotypes()
        assert match_any("CCO", cts) == []

    def test_bad_smiles_empty_list(self) -> None:
        cts = load_chemotypes()
        assert match_any("not_real", cts) == []

    def test_all_patterns_compile(self) -> None:
        cts = load_chemotypes()
        for ct in cts:
            assert ct.pattern() is not None
