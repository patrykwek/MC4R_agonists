"""Structure preparation (metadata-only)."""

from __future__ import annotations

from mc4gen.data.structures import DEFAULT_GRID_BOX, STRUCTURES, active_structures, all_structures


class TestStructureRegistry:
    def test_five_structures(self) -> None:
        assert set(STRUCTURES) == {"7AUE", "7PIU", "7PIV", "6W25", "7F53"}

    def test_each_has_grid_box(self) -> None:
        for pdb in STRUCTURES:
            assert pdb in DEFAULT_GRID_BOX
            center, size = DEFAULT_GRID_BOX[pdb]
            assert len(center) == 3
            assert all(s > 0 for s in size)

    def test_active_filter(self) -> None:
        active = active_structures()
        assert all(s.state == "active" for s in active)
        assert len(active) == 4

    def test_all_returns_five(self) -> None:
        assert len(all_structures()) == 5

    def test_setmelanotide_flag(self) -> None:
        assert STRUCTURES["7AUE"].ligand == "setmelanotide"
        assert STRUCTURES["7PIU"].has_calcium is True
