"""Ca2+ coordination predictor."""

from __future__ import annotations

import numpy as np

from mc4gen.docking.constraints import CA_COORDINATION, DistanceConstraint


class TestDistanceConstraint:
    def test_zero_penalty_when_close(self) -> None:
        coords = np.asarray([[0.6, -0.9, 4.0]], dtype=np.float32)
        penalty, ok = CA_COORDINATION.penalty(coords)
        assert penalty == 0.0
        assert ok is True

    def test_penalty_grows_quadratic(self) -> None:
        ref = CA_COORDINATION.reference_xyz
        far = np.asarray([[ref[0] + 10.0, ref[1], ref[2]]], dtype=np.float32)
        penalty, ok = CA_COORDINATION.penalty(far)
        assert ok is False
        assert penalty > 0.0

    def test_empty_coords_returns_default_penalty(self) -> None:
        empty = np.zeros((0, 3), dtype=np.float32)
        penalty, ok = CA_COORDINATION.penalty(empty)
        assert ok is False
        assert penalty > 0.0

    def test_custom_constraint_parameters(self) -> None:
        cons = DistanceConstraint("t", (0.0, 0.0, 0.0), "[O]", 1.0, 3.0)
        pen, ok = cons.penalty(np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32))
        assert ok is False
        assert pen == (3.0 * (3.0 - 1.0) ** 2)
