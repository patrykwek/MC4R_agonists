r"""Distance-based pose constraints used as Vina post-hoc penalties.

For each constraint we compute a flat-bottom energy

.. math::

    E(d) = \begin{cases}
    0 & d \le d_{\max} \\
    k (d - d_{\max})^2 & d > d_{\max}
    \end{cases}

and add it to the Vina score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DistanceConstraint:
    label: str
    reference_xyz: tuple[float, float, float]
    ligand_pattern: str
    d_max: float
    spring_k: float = 5.0

    def penalty(self, ligand_coords: NDArray[np.float32]) -> tuple[float, bool]:
        ref = np.asarray(self.reference_xyz, dtype=np.float32)
        if ligand_coords.shape[0] == 0:
            return (self.spring_k * self.d_max * self.d_max, False)
        d = float(np.linalg.norm(ligand_coords - ref, axis=1).min())
        if d <= self.d_max:
            return (0.0, True)
        return (self.spring_k * (d - self.d_max) ** 2, False)


CA_COORDINATION = DistanceConstraint(
    label="Ca2+ coordination (acceptor <= 2.8 A)",
    reference_xyz=(0.6, -0.9, 4.0),
    ligand_pattern="[O,N;X2;$([O]=C),$([n]),$([N;-])]",
    d_max=2.8,
)

GLU100_HBOND = DistanceConstraint(
    label="Glu100^3.33 H-bond (donor <= 3.2 A)",
    reference_xyz=(2.3, -1.5, 2.9),
    ligand_pattern="[N;H1,H2,H3;!$(NC=O)]",
    d_max=3.2,
)

HIS264_CONTACT = DistanceConstraint(
    label="His264^6.54 hydrophobic contact (<= 4.5 A)",
    reference_xyz=(-2.2, 2.5, 3.1),
    ligand_pattern="[c,C;R]",
    d_max=4.5,
    spring_k=2.0,
)


DEFAULT_CONSTRAINTS: tuple[DistanceConstraint, ...] = (
    CA_COORDINATION,
    GLU100_HBOND,
    HIS264_CONTACT,
)
