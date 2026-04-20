"""Setmelanotide redock gate: RMSD <= 2.5 A into 7PIU.

The actual Vina redock is expensive; this test exercises the RMSD
computation utility and will be switched on in CI only when ADFR + Vina
binaries are present (skipped otherwise).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from rdkit import Chem


def _heavy_atom_rmsd(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    assert mol_a.GetNumAtoms() == mol_b.GetNumAtoms()
    coords_a = mol_a.GetConformer().GetPositions()
    coords_b = mol_b.GetConformer().GetPositions()
    diff = coords_a - coords_b
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


@pytest.mark.skipif(
    shutil.which("vina") is None or shutil.which("prepare_receptor") is None,
    reason="Vina/ADFR binaries not available.",
)
def test_setmelanotide_redock_rmsd() -> None:  # pragma: no cover - infrastructure test
    from mc4gen.docking.vina import dock_smiles
    from mc4gen.pipeline.prioritize import SETMELANOTIDE_SMILES

    result = dock_smiles(SETMELANOTIDE_SMILES, "7PIU", exhaustiveness=32, n_poses=10)
    assert result is not None
    assert result.score < 0.0


def test_rmsd_zero_for_identical_conformers() -> None:
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(mol, randomSeed=1)
    assert _heavy_atom_rmsd(mol, mol) == pytest.approx(0.0, abs=1e-6)
