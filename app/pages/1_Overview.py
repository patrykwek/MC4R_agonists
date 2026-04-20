"""Overview page: abstract, research question, structure gallery."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.components.structure_viewer import render_pdb
from mc4gen.data.structures import STRUCTURES

st.header("Overview")

st.markdown(
    """
**Research question.** Using REINVENT 4 with structure-based multi-objective
reinforcement learning conditioned on the publicly available cryo-EM structures
of MC4R (PDB 7AUE, 7PIU, 7PIV, 6W25, 7F53), can we design *de novo*
small-molecule agonists that preserve the key orthosteric contacts of
setmelanotide and are predicted by a ChEMBL-trained QSAR panel to be selective
for MC4R over MC1R, MC3R, and MC5R?
"""
)

st.subheader("Structure gallery")
cols = st.columns(len(STRUCTURES))
for col, (pdb_id, struct) in zip(cols, STRUCTURES.items(), strict=True):
    with col:
        st.markdown(f"**{pdb_id}** — {struct.state} ({struct.ligand})")
        pdb_path = Path.home() / ".mc4gen" / "cache" / "structures" / f"{pdb_id.lower()}.pdb"
        if pdb_path.exists():
            render_pdb(pdb_path.read_text(encoding="utf-8"), height=280)
        else:
            st.info(f"Run `python analyses/02_structure_preparation.py` to cache {pdb_id}.")

st.subheader("Live run status")
runs_dir = Path("runs")
if runs_dir.exists():
    for child in sorted(runs_dir.iterdir()):
        if child.is_dir():
            csv = child / "molecules.csv"
            st.write(f"`{child.name}` — {'✅' if csv.exists() else '⏳ pending'}")
else:
    st.info("No runs on disk yet.")
