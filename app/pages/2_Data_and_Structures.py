"""ChEMBL panel stats + structural alignment overlay."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.structure_viewer import render_pdb

st.header("Data & Structures")

summary_path = Path("artifacts/chembl_panel_summary.csv")
if summary_path.exists():
    st.subheader("ChEMBL 34 panel")
    st.dataframe(pd.read_csv(summary_path))
else:
    st.info("Run `python analyses/01_chembl_descriptive.py` to populate this page.")

st.subheader("Setmelanotide binding mode")
st.markdown(
    "Key orthosteric contacts (from PDB 7PIU): **Ca²⁺ coordination**, "
    "**H-bond to Glu100³·³³**, **hydrophobic cluster with His264⁶·⁵⁴, "
    "Phe284⁷·³⁵, Phe262⁶·⁵²**."
)

overlay_pdb = Path("artifacts/figures/mc4r_overlay.pdb")
if overlay_pdb.exists():
    render_pdb(overlay_pdb.read_text(encoding="utf-8"), height=520)
else:
    st.info("Structural overlay not yet generated.")
