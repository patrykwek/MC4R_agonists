"""Chemical space + per-molecule scorecards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.components.chemical_space_plot import render

st.header("Results Explorer")

candidates_csv = Path("artifacts/top10_candidates.csv")
if not candidates_csv.exists():
    st.info("Run `python analyses/11_compound_prioritization.py` first.")
    st.stop()

df = pd.read_csv(candidates_csv)
st.subheader("Top candidates")
st.dataframe(df)

coords = np.column_stack(
    [np.arange(len(df), dtype=float), df.get("vina_score", pd.Series(dtype=float)).astype(float)]
)
render(coords[:, 0], coords[:, 1], ["candidate"] * len(df), df["smiles"].astype(str))

st.subheader("Filter")
max_vina = st.slider("Maximum Vina score (kcal/mol)", -14.0, -5.0, -9.0, 0.1)
min_pki = st.slider("Minimum QSAR pKi(MC4R)", 5.0, 10.0, 7.0, 0.1)
filtered = df[(df["vina_score"] <= max_vina) & (df.get("predicted_pki_mc4r", 0) >= min_pki)]
st.metric("Matching candidates", len(filtered))
