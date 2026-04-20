"""Cross-run sensitivity ablations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.header("Sensitivity Analysis")

tabs = st.tabs(["Structure", "RL strategy", "Constraints"])
for tab, name in zip(tabs, ("structure_sensitivity", "rl_strategy_sensitivity", "constraint_sensitivity"), strict=True):
    with tab:
        path = Path(f"artifacts/tables/{name}.tex")
        if not path.exists():
            st.info(f"Run the matching analysis to populate {path}.")
            continue
        st.code(path.read_text(encoding="utf-8"), language="latex")

st.markdown(
    "**Goodman–Bacon-style decomposition** splits the cross-structure variance in "
    "mean score into (i) within-structure variation across RL seeds and (ii) "
    "between-structure variation; see `analyses/06_structure_sensitivity.py`."
)
