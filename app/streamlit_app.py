"""Streamlit multipage root.

Run with ``streamlit run app/streamlit_app.py``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="mc4gen — MC4R de novo design",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬",
)

_CSS_PATH = Path(__file__).parent / "assets" / "style.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

st.title("mc4gen")
st.markdown(
    "**mc4gen** is a fully open-source REINVENT 4 pipeline for de novo design of "
    "MC4R-biased small-molecule agonists. Use the sidebar to navigate the seven "
    "workflow pages."
)
st.info(
    "No commercial software is used anywhere in this app: AutoDock Vina + smina for "
    "docking, open-source ChEMBL/ZINC20 data, RDKit + meeko + dimorphite-dl for "
    "ligand prep, OpenMM for MD, and AiZynthFinder for retrosynthesis."
)
