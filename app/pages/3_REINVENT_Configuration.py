"""Interactive TOML editor."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.components.toml_editor import render

st.header("REINVENT 4 Configuration")

configs = sorted(Path("configs").glob("*.toml"))
if not configs:
    st.error("No TOMLs in configs/.")
    st.stop()

labels = {c.stem: c for c in configs}
choice = st.selectbox("Pick a config", list(labels))
render(labels[choice], key=f"editor_{choice}")
