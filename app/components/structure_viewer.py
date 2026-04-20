"""3D structure viewer wrapper around py3Dmol / streamlit-molstar."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_pdb(pdb_text: str, *, height: int = 420) -> None:
    try:
        import py3Dmol
    except ImportError:
        st.warning("py3Dmol not available.")
        return
    view = py3Dmol.view(width=720, height=height)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    st.components.v1.html(view.write_html(), height=height + 40, scrolling=False)


def render_molstar(pdb_file: Path, *, height: int = 480) -> None:
    try:
        from streamlit_molstar import st_molstar
    except ImportError:
        st.warning("streamlit-molstar not available; falling back to py3Dmol.")
        render_pdb(pdb_file.read_text(encoding="utf-8"), height=height)
        return
    st_molstar(str(pdb_file), height=height, key=pdb_file.name)
