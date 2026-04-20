"""GUI TOML editor component."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render(path: Path, *, key: str = "toml_editor") -> str:
    default = path.read_text(encoding="utf-8") if path.exists() else ""
    text = st.text_area(
        "REINVENT 4 TOML",
        value=default,
        height=420,
        key=key,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Save to disk", key=f"{key}_save") and path is not None:
            path.write_text(text, encoding="utf-8")
            st.success(f"Wrote {path}")
    with col_b:
        st.download_button(
            label="Download TOML",
            data=text.encode("utf-8"),
            file_name=path.name if path else "config.toml",
            mime="text/toml",
            key=f"{key}_dl",
        )
    return text
