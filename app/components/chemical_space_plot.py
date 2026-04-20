"""Interactive chemical-space plot (Plotly)."""

from __future__ import annotations

from typing import Sequence

import plotly.express as px
import streamlit as st


def render(x: Sequence[float], y: Sequence[float], labels: Sequence[str], smiles: Sequence[str]) -> None:
    import pandas as pd

    df = pd.DataFrame({"x": x, "y": y, "label": labels, "smiles": smiles})
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        hover_data={"smiles": True, "x": False, "y": False},
        opacity=0.6,
        height=560,
    )
    fig.update_traces(marker={"size": 6})
    st.plotly_chart(fig, use_container_width=True)
