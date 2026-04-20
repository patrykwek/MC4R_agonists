"""Launch REINVENT 4 subprocesses and monitor reward curves."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

st.header("Run & Monitor")

config_path = st.selectbox(
    "Configuration",
    sorted(Path("configs").glob("*.toml")),
    format_func=lambda p: p.name,
)
output_dir = Path("runs") / Path(config_path).stem
st.code(
    f"reinvent -f toml -l {output_dir / 'reinvent.log'} {config_path}",
    language="bash",
)

if st.button("Start run (subprocess)"):
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        ["reinvent", "-f", "toml", "-l", str(output_dir / "reinvent.log"), str(config_path)]
    )
    st.success(f"Launched REINVENT; watching {output_dir}")

progress_csv = output_dir / "progress.csv"
if progress_csv.exists():
    df = pd.read_csv(progress_csv)
    st.line_chart(df.set_index("step")[["score_mean"]])
else:
    st.info("No progress CSV yet; check back after a few epochs.")
