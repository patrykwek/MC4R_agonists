"""py3Dmol / PyMOL-open-source structure renderers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def py3dmol_complex_html(pdb_block: str, ligand_sdf: str, *, width: int = 480, height: int = 360) -> str:
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_block, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addModel(ligand_sdf, "sdf")
    view.setStyle({"model": -1}, {"stick": {"colorscheme": "yellowCarbon"}})
    view.zoomTo()
    return view.write_html()


def pymol_render_png(
    session_pml: Path,
    output_png: Path,
    *,
    width: int = 1200,
    height: int = 900,
) -> Path:
    """Render a figure with open-source PyMOL (``pymol`` binary)."""
    import shutil

    exe = shutil.which("pymol")
    if exe is None:
        raise RuntimeError("PyMOL not found on $PATH (install PyMOL-open-source).")
    cmd = [
        exe,
        "-cqr",
        str(session_pml),
        "--",
        "-W",
        str(width),
        "-H",
        str(height),
        str(output_png),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_png


def write_standard_pml(target: Path, pdb_path: Path, ligand_path: Path, png_path: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"""\
load {pdb_path}, receptor
load {ligand_path}, ligand
hide everything
show cartoon, receptor
color grey80, receptor
show sticks, ligand
color yellow, ligand
zoom ligand, 5
ray 1200, 900
png {png_path}
quit
""",
        encoding="utf-8",
    )
    return target
