"""OpenMM MD stability testing of the top 10 candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mc4gen._logging import get_logger
from mc4gen.data import structures
from mc4gen.md import analysis, ligand_params, openmm_runner, system_builder
from mc4gen.utils.smiles import embed_conformer

log = get_logger(__name__)

INPUT_CSV = Path("artifacts/top10_candidates.csv")
OUT_CSV = Path("artifacts/top10_md.csv")
POCKET_RESIDUES = (100, 111, 112, 115, 116, 126, 181, 184, 193, 256, 260, 262, 264, 284)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    rows: list[dict[str, float | str]] = []
    for idx, candidate in df.iterrows():
        smi = str(candidate["smiles"])
        work_dir = Path(f"runs/md/cand_{idx}")
        work_dir.mkdir(parents=True, exist_ok=True)

        mol = embed_conformer(smi)
        if mol is None:
            log.warning("Failed to embed %s", smi[:40])
            continue
        mol_block = Path(work_dir / "ligand.mol")
        from rdkit import Chem

        mol_block.write_text(Chem.MolToMolBlock(mol), encoding="utf-8")

        ligand_topology = ligand_params.parameterize(
            mol_block.read_text(encoding="utf-8"),
            work_dir,
            name="LIG",
        )
        receptor_pdb = structures.prepare_receptor("7PIU").with_suffix(".pdb")
        membrane = system_builder.build_popc_system(
            receptor_pdb,
            ligand_topology.mol2,
            work_dir,
            name=f"cand_{idx}",
        )
        replicas = openmm_runner.run_all(
            membrane.prmtop, membrane.inpcrd, work_dir / "replicas"
        )
        summary = analysis.summarize(
            membrane.prmtop,
            [r.trajectory for r in replicas],
            pocket_residues=POCKET_RESIDUES,
        )
        rows.append(
            {
                "smiles": smi,
                "ligand_rmsd_mean": summary.ligand_rmsd_mean,
                "ligand_rmsd_std": summary.ligand_rmsd_std,
                "pocket_rmsd_mean": summary.pocket_backbone_rmsd_mean,
                "ca_contact_fraction": summary.ca_contact_fraction,
                "frames": summary.n_frames,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    log.info("Wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
