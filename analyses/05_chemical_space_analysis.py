"""UMAP/t-SNE of generated vs. training vs. known actives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.manifold import TSNE

from mc4gen._logging import get_logger
from mc4gen.reporting.plots import chemical_space_scatter
from mc4gen.utils.fingerprints import morgan_count

log = get_logger(__name__)

OUT_FIG = Path("artifacts/figures/chemical_space_tsne.png")


def _featurize(smiles: pd.Series) -> np.ndarray:
    fps: list[np.ndarray] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(morgan_count(mol))
    return np.stack(fps) if fps else np.zeros((0, 2048), dtype=np.int32)


def main() -> None:
    generated = pd.read_csv("runs/stage_2_7piu/molecules.csv")["SMILES"]
    training = pd.read_csv("artifacts/qsar_training_smiles.csv")["smiles"]
    known = pd.read_csv("artifacts/mc4r_chembl.csv")["smiles"]

    all_smiles = pd.concat([generated, training, known], ignore_index=True)
    labels = ["generated"] * len(generated) + ["training"] * len(training) + ["known"] * len(known)

    X = _featurize(all_smiles)
    tsne = TSNE(n_components=2, perplexity=30, metric="jaccard", init="random", random_state=0)
    coords = tsne.fit_transform(X.astype(np.float32))
    chemical_space_scatter(coords, labels, OUT_FIG)
    log.info("Wrote %s", OUT_FIG)


if __name__ == "__main__":
    main()
