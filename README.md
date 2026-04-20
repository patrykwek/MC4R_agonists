# mc4gen

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![100% open-source](https://img.shields.io/badge/deps-100%25%20open--source-success.svg)](#why-fully-open-source)

Extension of my previous publications on MC4R agonists. Design of small-molecule agonists of the **melanocortin-4 receptor (MC4R)** that are biased for MC4R over MC1R, MC3R and MC5R. Built around [REINVENT 4](https://github.com/MolecularAI/REINVENT4) with custom structure-based scoring plugins, a four-receptor QSAR panel trained on ChEMBL, AutoDock Vina docking against five public cryo-EM structures, and OpenMM MD stability testing.

## Install

Dependencies are grouped into lightweight core + optional extras. The core
is install-from-PyPI only (numpy / scipy / pandas / scikit-learn / lightgbm /
rdkit / matplotlib / plotly / py3Dmol / biopython / requests / joblib /
jinja2).

```bash
pip install -e .                      # base (numerics + rdkit + reporting)
pip install -e ".[docking]"           # + AutoDock Vina, meeko, PLIP, ProLIF
pip install -e ".[md]"                # + OpenMM, pdbfixer, MDAnalysis, propka
pip install -e ".[reinvent]"          # + REINVENT 4 (git) + ChemProp
pip install -e ".[retrosynth]"        # + AiZynthFinder
pip install -e ".[rascore]"           # + RAscore (git)
pip install -e ".[app]"               # + Streamlit companion
pip install -e ".[dev]"               # + ruff, mypy, pytest, hypothesis
pip install -e ".[full]"              # everything above
```


## Methods

| Component | Implementation | Replaces |
|---|---|---|
| Generator | REINVENT 4 de novo prior | — |
| RL | REINVENT / AHC / Augmented Memory | — |
| Docking | AutoDock Vina 1.2 + smina | Glide SP |
| MM-GBSA rescoring | `gmx_MMPBSA` + OpenMM | Prime MM-GBSA |
| Ligand prep | `meeko` + `dimorphite-dl` + RDKit | LigPrep |
| Receptor prep | `pdbfixer` + `propka` + ADFR | Protein Prep Wizard |
| Interactions | PLIP + ProLIF | MOE / GRID |
| MD | OpenMM + AMBER14 + Lipid17 | Desmond |
| Retrosynthesis | AiZynthFinder | — |
| Synthetic accessibility | RAscore | — |

## Design philosophy

**REINVENT-4-native.** Every structure-based and ML scoring term is a REINVENT 4 plugin registered through `project.entry-points` — no monkey-patching, no forked branches. Users can swap individual components without touching the RL core.

**Selectivity-first objective.** MC4R affinity alone has driven MC1R-mediated side effects (skin hyperpigmentation) in clinical agents; `mc4gen` scores a four-receptor panel by default and rewards selectivity deltas, not absolute potency.

**Structure-ensemble averaging.** The Vina score is averaged across five independent MC4R cryo-EM structures (7AUE, 7PIU, 7PIV, 6W25, 7F53). Candidates are required to score well on at least two.


## Limitations

- Vina rank-ordering is noisier than Glide SP; I mitigate with ensemble averaging over five structures but report the Spearman correlation to experimental pKi on held-out MC4R actives prominently.
- The QSAR panel is bounded by ChEMBL coverage. MC3R and MC5R datasets are smaller and their applicability domains narrower; we flag out-of-domain predictions and exclude them from headline claims.
- **Biased signaling is not predicted.** Setmelanotide's Gq/11 vs. Gs bias is a functional property not captured by structure-based scoring; `mc4gen` does not attempt to score bias.
- Docking pose ↔ functional agonism is an unvalidated assumption for any structure-based generative protocol, commercial or not.

## Architecture

```mermaid
flowchart TD
    A[ChEMBL 34] --> B[QSAR panel<br/>MC1/3/4/5R]
    C[PDB 7AUE / 7PIU / 7PIV<br/>6W25 / 7F53] --> D[Receptor prep<br/>pdbfixer + propka + ADFR]
    D --> E[AutoDock Vina]
    B --> F[REINVENT 4 agent]
    E --> F
    G[Chemotype SMARTS +<br/>ZINC20 / Enamine REAL] --> F
    F --> H[Raw molecules<br/>~270k]
    H --> I[11-stage<br/>prioritization funnel]
    I --> J[Top 10 candidates]
    J --> K[gmx_MMPBSA +<br/>OpenMM 100 ns × 3]
    K --> L[AiZynthFinder +<br/>vendor lookup]
```

## Run order

1. `python analyses/01_chembl_descriptive.py`
2. `python analyses/02_structure_preparation.py`
3. `python analyses/03_qsar_training_and_validation.py` *(gate: MC4R R² ≥ 0.5)*
4. `python analyses/04_reinvent_runs.py` *(21 runs)*
5. `python analyses/11_compound_prioritization.py`
6. `python analyses/12_docking_rescoring.py`
7. `python analyses/13_md_stability.py`
8. `python analyses/14_synthesis_candidate_generation.py`
9. `python analyses/15_benchmark_vs_setmelanotide.py`

## License

MIT. Every runtime dependency is permissively licensed (Apache 2.0, MIT, BSD, LGPL, or GPL-compatible).
