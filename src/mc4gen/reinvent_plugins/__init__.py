"""Custom REINVENT 4 scoring components registered via entry points.

All components are plain callables that accept a list of SMILES and return a
dataclass compatible with :mod:`mc4gen.reinvent_plugins.dtos`.
"""

from mc4gen.reinvent_plugins.components.calcium_coordination import CalciumCoordinationPredictor
from mc4gen.reinvent_plugins.components.chemotype_novelty import ChemotypeNoveltyFilter
from mc4gen.reinvent_plugins.components.mc4r_docking_vina import MC4RDockingVina
from mc4gen.reinvent_plugins.components.melanocortin_selectivity import (
    MelanocortinSelectivityPanel,
)
from mc4gen.reinvent_plugins.components.rascore_wrapper import RAScoreComponent
from mc4gen.reinvent_plugins.components.vendor_novelty import VendorNoveltyFilter

__all__ = [
    "CalciumCoordinationPredictor",
    "ChemotypeNoveltyFilter",
    "MC4RDockingVina",
    "MelanocortinSelectivityPanel",
    "RAScoreComponent",
    "VendorNoveltyFilter",
]
