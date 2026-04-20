"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture(scope="session")
def tmp_cache_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    path = tmp_path_factory.mktemp("mc4gen_cache")
    prior = os.environ.get("MC4GEN_CACHE_ROOT")
    os.environ["MC4GEN_CACHE_ROOT"] = str(path)
    try:
        yield path
    finally:
        if prior is None:
            os.environ.pop("MC4GEN_CACHE_ROOT", None)
        else:
            os.environ["MC4GEN_CACHE_ROOT"] = prior


@pytest.fixture
def aspirin_smiles() -> str:
    return "CC(=O)OC1=CC=CC=C1C(=O)O"


@pytest.fixture
def ibuprofen_smiles() -> str:
    return "CC(C)Cc1ccc(C(C)C(=O)O)cc1"


@pytest.fixture
def setmelanotide_smiles() -> str:
    from mc4gen.pipeline.prioritize import SETMELANOTIDE_SMILES

    return SETMELANOTIDE_SMILES
