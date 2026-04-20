"""Download and prepare the five MC4R PDB structures."""

from __future__ import annotations

from mc4gen._logging import get_logger
from mc4gen.data.structures import STRUCTURES, prepare_receptor

log = get_logger(__name__)


def main() -> None:
    for pdb_id in STRUCTURES:
        pdbqt_path = prepare_receptor(pdb_id)
        log.info("%s -> %s", pdb_id, pdbqt_path)


if __name__ == "__main__":
    main()
