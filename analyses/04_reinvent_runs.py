"""Execute the 21 primary REINVENT 4 runs."""

from __future__ import annotations

from pathlib import Path

from mc4gen._logging import get_logger
from mc4gen.pipeline.run_reinvent import run_reinvent_stage

log = get_logger(__name__)

CONFIG_DIR = Path("configs")
OUT_DIR = Path("runs")

PRIMARY_CONFIGS = [
    "stage_2_rl_7aue.toml",
    "stage_2_rl_7piu.toml",
    "stage_2_rl_7piv.toml",
    "stage_2_rl_6w25.toml",
    "stage_2_rl_7f53.toml",
    "stage_3_rl_strict_constraints.toml",
    "stage_3_rl_relaxed_constraints.toml",
    "ahc_variant.toml",
    "augmented_memory_variant.toml",
]


def main() -> None:
    for config_name in PRIMARY_CONFIGS:
        config_path = CONFIG_DIR / config_name
        out_sub = OUT_DIR / config_path.stem
        result = run_reinvent_stage(config_path, output_dir=out_sub)
        log.info(
            "Run %s -> %s (rc=%d)",
            config_name,
            result.csv_path,
            result.return_code,
        )


if __name__ == "__main__":
    main()
