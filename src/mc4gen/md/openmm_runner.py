"""OpenMM runner for equilibration + 100 ns production per replica."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mc4gen._logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RunConfig:
    replicas: int = 3
    timestep_fs: float = 4.0
    production_ns: float = 100.0
    temperature_k: float = 310.0
    pressure_bar: float = 1.0
    report_ps: float = 100.0
    platform: str = "CUDA"
    precision: str = "mixed"


@dataclass(frozen=True, slots=True)
class ReplicaResult:
    replica: int
    seed: int
    trajectory: Path
    log: Path


def run_replica(
    prmtop_path: Path,
    inpcrd_path: Path,
    out_dir: Path,
    *,
    replica: int,
    seed: int,
    config: RunConfig,
) -> ReplicaResult:
    from openmm import (
        LangevinMiddleIntegrator,
        MonteCarloMembraneBarostat,
        Platform,
        unit,
    )
    from openmm.app import (
        AmberInpcrdFile,
        AmberPrmtopFile,
        DCDReporter,
        HBonds,
        PME,
        Simulation,
        StateDataReporter,
    )

    prmtop = AmberPrmtopFile(str(prmtop_path))
    inpcrd = AmberInpcrdFile(str(inpcrd_path))
    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds,
        hydrogenMass=1.5 * unit.amu,
    )
    barostat = MonteCarloMembraneBarostat(
        1.0 * unit.bar,
        0.0 * unit.bar * unit.nanometer,
        config.temperature_k * unit.kelvin,
        MonteCarloMembraneBarostat.XYIsotropic,
        MonteCarloMembraneBarostat.ZFree,
    )
    system.addForce(barostat)
    integrator = LangevinMiddleIntegrator(
        config.temperature_k * unit.kelvin,
        1.0 / unit.picosecond,
        config.timestep_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(seed)

    platform_obj = Platform.getPlatformByName(config.platform)
    props = {"Precision": config.precision} if config.platform == "CUDA" else {}
    simulation = Simulation(prmtop.topology, system, integrator, platform_obj, props)
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(config.temperature_k * unit.kelvin, seed)

    production_steps = int(config.production_ns * 1_000_000 / config.timestep_fs)
    report_interval = int(config.report_ps * 1_000 / config.timestep_fs)

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / f"replica_{replica}.dcd"
    log_path = out_dir / f"replica_{replica}.log"
    simulation.reporters.append(DCDReporter(str(traj_path), report_interval))
    simulation.reporters.append(
        StateDataReporter(
            str(log_path),
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
        )
    )
    simulation.step(production_steps)
    return ReplicaResult(replica=replica, seed=seed, trajectory=traj_path, log=log_path)


def run_all(
    prmtop_path: Path,
    inpcrd_path: Path,
    out_dir: Path,
    *,
    config: RunConfig | None = None,
) -> list[ReplicaResult]:
    cfg = config or RunConfig()
    results: list[ReplicaResult] = []
    for i in range(cfg.replicas):
        seed = 0xDEAD + i * 7
        log.info("Running MD replica %d (seed=%d)", i, seed)
        results.append(run_replica(prmtop_path, inpcrd_path, out_dir, replica=i, seed=seed, config=cfg))
    return results
