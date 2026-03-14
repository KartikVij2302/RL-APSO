"""Compare SPSO/ARPSO variants under different search-space sizes.

Uses RL environments directly from:
- SPSO/rl_spso.py (RLSPSOEnv)
- ARPSO/rl_arpso.py (RLARPSOEnv)

Compares:
- SPSO vs RL-SPSO
- ARPSO vs RL-ARPSO

For area sizes: 25x25, 50x50, 75x75, 100x100
For fixed swarm size: 5 particles

Outputs a table with metrics:
- Average source seeking time (mu_Ts)
- Average iterations (mu_I)
- Average swarm distance (mu_SD)

Saves the table to CSV and prints it to stdout.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SPSO_DIR = os.path.join(REPO_ROOT, "SPSO")
ARPSO_DIR = os.path.join(REPO_ROOT, "ARPSO")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SPSO_DIR not in sys.path:
    sys.path.insert(0, SPSO_DIR)
if ARPSO_DIR not in sys.path:
    sys.path.insert(0, ARPSO_DIR)

from apso_rl_agent.PPO import PPOAgent
from SPSO.spso import SPSO
from SPSO.rl_spso import RLSPSOConfig, RLSPSOEnv
from ARPSO.arpso import ARPSO_SourceSeeker
from ARPSO.rl_arpso import RLARPSOEnv


@dataclass
class Config:
    areas: List[float]
    runs: int
    max_iter: int
    swarm_size: int
    seed: int
    deterministic: bool
    random_source: bool


def _sample_source(rng: np.random.Generator, side: float, random_source: bool) -> np.ndarray:
    if random_source:
        low = np.array([0.0, 0.0], dtype=float)
        high = np.array([side, side], dtype=float)
        return rng.uniform(low=low, high=high)
    return np.array([side / 2.0, side / 2.0], dtype=float)


def _run_spso_baseline_one(
    side: float,
    source: np.ndarray,
    swarm_size: int,
    max_iter: int,
    seed: int,
) -> Tuple[float, int, float]:
    np.random.seed(seed)
    spso = SPSO(
        n_particles=int(swarm_size),
        side_length=float(side),
        omega=0.721,
        c1=1.193,
        c2=1.193,
        T=1.0,
        speed=10.0,
    )
    spso.set_source(source)

    found = False
    iterations_used = int(max_iter)
    for k in range(1, int(max_iter) + 1):
        if spso.step():
            found = True
            iterations_used = int(k)
            break

    swarm_distance = float(sum(p.dist_travelled for p in spso.particles))
    if found:
        finder = min(spso.particles, key=lambda p: np.linalg.norm(p.position - spso.source))
        ts = float(finder.dist_travelled) / 10.0
    else:
        ts = swarm_distance / 10.0

    return float(ts), int(iterations_used), float(swarm_distance)


def _run_rl_spso_one(
    side: float,
    source: np.ndarray,
    swarm_size: int,
    max_iter: int,
    seed: int,
    agent: PPOAgent,
    deterministic: bool,
) -> Tuple[float, int, float]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = RLSPSOConfig(
        side_length=float(side),
        n_particles=int(swarm_size),
        max_iter=int(max_iter),
    )
    env = RLSPSOEnv(cfg=cfg, seed=seed)

    state = env.reset(source_pos=np.asarray(source, dtype=float), n_particles=int(swarm_size))
    done = False
    info: Dict = {}

    for _ in range(int(max_iter)):
        if deterministic:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32)
                action = agent.policy_old.actor(s).cpu().numpy()
        else:
            action, _ = agent.select_action(state)
        state, _reward, done, info = env.step(action)
        if done:
            break

    assert env.spso is not None
    swarm_distance = float(sum(p.dist_travelled for p in env.spso.particles))
    found = bool(info.get("found", False))
    if found:
        finder = min(env.spso.particles, key=lambda p: np.linalg.norm(p.position - env.spso.source))
        ts = float(finder.dist_travelled) / 10.0
    else:
        ts = swarm_distance / 10.0

    return float(ts), int(env.current_iter), float(swarm_distance)


def _run_arpso_baseline_one(
    side: float,
    source: np.ndarray,
    swarm_size: int,
    max_iter: int,
    seed: int,
) -> Tuple[float, int, float]:
    np.random.seed(seed)
    seeker = ARPSO_SourceSeeker(
        bounds=(np.array([0.0, 0.0], dtype=float), np.array([side, side], dtype=float)),
        source_pos=np.asarray(source, dtype=float),
        num_particles=int(swarm_size),
        c1=1.5,
        c2=1.5,
        c3=0.0,
        wi=0.7,
        T=1.0,
        obstacles=[],
        termination_dist=0.1,
        seed=seed,
    )

    ts, iters, sd, _found = seeker.run_single(max_iter=int(max_iter), param_scheduler=None)
    return float(ts), int(iters), float(sd)


def _run_rl_arpso_one(
    side: float,
    source: np.ndarray,
    swarm_size: int,
    max_iter: int,
    seed: int,
    agent: PPOAgent,
    deterministic: bool,
) -> Tuple[float, int, float]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    bounds = (np.array([0.0, 0.0], dtype=float), np.array([side, side], dtype=float))
    env = RLARPSOEnv(
        source_pos=np.asarray(source, dtype=float),
        bounds=bounds,
        obstacles=[],
        num_particles=int(swarm_size),
        max_iter=int(max_iter),
    )
    state = env.reset(source_pos=np.asarray(source, dtype=float), num_particles=int(swarm_size), obstacles=[])

    done = False
    for _ in range(int(max_iter)):
        if deterministic:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32)
                action = agent.policy_old.actor(s).cpu().numpy()
        else:
            action, _ = agent.select_action(state)
        state, _reward, done, _info = env.step(action)
        if done:
            break

    assert env.arpso is not None
    swarm_distance = float(sum(p.dist_travelled for p in env.arpso.particles))
    min_dist = float(min(np.linalg.norm(p.x - env.source_pos) for p in env.arpso.particles))
    found = min_dist <= float(env.arpso.termination_dist)

    if found:
        finder = min(env.arpso.particles, key=lambda p: np.linalg.norm(p.x - env.source_pos))
        ts = float(finder.dist_travelled) / 10.0
    else:
        ts = swarm_distance / 10.0

    return float(ts), int(env.current_iter), float(swarm_distance)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SPSO/RL-SPSO and ARPSO/RL-ARPSO over search-space sizes "
            "for fixed swarm size 5 (default)."
        )
    )
    parser.add_argument("--spso-model-path", type=str, required=True, help="Path to RL-SPSO PPO model")
    parser.add_argument("--arpso-model-path", type=str, required=True, help="Path to RL-ARPSO PPO model")

    parser.add_argument("--areas", nargs="+", type=float, default=[25.0, 50.0, 75.0, 100.0])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--swarm-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--random-source", action="store_true")
    parser.add_argument("--out-csv", type=str, default="results/area_size_variant_table.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        areas=[float(a) for a in args.areas],
        runs=int(args.runs),
        max_iter=int(args.max_iter),
        swarm_size=int(args.swarm_size),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        random_source=bool(args.random_source),
    )

    if cfg.swarm_size != 5:
        print("Warning: requested fixed swarm size is 5; running with provided value", cfg.swarm_size)

    rl_spso_agent = PPOAgent(state_dim=7, action_dim=2, lr=3e-4)
    rl_spso_agent.load(str(args.spso_model_path))

    rl_arpso_agent = PPOAgent(state_dim=12, action_dim=4, lr=3e-4)
    rl_arpso_agent.load(str(args.arpso_model_path))

    rng = np.random.default_rng(cfg.seed)
    rows: List[Dict[str, float]] = []

    for area in cfg.areas:
        spso_ts, spso_i, spso_sd = [], [], []
        rl_spso_ts, rl_spso_i, rl_spso_sd = [], [], []
        arpso_ts, arpso_i, arpso_sd = [], [], []
        rl_arpso_ts, rl_arpso_i, rl_arpso_sd = [], [], []

        for run_idx in range(cfg.runs):
            run_seed = int(cfg.seed + int(area * 1000) + run_idx)
            source = _sample_source(rng=rng, side=float(area), random_source=cfg.random_source)

            ts, iters, sd = _run_spso_baseline_one(
                side=float(area),
                source=source,
                swarm_size=cfg.swarm_size,
                max_iter=cfg.max_iter,
                seed=run_seed,
            )
            spso_ts.append(ts)
            spso_i.append(iters)
            spso_sd.append(sd)

            ts, iters, sd = _run_rl_spso_one(
                side=float(area),
                source=source,
                swarm_size=cfg.swarm_size,
                max_iter=cfg.max_iter,
                seed=run_seed,
                agent=rl_spso_agent,
                deterministic=cfg.deterministic,
            )
            rl_spso_ts.append(ts)
            rl_spso_i.append(iters)
            rl_spso_sd.append(sd)

            ts, iters, sd = _run_arpso_baseline_one(
                side=float(area),
                source=source,
                swarm_size=cfg.swarm_size,
                max_iter=cfg.max_iter,
                seed=run_seed,
            )
            arpso_ts.append(ts)
            arpso_i.append(iters)
            arpso_sd.append(sd)

            ts, iters, sd = _run_rl_arpso_one(
                side=float(area),
                source=source,
                swarm_size=cfg.swarm_size,
                max_iter=cfg.max_iter,
                seed=run_seed,
                agent=rl_arpso_agent,
                deterministic=cfg.deterministic,
            )
            rl_arpso_ts.append(ts)
            rl_arpso_i.append(iters)
            rl_arpso_sd.append(sd)

        area_label = f"{int(area)}x{int(area)}" if float(area).is_integer() else f"{area}x{area}"
        rows.extend(
            [
                {
                    "Area": area_label,
                    "Method": "SPSO",
                    "mu_I": _mean(spso_i),
                    "mu_Ts": _mean(spso_ts),
                    "mu_SD": _mean(spso_sd),
                },
                {
                    "Area": area_label,
                    "Method": "RL-SPSO",
                    "mu_I": _mean(rl_spso_i),
                    "mu_Ts": _mean(rl_spso_ts),
                    "mu_SD": _mean(rl_spso_sd),
                },
                {
                    "Area": area_label,
                    "Method": "ARPSO",
                    "mu_I": _mean(arpso_i),
                    "mu_Ts": _mean(arpso_ts),
                    "mu_SD": _mean(arpso_sd),
                },
                {
                    "Area": area_label,
                    "Method": "RL-ARPSO",
                    "mu_I": _mean(rl_arpso_i),
                    "mu_Ts": _mean(rl_arpso_ts),
                    "mu_SD": _mean(rl_arpso_sd),
                },
            ]
        )

    _write_csv(str(args.out_csv), rows)

    print("\nArea       Method       mu_I      mu_Ts      mu_SD")
    for row in rows:
        print(
            f"{row['Area']:<10} {row['Method']:<11} {row['mu_I']:>8.3f} {row['mu_Ts']:>10.3f} {row['mu_SD']:>10.3f}"
        )
    print(f"\nSaved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
