"""Evaluate baseline ARPSO vs RL-ARPSO across swarm sizes.

Metrics use the same formulas already implemented in this repository:
  - Source seeking time Ts:
      if found: finder.dist_travelled / 10.0
      else:     total_swarm_distance / 10.0
  - Iterations I:
      iterations used until found, else max_iter
  - Swarm distance SD:
      sum(p.dist_travelled for p in particles)

Outputs:
  - CSV summary by swarm size
  - One figure with 3 well-labeled comparison subplots
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from apso_rl_agent.PPO import PPOAgent

try:
    from .arpso import ARPSO_SourceSeeker
except Exception:  # pragma: no cover
    from arpso import ARPSO_SourceSeeker


@dataclass
class EvalConfig:
    max_iter: int = 400
    side_length: float = 100.0
    swarm_size_low: int = 5
    swarm_size_high: int = 30
    episodes_per_n: int = 100
    seed: int = 42

    use_obstacles: bool = False
    n_obstacles_low: int = 1
    n_obstacles_high: int = 4

    c1_base: float = 1.5
    c2_base: float = 1.5
    c3_base: float = 1.0
    wi_base: float = 0.7
    T: float = 1.0
    obstacle_margin: float = 4.0
    termination_dist: float = 0.1


def _sample_random_obstacles(
    bounds: Tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    n_obstacles: int,
) -> List[Tuple[np.ndarray, float]]:
    lo, hi = bounds
    obstacles: List[Tuple[np.ndarray, float]] = []
    for _ in range(n_obstacles):
        center = rng.uniform(low=lo + 10.0, high=hi - 10.0)
        radius = float(rng.uniform(4.0, 10.0))
        obstacles.append((center, radius))
    return obstacles


def _state_from_swarm(
    seeker: ARPSO_SourceSeeker,
    source_pos: np.ndarray,
    obstacles: Sequence[Tuple[np.ndarray, float]],
    prev_min_dist: float,
    cumulative_time: float,
    current_iter: int,
    max_iter: int,
) -> np.ndarray:
    min_dist = float(min(np.linalg.norm(p.x - source_pos) for p in seeker.particles))
    avg_vel = float(np.mean([np.linalg.norm(p.v) for p in seeker.particles]))
    avg_omega = float(np.mean([p.last_omega for p in seeker.particles]))
    time_left = 1.0 - (current_iter / max(max_iter, 1))

    has_obstacles = 1.0 if obstacles else 0.0
    nearest_obs = 1.0
    if obstacles:
        dists: List[float] = []
        for p in seeker.particles:
            for center, radius in obstacles:
                d = np.linalg.norm(p.x - np.asarray(center, dtype=float)) - float(radius)
                dists.append(float(d))
        nearest_obs = float(np.clip(np.min(dists) / 50.0, 0.0, 1.0))

    state = np.array(
        [
            np.clip(min_dist / 150.0, 0.0, 2.0),
            np.clip(prev_min_dist / 150.0, 0.0, 2.0),
            np.clip(avg_vel / 20.0, 0.0, 2.0),
            np.clip(cumulative_time / 60.0, 0.0, 2.0),
            np.clip(current_iter / max(max_iter, 1), 0.0, 1.0),
            np.clip(seeker.c1 / 4.0, 0.0, 2.0),
            np.clip(seeker.c2 / 4.0, 0.0, 2.0),
            np.clip(seeker.c3 / 4.0, 0.0, 2.0),
            np.clip(avg_omega / 1.2, 0.0, 1.5),
            time_left,
            has_obstacles,
            nearest_obs,
        ],
        dtype=np.float32,
    )
    return state


def _apply_action_like_training(
    seeker: ARPSO_SourceSeeker,
    action: np.ndarray,
    obstacles: Sequence[Tuple[np.ndarray, float]],
) -> Tuple[float, float, float, float]:
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

    delta = 0.25
    c1 = seeker.c1 * (1.0 + delta * a[0])
    c2 = seeker.c2 * (1.0 + delta * a[1])
    c3 = seeker.c3 * (1.0 + delta * a[2]) if obstacles else 0.0
    wi = seeker.wi * (1.0 + 0.2 * a[3])

    c1 = float(np.clip(c1, 0.05, 3.5))
    c2 = float(np.clip(c2, 0.05, 3.5))
    c3 = float(np.clip(c3, 0.0, 3.5))
    wi = float(np.clip(wi, 0.05, 1.2))

    if not np.isfinite([c1, c2, c3, wi]).all():
        c1 = 1.5
        c2 = 1.5
        c3 = 1.0 if obstacles else 0.0
        wi = 0.7

    return c1, c2, c3, wi


def _source_seeking_time_from_particles(particles, source_pos: np.ndarray, found: bool) -> float:
    if found:
        finder = min(particles, key=lambda p: np.linalg.norm(p.x - source_pos))
        return float(finder.dist_travelled) / 10.0
    sd = float(sum(p.dist_travelled for p in particles))
    return sd / 10.0


def _swarm_distance(particles) -> float:
    return float(sum(p.dist_travelled for p in particles))


def _run_baseline_one(
    cfg: EvalConfig,
    bounds: Tuple[np.ndarray, np.ndarray],
    source_pos: np.ndarray,
    obstacles: Sequence[Tuple[np.ndarray, float]],
    num_particles: int,
    seed: int,
) -> Tuple[float, int, float, int]:
    np.random.seed(seed)
    seeker = ARPSO_SourceSeeker(
        bounds=bounds,
        source_pos=source_pos,
        num_particles=num_particles,
        c1=cfg.c1_base,
        c2=cfg.c2_base,
        c3=(cfg.c3_base if obstacles else 0.0),
        wi=cfg.wi_base,
        T=cfg.T,
        obstacles=obstacles,
        obstacle_margin=cfg.obstacle_margin,
        termination_dist=cfg.termination_dist,
        seed=seed,
    )

    found = False
    iterations_used = cfg.max_iter
    for k in range(1, cfg.max_iter + 1):
        found, _min_dist, _step_time = seeker.step()
        if found:
            iterations_used = k
            break

    ts = _source_seeking_time_from_particles(seeker.particles, source_pos=source_pos, found=found)
    sd = _swarm_distance(seeker.particles)
    return ts, int(iterations_used), sd, int(found)


def _run_rl_one(
    cfg: EvalConfig,
    bounds: Tuple[np.ndarray, np.ndarray],
    source_pos: np.ndarray,
    obstacles: Sequence[Tuple[np.ndarray, float]],
    num_particles: int,
    seed: int,
    agent: PPOAgent,
    deterministic: bool,
) -> Tuple[float, int, float, int]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    seeker = ARPSO_SourceSeeker(
        bounds=bounds,
        source_pos=source_pos,
        num_particles=num_particles,
        c1=cfg.c1_base,
        c2=cfg.c2_base,
        c3=(cfg.c3_base if obstacles else 0.0),
        wi=cfg.wi_base,
        T=cfg.T,
        obstacles=obstacles,
        obstacle_margin=cfg.obstacle_margin,
        termination_dist=cfg.termination_dist,
        seed=seed,
    )

    cumulative_time = 0.0
    prev_min_dist = float(min(np.linalg.norm(p.x - source_pos) for p in seeker.particles))
    found = False
    iterations_used = cfg.max_iter

    for t in range(cfg.max_iter):
        state = _state_from_swarm(
            seeker=seeker,
            source_pos=source_pos,
            obstacles=obstacles,
            prev_min_dist=prev_min_dist,
            cumulative_time=cumulative_time,
            current_iter=t,
            max_iter=cfg.max_iter,
        )

        if deterministic:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32)
                action = agent.policy_old.actor(s).cpu().numpy()
        else:
            action, _ = agent.select_action(state)

        c1, c2, c3, wi = _apply_action_like_training(seeker=seeker, action=action, obstacles=obstacles)
        found, min_dist, step_time = seeker.step(c1=c1, c2=c2, c3=c3, wi=wi)
        cumulative_time += float(step_time)
        prev_min_dist = float(min_dist)

        if found:
            iterations_used = t + 1
            break

    ts = _source_seeking_time_from_particles(seeker.particles, source_pos=source_pos, found=found)
    sd = _swarm_distance(seeker.particles)
    return ts, int(iterations_used), sd, int(found)


def _agg_stats(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(
    axis: plt.Axes,
    x: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    rl_mean: np.ndarray,
    rl_std: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    axis.plot(x, baseline_mean, marker="o", linewidth=2.2, color="#1f77b4", label="ARPSO")
    axis.fill_between(
        x,
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        color="#1f77b4",
        alpha=0.16,
    )

    axis.plot(x, rl_mean, marker="s", linewidth=2.2, color="#d62728", label="RL-ARPSO")
    axis.fill_between(
        x,
        rl_mean - rl_std,
        rl_mean + rl_std,
        color="#d62728",
        alpha=0.16,
    )

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.28)
    axis.set_xlim(float(x[0]), float(x[-1]))
    axis.set_xticks(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare baseline ARPSO vs RL-ARPSO across swarm sizes and save CSV + plots."
        )
    )
    p.add_argument("--model-path", type=str, required=True, help="Path to trained RL-ARPSO PPO model")
    p.add_argument("--episodes-per-n", type=int, default=100)
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--swarm-size-low", type=int, default=5)
    p.add_argument("--swarm-size-high", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--side-length", type=float, default=100.0)
    p.add_argument("--fixed-source", nargs=2, type=float, default=[50.0, 50.0])
    p.add_argument("--random-source", action="store_true", help="Sample a random source per run")

    p.add_argument("--use-obstacles", action="store_true", help="Enable random circular obstacles")
    p.add_argument("--deterministic", action="store_true", help="Use policy mean action")

    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--prefix", type=str, default="arpso_vs_rlarpso_swarm_size")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.swarm_size_low) < 5 or int(args.swarm_size_high) > 30:
        raise SystemExit("Swarm size range must be within [5, 30].")
    if int(args.swarm_size_low) > int(args.swarm_size_high):
        raise SystemExit("--swarm-size-low must be <= --swarm-size-high")

    cfg = EvalConfig(
        max_iter=int(args.max_iter),
        side_length=float(args.side_length),
        swarm_size_low=int(args.swarm_size_low),
        swarm_size_high=int(args.swarm_size_high),
        episodes_per_n=int(args.episodes_per_n),
        seed=int(args.seed),
        use_obstacles=bool(args.use_obstacles),
    )

    lo = np.array([0.0, 0.0], dtype=float)
    hi = np.array([cfg.side_length, cfg.side_length], dtype=float)
    bounds = (lo, hi)

    rng = np.random.default_rng(cfg.seed)
    swarm_sizes = list(range(cfg.swarm_size_low, cfg.swarm_size_high + 1))

    agent = PPOAgent(state_dim=12, action_dim=4, lr=3e-4)
    agent.load(str(args.model_path))

    summary_rows: List[Dict[str, float]] = []

    for n in swarm_sizes:
        base_ts_vals: List[float] = []
        base_i_vals: List[float] = []
        base_sd_vals: List[float] = []

        rl_ts_vals: List[float] = []
        rl_i_vals: List[float] = []
        rl_sd_vals: List[float] = []

        for run_idx in range(cfg.episodes_per_n):
            seed_run = cfg.seed + run_idx

            if args.random_source:
                source = rng.uniform(low=lo + 8.0, high=hi - 8.0)
            else:
                source = np.array([float(args.fixed_source[0]), float(args.fixed_source[1])], dtype=float)

            if cfg.use_obstacles:
                n_obs = int(rng.integers(cfg.n_obstacles_low, cfg.n_obstacles_high + 1))
                obstacles = _sample_random_obstacles(bounds=bounds, rng=rng, n_obstacles=n_obs)
            else:
                obstacles = []

            ts_b, i_b, sd_b, _succ_b = _run_baseline_one(
                cfg=cfg,
                bounds=bounds,
                source_pos=source,
                obstacles=obstacles,
                num_particles=n,
                seed=seed_run,
            )
            ts_r, i_r, sd_r, _succ_r = _run_rl_one(
                cfg=cfg,
                bounds=bounds,
                source_pos=source,
                obstacles=obstacles,
                num_particles=n,
                seed=seed_run,
                agent=agent,
                deterministic=bool(args.deterministic),
            )

            base_ts_vals.append(ts_b)
            base_i_vals.append(i_b)
            base_sd_vals.append(sd_b)

            rl_ts_vals.append(ts_r)
            rl_i_vals.append(i_r)
            rl_sd_vals.append(sd_r)

        b_ts_mean, b_ts_std = _agg_stats(base_ts_vals)
        b_i_mean, b_i_std = _agg_stats(base_i_vals)
        b_sd_mean, b_sd_std = _agg_stats(base_sd_vals)

        r_ts_mean, r_ts_std = _agg_stats(rl_ts_vals)
        r_i_mean, r_i_std = _agg_stats(rl_i_vals)
        r_sd_mean, r_sd_std = _agg_stats(rl_sd_vals)

        summary_rows.append(
            {
                "swarm_size": int(n),
                "arpso_mean_time": b_ts_mean,
                "arpso_std_time": b_ts_std,
                "rl_arpso_mean_time": r_ts_mean,
                "rl_arpso_std_time": r_ts_std,
                "arpso_mean_iters": b_i_mean,
                "arpso_std_iters": b_i_std,
                "rl_arpso_mean_iters": r_i_mean,
                "rl_arpso_std_iters": r_i_std,
                "arpso_mean_swarm_dist": b_sd_mean,
                "arpso_std_swarm_dist": b_sd_std,
                "rl_arpso_mean_swarm_dist": r_sd_mean,
                "rl_arpso_std_swarm_dist": r_sd_std,
            }
        )

        print(
            f"[n={n:2d}] ARPSO Ts={b_ts_mean:8.3f}, I={b_i_mean:7.2f}, SD={b_sd_mean:9.3f} | "
            f"RL-ARPSO Ts={r_ts_mean:8.3f}, I={r_i_mean:7.2f}, SD={r_sd_mean:9.3f}"
        )

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{args.prefix}.csv")
    plot_path = os.path.join(out_dir, f"{args.prefix}.png")

    _write_csv(csv_path, summary_rows)

    x = np.asarray([int(r["swarm_size"]) for r in summary_rows], dtype=int)

    b_time = np.asarray([float(r["arpso_mean_time"]) for r in summary_rows], dtype=float)
    b_time_std = np.asarray([float(r["arpso_std_time"]) for r in summary_rows], dtype=float)
    r_time = np.asarray([float(r["rl_arpso_mean_time"]) for r in summary_rows], dtype=float)
    r_time_std = np.asarray([float(r["rl_arpso_std_time"]) for r in summary_rows], dtype=float)

    b_iter = np.asarray([float(r["arpso_mean_iters"]) for r in summary_rows], dtype=float)
    b_iter_std = np.asarray([float(r["arpso_std_iters"]) for r in summary_rows], dtype=float)
    r_iter = np.asarray([float(r["rl_arpso_mean_iters"]) for r in summary_rows], dtype=float)
    r_iter_std = np.asarray([float(r["rl_arpso_std_iters"]) for r in summary_rows], dtype=float)

    b_dist = np.asarray([float(r["arpso_mean_swarm_dist"]) for r in summary_rows], dtype=float)
    b_dist_std = np.asarray([float(r["arpso_std_swarm_dist"]) for r in summary_rows], dtype=float)
    r_dist = np.asarray([float(r["rl_arpso_mean_swarm_dist"]) for r in summary_rows], dtype=float)
    r_dist_std = np.asarray([float(r["rl_arpso_std_swarm_dist"]) for r in summary_rows], dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 14), sharex=True)

    _plot_metric(
        axes[0],
        x=x,
        baseline_mean=b_time,
        baseline_std=b_time_std,
        rl_mean=r_time,
        rl_std=r_time_std,
        title="Average Source Seeking Time vs Swarm Size",
        ylabel="Average source seeking time (s)",
    )
    _plot_metric(
        axes[1],
        x=x,
        baseline_mean=b_iter,
        baseline_std=b_iter_std,
        rl_mean=r_iter,
        rl_std=r_iter_std,
        title="Average Number of Iterations vs Swarm Size",
        ylabel="Average iterations",
    )
    _plot_metric(
        axes[2],
        x=x,
        baseline_mean=b_dist,
        baseline_std=b_dist_std,
        rl_mean=r_dist,
        rl_std=r_dist_std,
        title="Average Swarm Distance vs Swarm Size",
        ylabel="Average swarm distance",
    )

    axes[2].set_xlabel("Swarm size (n)")
    axes[0].legend(loc="best")

    fig.suptitle(
        (
            "ARPSO vs RL-ARPSO Across Swarm Sizes\n"
            f"runs per size = {cfg.episodes_per_n}, max_iter = {cfg.max_iter}, "
            f"arena = {cfg.side_length:.1f} x {cfg.side_length:.1f}, "
            f"obstacles = {'on' if cfg.use_obstacles else 'off'}"
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
