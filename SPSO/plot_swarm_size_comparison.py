from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from SPSO.rl_spso import RLSPSOConfig, compare_standard_vs_rl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep swarm sizes and plot SPSO vs RL-SPSO performance for "
            "average source seeking time, average iterations, and average swarm distance."
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained RL-SPSO model.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "ddpg", "td3"],
        help="RL algorithm used by the saved model.",
    )
    parser.add_argument(
        "--episodes-per-n",
        type=int,
        default=100,
        help="Number of Monte Carlo runs for each swarm size.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum iterations per run.")
    parser.add_argument(
        "--side-length",
        type=float,
        default=100.0,
        help="Side length of the square search arena.",
    )
    parser.add_argument(
        "--swarm-size-low",
        type=int,
        default=5,
        help="Lowest swarm size to include.",
    )
    parser.add_argument(
        "--swarm-size-high",
        type=int,
        default=31,
        help="Highest swarm size to include, inclusive.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Directory where plots and CSV summaries are saved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="spso_vs_rlspso_swarm_size",
        help="Prefix for output file names.",
    )
    return parser.parse_args()


def unpack_metric(results: Dict[str, float], swarm_sizes: List[int], prefix: str, metric: str) -> np.ndarray:
    values = []
    for swarm_size in swarm_sizes:
        key = f"N{swarm_size}_{prefix}_{metric}"
        values.append(float(results[key]))
    return np.asarray(values, dtype=float)


def build_rows(results: Dict[str, float], swarm_sizes: List[int]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for swarm_size in swarm_sizes:
        rows.append(
            {
                "swarm_size": int(swarm_size),
                "spso_mean_time": float(results[f"N{swarm_size}_std_mean_time"]),
                "spso_std_time": float(results[f"N{swarm_size}_std_std_time"]),
                "rl_spso_mean_time": float(results[f"N{swarm_size}_rl_mean_time"]),
                "rl_spso_std_time": float(results[f"N{swarm_size}_rl_std_time"]),
                "spso_mean_iters": float(results[f"N{swarm_size}_std_mean_iters"]),
                "spso_std_iters": float(results[f"N{swarm_size}_std_std_iters"]),
                "rl_spso_mean_iters": float(results[f"N{swarm_size}_rl_mean_iters"]),
                "rl_spso_std_iters": float(results[f"N{swarm_size}_rl_std_iters"]),
                "spso_mean_swarm_dist": float(results[f"N{swarm_size}_std_mean_swarm_dist"]),
                "spso_std_swarm_dist": float(results[f"N{swarm_size}_std_std_swarm_dist"]),
                "rl_spso_mean_swarm_dist": float(results[f"N{swarm_size}_rl_mean_swarm_dist"]),
                "rl_spso_std_swarm_dist": float(results[f"N{swarm_size}_rl_std_swarm_dist"]),
            }
        )
    return rows


def write_summary_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    axis: plt.Axes,
    swarm_sizes: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    rl_mean: np.ndarray,
    rl_std: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    axis.plot(swarm_sizes, baseline_mean, marker="o", linewidth=2.2, color="#1f77b4", label="SPSO")
    axis.fill_between(
        swarm_sizes,
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        color="#1f77b4",
        alpha=0.16,
    )

    axis.plot(swarm_sizes, rl_mean, marker="s", linewidth=2.2, color="#d62728", label="RL-SPSO")
    axis.fill_between(
        swarm_sizes,
        rl_mean - rl_std,
        rl_mean + rl_std,
        color="#d62728",
        alpha=0.16,
    )

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.28)
    axis.set_xlim(float(swarm_sizes[0]), float(swarm_sizes[-1]))
    axis.set_xticks(swarm_sizes)


def main() -> None:
    args = parse_args()

    if int(args.swarm_size_low) > int(args.swarm_size_high):
        raise SystemExit("--swarm-size-low must be <= --swarm-size-high")

    cfg = RLSPSOConfig(
        side_length=float(args.side_length),
        n_particles=int(args.swarm_size_low),
        max_iter=int(args.max_iter),
    )

    results = compare_standard_vs_rl(
        cfg=cfg,
        model_path=str(args.model_path),
        episodes=int(args.episodes_per_n),
        seed=int(args.seed),
        algo=str(args.algo),
        swarm_size_low=int(args.swarm_size_low),
        swarm_size_high=int(args.swarm_size_high) + 1,
    )

    swarm_sizes = list(range(int(args.swarm_size_low), int(args.swarm_size_high) + 1))
    swarm_sizes_arr = np.asarray(swarm_sizes, dtype=int)
    rows = build_rows(results, swarm_sizes)

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{args.prefix}.csv")
    plot_path = os.path.join(out_dir, f"{args.prefix}.png")

    write_summary_csv(csv_path, rows)

    spso_mean_time = unpack_metric(results, swarm_sizes, "std", "mean_time")
    spso_std_time = unpack_metric(results, swarm_sizes, "std", "std_time")
    rl_mean_time = unpack_metric(results, swarm_sizes, "rl", "mean_time")
    rl_std_time = unpack_metric(results, swarm_sizes, "rl", "std_time")

    spso_mean_iters = unpack_metric(results, swarm_sizes, "std", "mean_iters")
    spso_std_iters = unpack_metric(results, swarm_sizes, "std", "std_iters")
    rl_mean_iters = unpack_metric(results, swarm_sizes, "rl", "mean_iters")
    rl_std_iters = unpack_metric(results, swarm_sizes, "rl", "std_iters")

    spso_mean_dist = unpack_metric(results, swarm_sizes, "std", "mean_swarm_dist")
    spso_std_dist = unpack_metric(results, swarm_sizes, "std", "std_swarm_dist")
    rl_mean_dist = unpack_metric(results, swarm_sizes, "rl", "mean_swarm_dist")
    rl_std_dist = unpack_metric(results, swarm_sizes, "rl", "std_swarm_dist")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 1, figsize=(13, 14), sharex=True)

    plot_metric(
        axes[0],
        swarm_sizes_arr,
        spso_mean_time,
        spso_std_time,
        rl_mean_time,
        rl_std_time,
        title="Average Source Seeking Time vs Swarm Size",
        ylabel="Average source seeking time (s)",
    )
    plot_metric(
        axes[1],
        swarm_sizes_arr,
        spso_mean_iters,
        spso_std_iters,
        rl_mean_iters,
        rl_std_iters,
        title="Average Number of Iterations vs Swarm Size",
        ylabel="Average iterations",
    )
    plot_metric(
        axes[2],
        swarm_sizes_arr,
        spso_mean_dist,
        spso_std_dist,
        rl_mean_dist,
        rl_std_dist,
        title="Average Swarm Distance vs Swarm Size",
        ylabel="Average swarm distance",
    )

    axes[2].set_xlabel("Swarm size (n)")
    axes[0].legend(loc="best")

    figure.suptitle(
        (
            "SPSO vs RL-SPSO Across Swarm Sizes\n"
            f"runs per size = {int(args.episodes_per_n)}, max_iter = {int(args.max_iter)}, "
            f"arena = {float(args.side_length):.1f} x {float(args.side_length):.1f}"
        ),
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(plot_path, dpi=220)
    plt.close(figure)

    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()