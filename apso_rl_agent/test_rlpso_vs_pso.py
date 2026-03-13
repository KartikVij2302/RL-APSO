import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from apso_rl_agent.RLPSO import RLPSOEnv
    from apso_rl_agent.pso import PSO_SourceSeeker
    from apso_rl_agent.PPO import PPOAgent
except ImportError:
    from RLPSO import RLPSOEnv
    from pso import PSO_SourceSeeker
    from PPO import PPOAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RL-guided PSO (RLPSOEnv) against standard PSO_SourceSeeker."
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of Monte Carlo runs.")
    parser.add_argument("--max-iter", type=int, default=400, help="Max iterations per run.")
    parser.add_argument(
        "--num-particles",
        type=int,
        default=20,
        help="Swarm size used for both methods.",
    )
    parser.add_argument(
        "--arena-size",
        type=float,
        default=100.0,
        help="Square arena side length.",
    )
    parser.add_argument(
        "--termination-dist",
        type=float,
        default=0.1,
        help="Distance threshold to count source as found.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=10.0,
        help="Particle speed used for Ts metric (distance / speed).",
    )
    parser.add_argument(
        "--source-mode",
        choices=["fixed", "random"],
        default="fixed",
        help="Use one fixed source for all runs or random source per run.",
    )
    parser.add_argument("--source-x", type=float, default=None, help="Fixed source X.")
    parser.add_argument("--source-y", type=float, default=None, help="Fixed source Y.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to PPO model for RLPSO evaluation or warm-start training.",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=0,
        help="If >0, train PPO in RLPSOEnv before evaluation.",
    )
    parser.add_argument(
        "--train-max-iter",
        type=int,
        default=None,
        help="Max iterations per training episode (defaults to --max-iter).",
    )
    parser.add_argument(
        "--train-source-mode",
        choices=["fixed", "random"],
        default="random",
        help="Source sampling mode used during training.",
    )
    parser.add_argument(
        "--train-lr",
        type=float,
        default=3e-4,
        help="Learning rate for PPO training.",
    )
    parser.add_argument(
        "--train-save-path",
        type=str,
        default="apso_rl_agent/models/latest_ppo_rlpso_trained.pth",
        help="Where to save model trained by this script.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Training progress print frequency in episodes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global seed.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV path to save per-run metrics.",
    )
    return parser.parse_args()


def build_sources(
    runs: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    mode: str,
    fixed_source: np.ndarray,
    seed: int,
) -> np.ndarray:
    if mode == "fixed":
        return np.repeat(fixed_source[None, :], runs, axis=0)

    rng = np.random.default_rng(seed)
    lo, hi = bounds
    return rng.uniform(low=lo, high=hi, size=(runs, lo.shape[0]))


def compute_run_metrics(
    particles: List,
    source: np.ndarray,
    success: bool,
    speed: float,
) -> Tuple[float, float]:
    swarm_distance = float(sum(getattr(p, "dist_travelled", 0.0) for p in particles))

    if success:
        finder = min(particles, key=lambda p: np.linalg.norm(p.x - source))
        time_to_find = float(getattr(finder, "dist_travelled", 0.0)) / speed
    else:
        time_to_find = swarm_distance / speed

    return time_to_find, swarm_distance


def load_agent(model_path: str):
    if model_path is None:
        return None

    if torch is None:
        raise RuntimeError("PyTorch is required to load an RL model, but torch is unavailable.")

    agent = PPOAgent(state_dim=7, action_dim=3, lr=3e-4)
    agent.load(model_path)
    return agent


def train_rlpso_agent(
    episodes: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    num_particles: int,
    max_iter: int,
    termination_dist: float,
    seed: int,
    train_source_mode: str,
    fixed_source: np.ndarray,
    lr: float,
    log_every: int,
    init_model_path: str,
):
    if torch is None:
        raise RuntimeError("PyTorch is required for training, but torch is unavailable.")

    agent = PPOAgent(state_dim=7, action_dim=3, lr=lr)

    if init_model_path:
        agent.load(init_model_path)
        print(f"[Info] Warm-started training from: {init_model_path}")

    ep_rewards: List[float] = []
    ep_success: List[float] = []

    lo, hi = bounds
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        np.random.seed(seed + ep)

        if train_source_mode == "fixed":
            source = fixed_source.copy()
        else:
            source = rng.uniform(low=lo, high=hi)

        env = RLPSOEnv(
            source_pos=source,
            bounds=bounds,
            num_particles=num_particles,
            max_iter=max_iter,
        )
        state = env.reset(source_pos=source, num_particles=num_particles)
        env.pso.termination_dist = termination_dist

        done = False
        reward_sum = 0.0
        steps = 0

        while (not done) and (steps < max_iter):
            action, logprob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, logprob, reward, done)
            reward_sum += float(reward)
            state = next_state
            steps += 1

        agent.update()

        min_dist = float(min(np.linalg.norm(p.x - source) for p in env.pso.particles))
        success = float(min_dist <= termination_dist)

        ep_rewards.append(reward_sum)
        ep_success.append(success)

        if (ep + 1) % max(1, log_every) == 0 or ep == episodes - 1:
            window = min(log_every, len(ep_rewards))
            mean_reward = float(np.mean(ep_rewards[-window:]))
            mean_success = float(np.mean(ep_success[-window:]))
            print(
                f"[Train] ep={ep + 1}/{episodes} "
                f"reward_mean({window})={mean_reward:.3f} "
                f"success_mean({window})={100.0 * mean_success:.1f}%"
            )

    return agent


def evaluate_rlpso(
    agent,
    sources: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    num_particles: int,
    max_iter: int,
    termination_dist: float,
    speed: float,
    seed: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for run_idx, source in enumerate(sources):
        np.random.seed(seed + run_idx)

        env = RLPSOEnv(
            source_pos=source,
            bounds=bounds,
            num_particles=num_particles,
            max_iter=max_iter,
        )
        state = env.reset(source_pos=source, num_particles=num_particles)
        env.pso.termination_dist = termination_dist

        done = False
        steps = 0
        while (not done) and (steps < max_iter):
            if agent is None:
                action = np.zeros(3, dtype=np.float32)
            else:
                action, _ = agent.select_action(state)

            state, _, done, _ = env.step(action)
            steps += 1

        min_dist = float(min(np.linalg.norm(p.x - source) for p in env.pso.particles))
        success = bool(min_dist <= termination_dist)

        ts, sd = compute_run_metrics(env.pso.particles, source, success, speed)

        rows.append(
            {
                "run": run_idx,
                "method": "RLPSO",
                "Ts": ts,
                "I": float(env.current_iter),
                "SD": sd,
                "Success": float(success),
                "min_dist": min_dist,
            }
        )

    return rows


def evaluate_standard_pso(
    sources: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    num_particles: int,
    max_iter: int,
    termination_dist: float,
    speed: float,
    seed: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for run_idx, source in enumerate(sources):
        np.random.seed(seed + run_idx)

        pso = PSO_SourceSeeker(
            bounds=bounds,
            source_pos=source,
            num_particles=num_particles,
            w=0.7,
            c1=1.5,
            c2=1.5,
            termination_dist=termination_dist,
            seed=seed + run_idx,
        )

        found = False
        iterations = 0
        for _ in range(max_iter):
            found, _ = pso.step()
            iterations += 1
            if found:
                break

        min_dist = float(min(np.linalg.norm(p.x - source) for p in pso.particles))
        success = bool(found and min_dist <= termination_dist)

        ts, sd = compute_run_metrics(pso.particles, source, success, speed)

        rows.append(
            {
                "run": run_idx,
                "method": "PSO",
                "Ts": ts,
                "I": float(iterations),
                "SD": sd,
                "Success": float(success),
                "min_dist": min_dist,
            }
        )

    return rows


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    ts = np.array([r["Ts"] for r in rows], dtype=float)
    iters = np.array([r["I"] for r in rows], dtype=float)
    sd = np.array([r["SD"] for r in rows], dtype=float)
    success = np.array([r["Success"] for r in rows], dtype=float)

    return {
        "runs": len(rows),
        "mu_Ts": float(np.mean(ts)),
        "std_Ts": float(np.std(ts)),
        "mu_I": float(np.mean(iters)),
        "std_I": float(np.std(iters)),
        "mu_SD": float(np.mean(sd)),
        "std_SD": float(np.std(sd)),
        "success_rate": float(np.mean(success)),
    }


def print_summary(title: str, stats: Dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Runs         : {stats['runs']}")
    print(f"Ts (mean+-sd): {stats['mu_Ts']:.4f} +- {stats['std_Ts']:.4f}")
    print(f"I  (mean+-sd): {stats['mu_I']:.2f} +- {stats['std_I']:.2f}")
    print(f"SD (mean+-sd): {stats['mu_SD']:.4f} +- {stats['std_SD']:.4f}")
    print(f"Success rate : {100.0 * stats['success_rate']:.2f}%")


def maybe_write_csv(rows: List[Dict[str, float]], path: str) -> None:
    if path is None:
        return

    import csv

    fieldnames = ["run", "method", "Ts", "I", "SD", "Success", "min_dist"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-run metrics to: {path}")


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)

    lo = np.array([0.0, 0.0], dtype=float)
    hi = np.array([args.arena_size, args.arena_size], dtype=float)
    bounds = (lo, hi)

    if args.source_x is None or args.source_y is None:
        fixed_source = np.array([0.5 * args.arena_size, 0.5 * args.arena_size], dtype=float)
    else:
        fixed_source = np.array([args.source_x, args.source_y], dtype=float)

    sources = build_sources(args.runs, bounds, args.source_mode, fixed_source, args.seed)

    train_max_iter = args.max_iter if args.train_max_iter is None else args.train_max_iter

    if args.train_episodes > 0:
        print(
            "[Info] Training PPO in RLPSOEnv "
            f"for {args.train_episodes} episodes (max_iter={train_max_iter})."
        )
        agent = train_rlpso_agent(
            episodes=args.train_episodes,
            bounds=bounds,
            num_particles=args.num_particles,
            max_iter=train_max_iter,
            termination_dist=args.termination_dist,
            seed=args.seed,
            train_source_mode=args.train_source_mode,
            fixed_source=fixed_source,
            lr=args.train_lr,
            log_every=args.log_every,
            init_model_path=args.model_path,
        )

        save_dir = os.path.dirname(args.train_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        agent.save(args.train_save_path)
        print(f"[Info] Saved trained model to: {args.train_save_path}")
    elif args.model_path is not None:
        agent = load_agent(args.model_path)
        print(f"[Info] Loaded RL policy from: {args.model_path}")
    else:
        agent = None
        print("[Info] No --model-path provided. RLPSO will use zero-action policy.")

    rl_rows = evaluate_rlpso(
        agent=agent,
        sources=sources,
        bounds=bounds,
        num_particles=args.num_particles,
        max_iter=args.max_iter,
        termination_dist=args.termination_dist,
        speed=args.speed,
        seed=args.seed,
    )

    pso_rows = evaluate_standard_pso(
        sources=sources,
        bounds=bounds,
        num_particles=args.num_particles,
        max_iter=args.max_iter,
        termination_dist=args.termination_dist,
        speed=args.speed,
        seed=args.seed,
    )

    rl_stats = summarize(rl_rows)
    pso_stats = summarize(pso_rows)

    print_summary("RLPSO", rl_stats)
    print_summary("Standard PSO", pso_stats)

    print("\nDelta (RLPSO - PSO)")
    print("---------------------")
    print(f"dTs        : {rl_stats['mu_Ts'] - pso_stats['mu_Ts']:+.4f}")
    print(f"dI         : {rl_stats['mu_I'] - pso_stats['mu_I']:+.2f}")
    print(f"dSD        : {rl_stats['mu_SD'] - pso_stats['mu_SD']:+.4f}")
    print(
        f"dSuccess   : {100.0 * (rl_stats['success_rate'] - pso_stats['success_rate']):+.2f}%"
    )

    maybe_write_csv(rl_rows + pso_rows, args.output_csv)


if __name__ == "__main__":
    main()
