from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import sys

import numpy as np

try:
    from apso_rl_agent.PPO import PPOAgent
    from rl_arpso import RLARPSOEnv
except ImportError:  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from apso_rl_agent.PPO import PPOAgent
    from rl_arpso import RLARPSOEnv


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


def train_rl_arpso(
    num_episodes: int = 800,
    max_iter: int = 250,
    min_particles: int = 5,
    max_particles: int = 30,
    lr: float = 3e-4,
    save_path: str = "apso_rl_agent/models/latest_ppo_arpso.pth",
    seed: int = 42,
    use_obstacles: bool = True,
) -> List[float]:
    rng = np.random.default_rng(seed)

    lo = np.array([0.0, 0.0], dtype=float)
    hi = np.array([100.0, 100.0], dtype=float)
    bounds = (lo, hi)
    source = np.array([50.0, 50.0], dtype=float)

    min_particles = int(min_particles)
    max_particles = int(max_particles)
    if min_particles < 5 or max_particles > 30 or min_particles > max_particles:
        raise ValueError("Swarm size bounds must satisfy 5 <= min_particles <= max_particles <= 30")

    init_num_particles = int(rng.integers(min_particles, max_particles + 1))

    init_obstacles = (
        _sample_random_obstacles(bounds, rng, n_obstacles=3) if use_obstacles else []
    )
    env = RLARPSOEnv(
        source_pos=source,
        bounds=bounds,
        obstacles=init_obstacles,
        num_particles=init_num_particles,
        max_iter=max_iter,
    )

    state_dim = 12
    action_dim = 4  # c1,c2,c3,wi
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=lr)

    rewards_history: List[float] = []
    best_window_reward = -np.inf
    model_path = Path(save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, num_episodes + 1):
        # random source & random obstacles each episode for robustness
        ep_source = rng.uniform(low=lo + 8.0, high=hi - 8.0)
        ep_num_particles = int(rng.integers(min_particles, max_particles + 1))
        ep_obstacles = (
            _sample_random_obstacles(bounds, rng, n_obstacles=int(rng.integers(1, 5)))
            if use_obstacles
            else []
        )
        state = env.reset(
            source_pos=ep_source,
            num_particles=ep_num_particles,
            obstacles=ep_obstacles,
        )

        done = False
        ep_reward = 0.0
        while not done:
            action, logprob = agent.select_action(state)
            next_state, reward, done, _info = env.step(action)
            agent.store(state, action, logprob, reward, done)
            ep_reward += float(reward)
            state = next_state

        agent.update()
        rewards_history.append(ep_reward)

        if ep % 10 == 0:
            recent = float(np.mean(rewards_history[-10:]))
            print(
                f"[Episode {ep:4d}] reward_mean(10)={recent:8.3f} "
                f"last={ep_reward:8.3f} particles={ep_num_particles:2d}"
            )

        if ep >= 30:
            window_reward = float(np.mean(rewards_history[-30:]))
            if window_reward > best_window_reward:
                best_window_reward = window_reward
                agent.save(str(model_path))

    # always save final checkpoint
    final_path = model_path.with_name(model_path.stem + "_final" + model_path.suffix)
    agent.save(str(final_path))
    print(f"Saved best model to:  {model_path}")
    print(f"Saved final model to: {final_path}")
    return rewards_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO controller for ARPSO.")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--min-particles", type=int, default=5)
    parser.add_argument("--max-particles", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=str,
        default="apso_rl_agent/models/latest_ppo_arpso.pth",
    )
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="Disable obstacles (forces c3=0 by design).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _ = train_rl_arpso(
        num_episodes=args.episodes,
        max_iter=args.max_iter,
        min_particles=args.min_particles,
        max_particles=args.max_particles,
        lr=args.lr,
        save_path=args.save_path,
        seed=args.seed,
        use_obstacles=not args.no_obstacles,
    )
