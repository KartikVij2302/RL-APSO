import os
import sys
import random
import copy
import numpy as np
import torch
# Allow running this file directly (not just as a package module)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # When imported as part of the package
    from apso_rl_agent.apso import APSO_SourceSeeker
except Exception:
    # When run as a standalone script from within this folder
    from apso import APSO_SourceSeeker

try:
    from apso_rl_agent.PPO import PPOAgent
except Exception:
    from PPO import PPOAgent


class RLAPSOEnvErrorOnly:
    """
    RL environment where:
      - APSO hyperparameters (w1,w2,c1,c2) are FIXED
      - RL controls ONLY an additive error term in the position update
      - No normalization / smoothing for w1,w2,c1,c2
      - No smoothing of the error term
    """

    def __init__(self, source_pos, bounds, num_particles=20, max_iter=300):
        self.source_pos = np.array(source_pos, dtype=np.float32)
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter

        self.apso = None
        self.current_iter = 0
        self.prev_signal = 0.0
        self.prev_gbest_dist = 0.0

    # -------------------------------------------------
    # Reset
    # -------------------------------------------------
    def reset(self, num_particles: int | None = None):
        """Reset environment for a new episode.

        Parameters
        ----------
        num_particles : int, optional
            If provided, overrides swarm size for this episode.
        """

        if num_particles is not None:
            self.num_particles = int(num_particles)

        self.apso = APSO_SourceSeeker(
            objective=lambda x: 0.0,
            bounds=self.bounds,
            source_pos=self.source_pos,
            num_particles=self.num_particles,
            # FIXED APSO PARAMETERS
            w1=0.675,
            w2=-0.285,
            c1=1.193,
            c2=1.193,
            T=1.0,
            S_s=1.0,
            alpha=0.01,
            termination_dist=0.1
        )

        self.current_iter = 0
        self.prev_signal = getattr(self.apso, "gbest_signal", 0.0)
        self.prev_gbest_dist = np.linalg.norm(self.apso.gbest_x - self.source_pos)
        self.elapsed_time = 0.0
        return self._get_state()

    # -------------------------------------------------
    # State (NO normalization of w1,w2,c1,c2)
    # -------------------------------------------------
    def _get_state(self):
        # 1. Swarm diversity
        dists = [np.linalg.norm(p.x - self.apso.gbest_x) for p in self.apso.particles]
        diversity = np.mean(dists) if dists else 0.0

        # 2. Signal change
        current_signal = getattr(self.apso, "gbest_signal", 0.0)
        signal_change = current_signal - self.prev_signal

        # 3. Normalized time
        time_left = 1.0 - self.current_iter / self.max_iter

        # 4. Average velocity
        avg_vel = np.mean([np.linalg.norm(p.v) for p in self.apso.particles])

        # 5–8. RAW APSO parameters (no scaling)
        w1 = self.apso.w1
        w2 = self.apso.w2
        c1 = self.apso.c1
        c2 = self.apso.c2

        return np.array([
            diversity,
            signal_change,
            time_left,
            avg_vel,
            w1,
            w2,
            c1,
            c2
        ], dtype=np.float32)

    # -------------------------------------------------
    # Step
    # -------------------------------------------------
    def step(self, action):
        """
        action: np.array shape (2,)
                Direct positional error [ex, ey] in meters
        """

        # -------------------------------------------------
        # 1. Save previous positions
        # -------------------------------------------------
        prev_pos = np.array([p.x.copy() for p in self.apso.particles])

        # -------------------------------------------------
        # 2. Run APSO physics (fixed params)
        # -------------------------------------------------
        try:
            found, min_dist = self.apso.step()
        except Exception:
            found = False
            min_dist = 1e6

        # -------------------------------------------------
        # 3. Apply RL error term DIRECTLY (no smoothing)
        # -------------------------------------------------
        error = np.asarray(action, dtype=np.float32)

        for p in self.apso.particles:
            p.x = p.x + error

            # Keep inside bounds
            lo, hi = self.bounds
            p.x = np.minimum(np.maximum(p.x, lo), hi)

            if hasattr(p, "dist_travelled"):
                p.dist_travelled += np.linalg.norm(error)

        # -------------------------------------------------
        # 4. Update global best after error
        # -------------------------------------------------
        best_signal = -np.inf
        best_pos = None
        min_dist = np.inf

        Ss = getattr(self.apso, "S_s", 1.0)
        alpha = getattr(self.apso, "alpha", 0.01)

        for p in self.apso.particles:
            d = np.linalg.norm(p.x - self.source_pos)
            min_dist = min(min_dist, d)

            # Correct signal model from paper
            signal = Ss * np.exp(-alpha * d**2)

            if signal > best_signal:
                best_signal = signal
                best_pos = p.x.copy()

                self.apso.gbest_x = best_pos
                self.apso.gbest_signal = best_signal

        # -------------------------------------------------
        # 5. Reward
        # -------------------------------------------------
        curr_pos = np.array([p.x for p in self.apso.particles])
        per_particle_step_dist = np.linalg.norm(curr_pos - prev_pos, axis=1)
        step_dist_sum = float(np.sum(per_particle_step_dist))
        mean_step_dist = step_dist_sum / max(1, self.num_particles)

        reward = 0.0
        done = False

        # -------------------------------------------------
        # (1) OUT-OF-BOUNDS CHECK  → immediate termination
        # -------------------------------------------------
        lo, hi = self.bounds
        if np.any(curr_pos < lo) or np.any(curr_pos > hi):
            reward = -1000.0   # very large negative penalty
            done = True
        else:

            # -------------------------------------------------
            # (2) LOGARITHMIC PROXIMITY REWARD (PRIORITY)
            # -------------------------------------------------
            # min_dist should already be computed earlier
            eps = 1e-6
            PROX_SCALE = 20.0   # tune if needed
            proximity_reward = 0.1 + PROX_SCALE * np.log(1.0 / (min_dist + eps))
            reward += proximity_reward
            
            # -------------------------------------------------
            # (3) TIME PENALTY (moderate, not dominant)
            # -------------------------------------------------
            # Each step adds penalty
            reward -= 5.0

            # Convert mean travel to time using 10 m/s UAV speed
            UAV_SPEED = 10.0
            step_time = mean_step_dist / UAV_SPEED
            TIME_SCALE = 10.0
            reward -= TIME_SCALE * np.log1p(step_time)

            # -------------------------------------------------
            # (4) SUCCESS BONUS (dominant objective)
            # -------------------------------------------------
            termination_dist = getattr(self.apso, "termination_dist", 0.1)
            if min_dist <= termination_dist:
                reward += 1500.0   # very large positive reward
                done = True


        # -------------------------------------------------
        # 6. Bookkeeping
        # -------------------------------------------------
        self.prev_signal = best_signal
        self.prev_gbest_dist = min_dist
        self.current_iter += 1

        if self.current_iter >= self.max_iter:
            reward -= 20.0
            done = True

        return self._get_state(), float(reward), done, True


def set_global_seed(seed: int = 42) -> None:
    """Seed Python/NumPy/(PyTorch) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_rl_apso_error_term_training(
    num_episodes: int = 5000,
    max_iter: int = 500,
    num_particles: int = 20,
    swarm_size_range: tuple[int, int] = (5, 15),
    error_scale: float = 1.0,
    lr: float = 3e-4,
    seed: int = 42,
    save_filename: str = "latest_ppo_apso_error_term_random_particles.pth",
    early_stop: bool = False,
    early_stop_window: int = 100,
    early_stop_min_improve_pct: float = 0.05,
    early_stop_patience: int = 300,
) -> str:
    """Train PPO to output a 2D positional error term for APSO.

    Notes
    -----
    - PPO policy outputs actions in roughly [-1, 1]. We keep PPO's *raw* action
      for learning/log-prob calculations, but the environment receives
      `raw_action * error_scale` as the actual positional error (meters).
    """


    set_global_seed(seed)

    lo = np.array([0.0, 0.0], dtype=np.float32)
    hi = np.array([100.0, 100.0], dtype=np.float32)
    bounds = (lo, hi)

    # Fixed source location (per request)
    source = np.array([50.0, 50.0], dtype=np.float32)
    env = RLAPSOEnvErrorOnly(source, bounds, num_particles=num_particles, max_iter=max_iter)

    state_dim = 8
    action_dim = 2
    agent = PPOAgent(state_dim, action_dim, lr=lr)

    rewards_history = []

    # Early-stopping bookkeeping (tracks best moving-average reward).
    best_ma_reward = -np.inf
    best_state_dict = None
    episodes_since_improve = 0
    stop_reason = None

    save_path = os.path.join(current_dir, save_filename)
    print(
        "Starting RL-APSO (error-term only) training | "
        f"episodes={num_episodes}, max_iter={max_iter}, particles_range={swarm_size_range}, "
        f"error_scale={error_scale}"
    )
    if early_stop:
        print(
            "Early stopping enabled | "
            f"window={early_stop_window}, min_improve={early_stop_min_improve_pct*100:.1f}%, "
            f"patience={early_stop_patience}"
        )

    for ep in range(num_episodes):
        # Randomize swarm size (integer) each episode; keep source fixed.
        min_p, max_p = int(swarm_size_range[0]), int(swarm_size_range[1])
        if max_p < min_p:
            min_p, max_p = max_p, min_p
        ep_num_particles = random.randint(min_p, max_p)
        state = env.reset(num_particles=ep_num_particles)

        ep_reward = 0.0

        for t in range(max_iter):
            raw_action, logprob = agent.select_action(state)
            env_action = np.asarray(raw_action, dtype=np.float32) * float(error_scale)

            next_state, reward, done, _valid = env.step(env_action)
            agent.store(state, raw_action, logprob, reward, done)

            state = next_state
            ep_reward += reward
            if done:
                break

        agent.update()
        rewards_history.append(ep_reward)

        # -------------------------------
        # Early stopping (moving average)
        # -------------------------------
        if early_stop and len(rewards_history) >= int(early_stop_window):
            window = int(early_stop_window)
            ma_reward = float(np.mean(rewards_history[-window:]))

            if best_ma_reward == -np.inf:
                best_ma_reward = ma_reward
                best_state_dict = copy.deepcopy(agent.policy.state_dict())
                agent.save(save_path)
                episodes_since_improve = 0
            else:
                required = best_ma_reward * (1.0 + float(early_stop_min_improve_pct))
                if ma_reward >= required:
                    best_ma_reward = ma_reward
                    best_state_dict = copy.deepcopy(agent.policy.state_dict())
                    agent.save(save_path)
                    episodes_since_improve = 0
                else:
                    episodes_since_improve += 1
                    if episodes_since_improve >= int(early_stop_patience):
                        stop_reason = (
                            f"moving-average reward (last {window}) failed to improve by "
                            f"{early_stop_min_improve_pct*100:.1f}% for {early_stop_patience} episodes"
                        )
                        break

        if (ep + 1) % 10 == 0:
            avg_rew = float(np.mean(rewards_history[-10:]))
            msg = (
                f"Episode {ep+1}/{num_episodes} | Avg Reward (last 10): {avg_rew:.4f} | "
                f"Last swarm size: {ep_num_particles}"
            )
            if early_stop and len(rewards_history) >= int(early_stop_window):
                ma_reward = float(np.mean(rewards_history[-int(early_stop_window):]))
                msg += f" | MA{int(early_stop_window)}: {ma_reward:.4f} | Best MA: {best_ma_reward:.4f}"
                if episodes_since_improve > 0:
                    msg += f" | No-improve: {episodes_since_improve}/{early_stop_patience}"
            print(msg)

    # Save best (preferred) parameters.
    if best_state_dict is not None:
        agent.policy.load_state_dict(best_state_dict)
        agent.policy_old.load_state_dict(best_state_dict)
    agent.save(save_path)
    if stop_reason is not None:
        print(f"Early stopping triggered: {stop_reason}")
    print(f"Model saved to {save_path}")
    return save_path


if __name__ == "__main__":
    # Minimal CLI: optionally allow overriding episode count
    episodes = 10000
    if len(sys.argv) > 1:
        try:
            episodes = int(sys.argv[1])
        except Exception:
            pass
    run_rl_apso_error_term_training(num_episodes=episodes)
