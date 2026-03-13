from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from .arpso import ARPSO_SourceSeeker
except ImportError:  # pragma: no cover
    from arpso import ARPSO_SourceSeeker


class RLARPSOEnv:
    """Environment where PPO controls ARPSO params (c1,c2,c3,wi) every step."""

    def __init__(
        self,
        source_pos: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        obstacles: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
        num_particles: int = 20,
        max_iter: int = 300,
    ) -> None:
        self.source_pos = np.asarray(source_pos, dtype=float)
        self.bounds = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))
        self.obstacles = list(obstacles) if obstacles is not None else []
        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)

        self.arpso: Optional[ARPSO_SourceSeeker] = None
        self.current_iter = 0
        self.cumulative_time = 0.0
        self.prev_min_dist = np.inf

        # reward scaling
        self.success_bonus = 500.0
        self.timeout_penalty = -20.0
        self.invalid_action_penalty = -80.0
        self.collision_penalty = -12.0
        self.reward_clip = 200.0

        # Running statistics for reward normalization
        self.reward_rmean = 0.0
        self.reward_rvar = 1.0
        self.reward_count = 1e-4
        self.RN_BETA = 0.999

    def reset(
        self,
        source_pos: Optional[np.ndarray] = None,
        num_particles: Optional[int] = None,
        obstacles: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
    ) -> np.ndarray:
        if source_pos is not None:
            self.source_pos = np.asarray(source_pos, dtype=float)
        else:
            self.source_pos = np.random.uniform(low=self.bounds[0], high=self.bounds[1])

        if num_particles is not None:
            self.num_particles = int(num_particles)
        if obstacles is not None:
            self.obstacles = list(obstacles)

        c3_init = 1.0 if len(self.obstacles) > 0 else 0.0
        self.arpso = ARPSO_SourceSeeker(
            bounds=self.bounds,
            source_pos=self.source_pos,
            num_particles=self.num_particles,
            c1=1.5,
            c2=1.5,
            c3=c3_init,
            wi=0.7,
            T=1.0,
            obstacles=self.obstacles,
            obstacle_margin=4.0,
            termination_dist=0.1,
        )

        self.current_iter = 0
        self.cumulative_time = 0.0
        self.prev_min_dist = float(
            min(np.linalg.norm(p.x - self.source_pos) for p in self.arpso.particles)
        )
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        assert self.arpso is not None
        min_dist = float(min(np.linalg.norm(p.x - self.source_pos) for p in self.arpso.particles))
        avg_vel = float(np.mean([np.linalg.norm(p.v) for p in self.arpso.particles]))
        avg_omega = float(np.mean([p.last_omega for p in self.arpso.particles]))
        time_left = 1.0 - (self.current_iter / max(self.max_iter, 1))

        has_obstacles = 1.0 if self.obstacles else 0.0
        nearest_obs = 1.0
        if self.obstacles:
            dists = []
            for p in self.arpso.particles:
                for center, radius in self.obstacles:
                    d = np.linalg.norm(p.x - np.asarray(center)) - float(radius)
                    dists.append(d)
            nearest_obs = float(np.clip(np.min(dists) / 50.0, 0.0, 1.0))

        state = np.array(
            [
                np.clip(min_dist / 150.0, 0.0, 2.0),
                np.clip(self.prev_min_dist / 150.0, 0.0, 2.0),
                np.clip(avg_vel / 20.0, 0.0, 2.0),
                np.clip(self.cumulative_time / 60.0, 0.0, 2.0),
                np.clip(self.current_iter / max(self.max_iter, 1), 0.0, 1.0),
                np.clip(self.arpso.c1 / 4.0, 0.0, 2.0),
                np.clip(self.arpso.c2 / 4.0, 0.0, 2.0),
                np.clip(self.arpso.c3 / 4.0, 0.0, 2.0),
                np.clip(avg_omega / 1.2, 0.0, 1.5),
                time_left,
                has_obstacles,
                nearest_obs,
            ],
            dtype=np.float32,
        )
        return state

    def _map_action_to_params(self, action: np.ndarray) -> Tuple[float, float, float, float]:
        assert self.arpso is not None
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        # multiplicative updates from previous params (smooth changes)
        delta = 0.25
        c1 = self.arpso.c1 * (1.0 + delta * a[0])
        c2 = self.arpso.c2 * (1.0 + delta * a[1])
        c3 = self.arpso.c3 * (1.0 + delta * a[2]) if self.obstacles else 0.0
        wi = self.arpso.wi * (1.0 + 0.2 * a[3])

        c1 = float(np.clip(c1, 0.05, 3.5))
        c2 = float(np.clip(c2, 0.05, 3.5))
        c3 = float(np.clip(c3, 0.0, 3.5))
        wi = float(np.clip(wi, 0.05, 1.2))
        return c1, c2, c3, wi

    def _count_particles_inside_obstacles(self) -> int:
        assert self.arpso is not None
        if not self.obstacles:
            return 0
        count = 0
        for p in self.arpso.particles:
            for center, radius in self.obstacles:
                if np.linalg.norm(p.x - np.asarray(center, dtype=float)) <= float(radius):
                    count += 1
                    break
        return count

    def step(self, action: np.ndarray):
        assert self.arpso is not None

        c1, c2, c3, wi = self._map_action_to_params(action)
        invalid_pen = 0.0
        if not np.isfinite([c1, c2, c3, wi]).all():
            c1, c2, c3, wi = 1.5, 1.5, (1.0 if self.obstacles else 0.0), 0.7
            invalid_pen = self.invalid_action_penalty

        found, min_dist, step_time = self.arpso.step(c1=c1, c2=c2, c3=c3, wi=wi)
        collisions = self._count_particles_inside_obstacles()
        self.cumulative_time += step_time

        # Reward shaping mirrors rl_apso_dynamic_swarm_size:
        # constant step penalty + exponential time pressure + distance progress term.
        base_clock_penalty = -5.0
        time_pressure = -10.0 * np.exp(2.0 * (self.current_iter / max(self.max_iter, 1)) - 1.0)
        dist_delta = self.prev_min_dist - min_dist
        progress_reward = 75.0 * dist_delta
        success_bonus_time_sensitive = (
            1500.0 * np.cos((np.pi / 2.0) * (self.current_iter / max(self.max_iter, 1)))
            if found
            else 0.0
        )
        reward = (
            base_clock_penalty
            + time_pressure
            + progress_reward
            + success_bonus_time_sensitive
            + invalid_pen
        )

        done = False
        if found:
            reward += self.success_bonus
            done = True

        self.current_iter += 1
        if self.current_iter >= self.max_iter and not done:
            reward += self.timeout_penalty
            done = True

        self.prev_min_dist = float(min_dist)

        # Running reward normalization and clipping (same strategy as rl_apso_dynamic_swarm_size).
        old_mean = self.reward_rmean
        self.reward_rmean = self.RN_BETA * self.reward_rmean + (1.0 - self.RN_BETA) * reward
        self.reward_rvar = (
            self.RN_BETA * self.reward_rvar
            + (1.0 - self.RN_BETA) * (reward - old_mean) ** 2
        )
        r_std = np.sqrt(self.reward_rvar) + 1e-6
        reward = float(
            np.clip(
                (reward - self.reward_rmean) / r_std,
                -self.reward_clip,
                self.reward_clip,
            )
        )
        info = {
            "min_dist": float(min_dist),
            "cumulative_time": float(self.cumulative_time),
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "wi": wi,
            "collisions": collisions,
            "base_clock_penalty": float(base_clock_penalty),
            "time_pressure": float(time_pressure),
            "progress_reward": float(progress_reward),
            "success_bonus_time_sensitive": float(success_bonus_time_sensitive),
            "invalid_penalty": float(invalid_pen),
        }
        return self._get_state(), reward, done, info
