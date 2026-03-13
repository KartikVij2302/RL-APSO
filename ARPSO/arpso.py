from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def measure_signal(
    pos: np.ndarray,
    source_pos: np.ndarray,
    source_strength: float = 1.0,
    alpha: float = 0.01,
) -> float:
    """Deterministic source signal model."""
    d = np.linalg.norm(pos - source_pos)
    return float(source_strength * np.exp(-alpha * (d**2)))


@dataclass
class CircularObstacle:
    center: np.ndarray
    radius: float


class Particle:
    def __init__(self, dim: int, lo: np.ndarray, hi: np.ndarray) -> None:
        self.x = np.random.uniform(lo, hi, size=dim)
        self.v = np.zeros(dim, dtype=float)
        self.best_x = self.x.copy()
        self.best_signal = -np.inf
        self.dist_travelled = 0.0
        self.last_omega = 0.7


class ARPSO_SourceSeeker:
    """Adaptive Robotic PSO with obstacle-aware attractive point x_a.

    Update rule:
      v_i(k+1)=w_i v_i(k)+R(0,c1)(x_ib-x_i)+R(0,c2)(x_gb-x_i)+R(0,c3)(x_a-x_i)
      x_i(k+1)=x_i(k)+v_i(k+1)T
    """

    def __init__(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        source_pos: np.ndarray,
        num_particles: int = 20,
        c1: float = 1.5,
        c2: float = 1.5,
        c3: float = 0.0,
        wi: float = 0.7,
        T: float = 1.0,
        obstacles: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
        obstacle_margin: float = 4.0,
        source_strength: float = 1.0,
        alpha: float = 0.01,
        termination_dist: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.bounds = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))
        self.dim = int(self.bounds[0].shape[0])
        if self.dim != 2:
            raise ValueError("ARPSO_SourceSeeker currently supports 2D only.")

        self.source_pos = np.asarray(source_pos, dtype=float)
        self.num_particles = int(num_particles)
        self.T = float(T)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3)
        self.wi = float(wi)
        self.obstacle_margin = float(obstacle_margin)
        self.source_strength = float(source_strength)
        self.alpha = float(alpha)
        self.termination_dist = float(termination_dist)

        self.rng = np.random.default_rng(seed)
        self.obstacles: List[CircularObstacle] = []
        if obstacles:
            for center, radius in obstacles:
                self.obstacles.append(
                    CircularObstacle(center=np.asarray(center, dtype=float), radius=float(radius))
                )

        self.particles: List[Particle] = [
            Particle(self.dim, self.bounds[0], self.bounds[1]) for _ in range(self.num_particles)
        ]
        self._init_particles_on_boundary()

        for p in self.particles:
            s = measure_signal(
                p.x,
                self.source_pos,
                source_strength=self.source_strength,
                alpha=self.alpha,
            )
            p.best_signal = s
            p.best_x = p.x.copy()

        best_particle = max(self.particles, key=lambda p: p.best_signal)
        self.gbest_signal = float(best_particle.best_signal)
        self.gbest_x = best_particle.best_x.copy()
        self.iteration = 0

    def _init_particles_on_boundary(self) -> None:
        lo, hi = self.bounds
        for p in self.particles:
            side = int(self.rng.integers(0, 4))
            if side == 0:
                pos = np.array([self.rng.uniform(lo[0], hi[0]), lo[1]], dtype=float)
            elif side == 1:
                pos = np.array([hi[0], self.rng.uniform(lo[1], hi[1])], dtype=float)
            elif side == 2:
                pos = np.array([self.rng.uniform(lo[0], hi[0]), hi[1]], dtype=float)
            else:
                pos = np.array([lo[0], self.rng.uniform(lo[1], hi[1])], dtype=float)

            p.x = pos
            p.v.fill(0.0)
            p.best_x = p.x.copy()
            p.best_signal = -np.inf
            p.dist_travelled = 0.0

    def _in_obstacle(self, x: np.ndarray) -> bool:
        for obs in self.obstacles:
            if np.linalg.norm(x - obs.center) <= obs.radius:
                return True
        return False

    def _nearest_obstacle(self, x: np.ndarray) -> Optional[CircularObstacle]:
        if not self.obstacles:
            return None
        return min(self.obstacles, key=lambda o: np.linalg.norm(x - o.center) - o.radius)

    def _compute_attractive_position(self, p: Particle) -> np.ndarray:
        """x_a is defined away from obstacle; if none, default to gbest."""
        nearest = self._nearest_obstacle(p.x)
        if nearest is None:
            return self.gbest_x.copy()

        direction = p.x - nearest.center
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = self.rng.normal(size=self.dim)
            norm = np.linalg.norm(direction) + 1e-9
        unit = direction / norm

        radial_target = nearest.center + unit * (nearest.radius + self.obstacle_margin)
        toward_source = 0.35 * (self.source_pos - p.x)
        xa = radial_target + toward_source
        return np.clip(xa, self.bounds[0], self.bounds[1])

    def _project_out_of_obstacles(self, x: np.ndarray) -> np.ndarray:
        x_safe = x.copy()
        for obs in self.obstacles:
            vec = x_safe - obs.center
            d = np.linalg.norm(vec)
            min_d = obs.radius + 1e-3
            if d < min_d:
                if d < 1e-9:
                    vec = self.rng.normal(size=self.dim)
                    d = np.linalg.norm(vec) + 1e-9
                x_safe = obs.center + (vec / d) * min_d
        return np.clip(x_safe, self.bounds[0], self.bounds[1])

    def step(
        self,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
        c3: Optional[float] = None,
        wi: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """One ARPSO iteration.

        Returns:
          found, min_dist_to_source, step_time_mean
        """
        if c1 is not None:
            self.c1 = float(c1)
        if c2 is not None:
            self.c2 = float(c2)
        if c3 is not None:
            self.c3 = float(c3)
        if wi is not None:
            self.wi = float(wi)

        for p in self.particles:
            s = measure_signal(
                p.x,
                self.source_pos,
                source_strength=self.source_strength,
                alpha=self.alpha,
            )
            if s > p.best_signal:
                p.best_signal = s
                p.best_x = p.x.copy()

        best_particle = max(self.particles, key=lambda p: p.best_signal)
        self.gbest_signal = float(best_particle.best_signal)
        self.gbest_x = best_particle.best_x.copy()

        step_dist_sum = 0.0
        speed = 10.0

        for p in self.particles:
            r1 = self.rng.uniform(0.0, max(0.0, self.c1))
            r2 = self.rng.uniform(0.0, max(0.0, self.c2))
            r3 = self.rng.uniform(0.0, max(0.0, self.c3))

            # omega_i differs per particle and per iteration
            omega_i = np.clip(self.wi + self.rng.uniform(-0.08, 0.08), 0.05, 1.2)
            p.last_omega = float(omega_i)

            xa = self._compute_attractive_position(p)
            personal_term = p.best_x - p.x
            global_term = self.gbest_x - p.x
            obstacle_term = xa - p.x

            v_new = (
                omega_i * p.v
                + r1 * personal_term
                + r2 * global_term
                + r3 * obstacle_term
            )
            x_new = p.x + v_new * self.T
            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
            x_new = self._project_out_of_obstacles(x_new)

            step_dist = np.linalg.norm(x_new - p.x)
            step_dist_sum += float(step_dist)
            p.dist_travelled += float(step_dist)
            p.v = v_new
            p.x = x_new

        self.iteration += 1
        min_dist = float(min(np.linalg.norm(p.x - self.source_pos) for p in self.particles))
        found = min_dist <= self.termination_dist
        step_time_mean = (step_dist_sum / max(self.num_particles, 1)) / speed
        return found, min_dist, float(step_time_mean)

    def run_single(
        self,
        max_iter: int = 400,
        param_scheduler: Optional[Callable[[int], Tuple[float, float, float, float]]] = None,
    ) -> Tuple[float, int, float, bool]:
        """Run until source found or timeout."""
        for k in range(max_iter):
            if param_scheduler is None:
                found, _, _ = self.step()
            else:
                c1, c2, c3, wi = param_scheduler(k)
                found, _, _ = self.step(c1=c1, c2=c2, c3=c3, wi=wi)
            if found:
                finder = min(self.particles, key=lambda p: np.linalg.norm(p.x - self.source_pos))
                ts = finder.dist_travelled / 10.0
                sd = float(sum(p.dist_travelled for p in self.particles))
                return float(ts), k + 1, sd, True

        sd = float(sum(p.dist_travelled for p in self.particles))
        return sd / 10.0, max_iter, sd, False
