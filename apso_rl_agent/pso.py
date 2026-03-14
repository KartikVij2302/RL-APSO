import numpy as np
from typing import Callable, Tuple, List, Optional


# -------------------------
# Signal model
# -------------------------
def measure_signal(
    pos: np.ndarray, source_pos: np.ndarray, S_s: float = 1.0, alpha: float = 0.01
) -> float:

    d = np.linalg.norm(pos - source_pos)
    return float(S_s * np.exp(-alpha * (d**2)))


# -------------------------
# Particle
# -------------------------
class Particle:

    def __init__(self, dim: int, lo: np.ndarray, hi: np.ndarray):

        self.x = np.random.uniform(lo, hi, size=dim)
        self.v = np.zeros(dim)

        self.best_x = self.x.copy()
        self.best_signal = -np.inf

        self.dist_travelled = 0.0


# -------------------------
# PSO Source Seeker
# -------------------------
class PSO_SourceSeeker:

    def __init__(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        source_pos: np.ndarray,
        num_particles: int = 10,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        S_s: float = 1.0,
        alpha: float = 0.01,
        termination_dist: float = 0.1,
        seed: Optional[int] = None,
    ):

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.bounds = (np.asarray(bounds[0]), np.asarray(bounds[1]))
        self.dim = self.bounds[0].shape[0]

        self.N = num_particles

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.S_s = S_s
        self.alpha = alpha

        self.source_pos = np.asarray(source_pos)

        self.termination_dist = termination_dist

        # swarm
        self.particles = [
            Particle(self.dim, self.bounds[0], self.bounds[1])
            for _ in range(self.N)
        ]

        self._init_particles_on_boundary()

        for p in self.particles:

            s = measure_signal(p.x, self.source_pos, S_s=self.S_s, alpha=self.alpha)

            p.best_signal = s
            p.best_x = p.x.copy()

        best_particle = max(self.particles, key=lambda p: p.best_signal)

        self.gbest_signal = best_particle.best_signal
        self.gbest_x = best_particle.best_x.copy()

        self.iteration = 0


    # -------------------------
    # Boundary initialization
    # -------------------------
    def _init_particles_on_boundary(self):

        lo, hi = self.bounds

        for p in self.particles:

            side = int(self.rng.integers(0, 4))

            if side == 0:
                x = self.rng.uniform(lo[0], hi[0])
                pos = np.array([x, lo[1]])

            elif side == 1:
                y = self.rng.uniform(lo[1], hi[1])
                pos = np.array([hi[0], y])

            elif side == 2:
                x = self.rng.uniform(lo[0], hi[0])
                pos = np.array([x, hi[1]])

            else:
                y = self.rng.uniform(lo[1], hi[1])
                pos = np.array([lo[0], y])

            p.x = pos
            p.v = np.zeros_like(p.v)

            p.dist_travelled = 0.0

            p.best_x = p.x.copy()
            p.best_signal = measure_signal(
                p.x, self.source_pos, S_s=self.S_s, alpha=self.alpha
            )


    # -------------------------
    # Single PSO step
    # -------------------------
    def step(self):

        # update personal best
        for p in self.particles:

            s = measure_signal(p.x, self.source_pos, self.S_s, self.alpha)

            if s > p.best_signal:

                p.best_signal = s
                p.best_x = p.x.copy()

        # update global best
        best_particle = max(self.particles, key=lambda p: p.best_signal)

        self.gbest_signal = best_particle.best_signal
        self.gbest_x = best_particle.best_x.copy()

        # velocity update
        for p in self.particles:

            r1 = self.rng.random()
            r2 = self.rng.random()

            cognitive = self.c1 * r1 * (p.best_x - p.x)
            social = self.c2 * r2 * (self.gbest_x - p.x)

            v_new = self.w * p.v + cognitive + social
            x_new = p.x + v_new

            p.dist_travelled += np.linalg.norm(x_new - p.x)

            p.v = v_new
            p.x = np.clip(x_new, self.bounds[0], self.bounds[1])

        self.iteration += 1

        min_dist = min(np.linalg.norm(p.x - self.source_pos) for p in self.particles)

        found = min_dist <= self.termination_dist

        return found, float(min_dist)


    # -------------------------
    # Run single trial
    # -------------------------
    def run_single(self, max_iter=1000, speed=10.0):

        traj_history = [np.vstack([p.x.copy() for p in self.particles])]

        for it in range(1, max_iter + 1):

            found, _ = self.step()

            traj_history.append(np.vstack([p.x.copy() for p in self.particles]))

            if found:

                finder = min(
                    self.particles, key=lambda p: np.linalg.norm(p.x - self.source_pos)
                )

                time_to_find = finder.dist_travelled / speed
                iterations_used = it
                swarm_distance = sum(p.dist_travelled for p in self.particles)

                return time_to_find, iterations_used, swarm_distance, traj_history

        swarm_distance = sum(p.dist_travelled for p in self.particles)

        return swarm_distance / speed, max_iter, swarm_distance, traj_history


    # -------------------------
    # Monte Carlo evaluation
    # -------------------------
    def run_monte_carlo(self, runs=30, max_iter=1000):

        Ts_list = []
        I_list = []
        SD_list = []

        histories = []

        for r in range(runs):

            Ts, I, SD, hist = self.run_single(max_iter)

            Ts_list.append(Ts)
            I_list.append(I)
            SD_list.append(SD)

            histories.append(hist)

        return {

            "mu_Ts": float(np.mean(Ts_list)),
            "mu_I": float(np.mean(I_list)),
            "mu_SD": float(np.mean(SD_list)),

            "Ts_list": Ts_list,
            "I_list": I_list,
            "SD_list": SD_list,

            "histories": histories,
        }
