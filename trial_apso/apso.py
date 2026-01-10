# apso_with_boundary_and_speed.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from .utils import SPEED, T, TERMINATION_DIST, measure_signal


# ---- APSO implementation ----
class Drone:
    def __init__(self, id: int, minimize: bool = False):
        self.position = np.zeros(2, dtype=float)
        self.velocity = np.zeros(2, dtype=float)      # stored velocity (direction * speed)
        self.acceleration = np.zeros(2, dtype=float)
        self.best_position = np.zeros(2, dtype=float)
        self.best_signal = float('inf') if minimize else float('-inf')
        self.id = id

    def update_personal_best(self, signal: float):
        """Update personal best using current position and measured signal."""
        if signal < self.best_signal:
            # assume minimize flag handled externally; for source we maximize so callers will invert sign if needed
            self.best_signal = signal
            self.best_position = self.position.copy()

class APSO:
    def __init__(self,
                 n_drones: int = 5,
                 side_length: float = 100.0,
                 w1: float = 0.675,
                 w2: float = -0.285,
                 c1: float = 1.193,
                 c2: float = 1.193,
                 T_sample: float = T,
                 speed: float = SPEED,
                 objective: str = "source"):
        self.n_drones = int(n_drones)
        self.L = float(side_length)         # square side length (meters)
        self.search_space = (self.L, self.L)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.T = float(T_sample)
        self.speed = float(speed)

        self.objective = objective.lower()
        self.minimize = (self.objective == "rastrigin")  # only source used by default

        # source is at center
        self.source_position = np.array([self.L / 2.0, self.L / 2.0], dtype=float)

        # initialize drones (positions will be set to boundary points)
        self.drones = [Drone(i, minimize=self.minimize) for i in range(self.n_drones)]
        self._init_drones_on_boundary()

        # initialize personal bests using initial measurements
        for d in self.drones:
            s = measure_signal(d.position,self.source_position)
            # For source objective we want to maximize signal; to use np.argmin for selecting best we store negative signal
            if self.minimize:
                d.best_signal = s
            else:
                d.best_signal = -s   # store negative so min corresponds to max signal
            d.best_position = d.position.copy()

        # global best stored similarly (use negative signal for source so minimization picks max)
        if self.minimize:
            self.global_best_signal = float('inf')
        else:
            self.global_best_signal = float('inf')  # since we store negative signals, start at +inf
        self.global_best_position = None
        self._update_global_best_from_personal()

        # history
        self.score_history: list[float] = []
        self.min_distances: list[float] = []

    def _init_drones_on_boundary(self):
        """Place each drone at a random location on the boundary of the square [0,L]x[0,L]."""
        for d in self.drones:
            side = np.random.randint(4)
            u = np.random.uniform(0.0, self.L)
            if side == 0:    # bottom edge (x in [0,L], y=0)
                pos = np.array([u, 0.0])
            elif side == 1:  # right edge (x=L, y in [0,L])
                pos = np.array([self.L, u])
            elif side == 2:  # top edge (x in [0,L], y=L)
                pos = np.array([u, self.L])
            else:            # left edge (x=0, y in [0,L])
                pos = np.array([0.0, u])
            d.position = pos
            d.velocity = np.zeros(2, dtype=float)
            d.acceleration = np.zeros(2, dtype=float)
            d.best_position = pos.copy()

    

    def _update_global_best_from_personal(self):
        """Choose global best from personal bests. We stored personal bests as:
           - minimize case: actual signal (minimize)
           - source case: negative(signal) so smaller -> better (i.e., larger original signal)
        """
        best_values = [d.best_signal for d in self.drones]
        best_idx = int(np.argmin(best_values))
        self.global_best_signal = best_values[best_idx]
        self.global_best_position = self.drones[best_idx].best_position.copy()

    def _apply_constant_speed(self, v_vec: np.ndarray) -> np.ndarray:
        """Given a velocity vector from APSO update, set its magnitude to self.speed
           while keeping direction. If v_vec is zero, return zero vector.
        """
        norm = np.linalg.norm(v_vec)
        if norm <= 1e-12:
            return np.zeros_like(v_vec)
        return (v_vec / norm) * self.speed

    def step(self) -> bool:
        """Perform one APSO iteration.
           Returns True if termination criterion met (any drone within TERMINATION_DIST of source).
        """
        # 1) measure current positions and update personal bests
        for d in self.drones:
            s = measure_signal(d.position,self.source_position)
            if self.minimize:
                # minimize this value
                if s < d.best_signal:
                    d.best_signal = s
                    d.best_position = d.position.copy()
            else:
                # for source we want to maximize: store negative so min = best
                stored = -s
                if stored < d.best_signal:
                    d.best_signal = stored
                    d.best_position = d.position.copy()

        # 2) determine global best (using stored personal best signals)
        self._update_global_best_from_personal()
        # record actual global best signal (positive) for history
        if self.minimize:
            self.score_history.append(self.global_best_signal)
        else:
            # convert back to positive signal
            self.score_history.append(-self.global_best_signal)

        # 3) APSO third-order updates (vectorized per drone)
        for d in self.drones:
            # R(0,c) ~ Gaussian(0, c): std = sqrt(c)
            r1 = np.random.uniform(0.0, self.c1, size=2)
            r2 = np.random.uniform(0.0, self.c2, size=2)

            personal_best = (d.best_position - d.position)
            global_best = (self.global_best_position - d.position)

            # acceleration update
            a_new = self.w1 * d.acceleration + r1 * personal_best + r2 * global_best
            d.acceleration = a_new

            # velocity update according to eqn: v(k+1) = w2*v(k) + a(k+1)*T
            d.velocity = self.w2 * d.velocity + d.acceleration * self.T

            # position update
            d.position = d.position + d.velocity * self.T

            # clip to square
            d.position = np.clip(d.position, 0.0, self.L)

        # diagnostics: min distance to source for termination & history
        min_dist = min(np.linalg.norm(d.position - self.source_position) for d in self.drones)
        self.min_distances.append(min_dist)

        return min_dist <= TERMINATION_DIST

    def run(self, max_iterations: int = 1000) -> tuple[int, float]:
        """
        Run until termination or max_iterations.
        Returns (iterations_used, simulated_time_seconds).
        We treat simulated_time = iterations * T (not wall-clock).
        """
        for it in range(1, max_iterations + 1):
            found = self.step()
            if found:
                return it, it * self.T
        return max_iterations, max_iterations * self.T




