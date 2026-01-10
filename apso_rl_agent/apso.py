import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple, List, Optional

# -------------------------
# Utility: validate params (eqns 14-17)
# -------------------------
def validate_apso_params(w1: float, w2: float, c1: float, c2: float, T: float) -> None:
    if not (2.0 / T) * (1 + w1 + w2 + w1 * w2) > (c1 + c2):
        raise ValueError("Stability condition (14) violated: (2/T)*(1+w1+w2+w1*w2) > c1+c2 must hold.")
    if abs(w1 * w2) >= 1.0:
        raise ValueError("Stability condition (15) violated: |w1*w2| < 1 required.")
    lhs = abs((1 - w1 * w2) * (w1 + w2) + (w1 * w2) * (c1 * T + c2 * T))
    rhs = abs(1 - (w1 * w2) ** 2)
    if lhs >= rhs:
        raise ValueError("Stability condition (16) violated.")
    if abs(w1 + w2) >= abs(1 + w1 * w2):
        raise ValueError("Stability condition (17) violated.")


# -------------------------
# Signal model (S_mi) - deterministic
# -------------------------
def measure_signal(pos: np.ndarray,
                   source_pos: np.ndarray,
                   S_s: float = 1.0,
                   alpha: float = 0.01) -> float:
    """
    Deterministic signal model (noise-free):

    S_mi(t) = S_s * exp(-alpha * d_i(t)^2)
    """
    d = np.linalg.norm(pos - source_pos)
    S_true = S_s * np.exp(-alpha * (d ** 2))
    return float(S_true)


# -------------------------
# Particle class (maximization objective for signal)
# -------------------------
class Particle:
    def __init__(self, dim: int, lo: np.ndarray, hi: np.ndarray):
        self.x = np.random.uniform(lo, hi, size=dim)
        self.v = np.zeros(dim, dtype=float)
        self.a = np.zeros(dim, dtype=float)
        # personal best is based on measured signal (maximize)
        self.best_x = self.x.copy()
        self.best_signal = -np.inf
        # cumulative distance travelled (for swarm-distance metric)
        self.dist_travelled = 0.0


# -------------------------
# APSO for source-seeking
# -------------------------
class APSO_SourceSeeker:
    def __init__(self,
                 objective: Callable[[np.ndarray], float],  # kept for interface compatibility
                 bounds: Tuple[np.ndarray, np.ndarray],
                 source_pos: np.ndarray,
                 num_particles: int = 10,
                 w1: float = 0.6,
                 w2: float = 0.4,
                 c1: float = 1.0,
                 c2: float = 1.0,
                 T: float = 1.0,
                 S_s: float = 1.0,
                 alpha: float = 0.01,
                 termination_dist: float = 0.1,   # <-- default now 0.1 m
                 seed: Optional[int] = None):
        """
        APSO source seeker (deterministic measurement).
        termination_dist default set to 0.1 m per your requirement.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        validate_apso_params(w1, w2, c1, c2, T)

        self.bounds = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))
        self.dim = self.bounds[0].shape[0]
        self.N = int(num_particles)

        self.w1 = float(w1)
        self.w2 = float(w2)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.T = float(T)

        self.S_s = float(S_s)
        self.alpha = float(alpha)
        self.termination_dist = float(termination_dist)

        self.source_pos = np.asarray(source_pos, dtype=float)

        # swarm
        self.particles: List[Particle] = [Particle(self.dim, self.bounds[0], self.bounds[1]) for _ in range(self.N)]
        self._init_particles_on_boundary()
        # initialize personal bests with initial measurements
        for p in self.particles:
            s = measure_signal(p.x, self.source_pos, S_s=self.S_s, alpha=self.alpha)
            p.best_signal = s
            p.best_x = p.x.copy()
        # global best (choose particle whose personal best signal is maximum)
        self.gbest_signal = max(p.best_signal for p in self.particles)
        self.gbest_x = max(self.particles, key=lambda p: p.best_signal).best_x.copy()

        # bookkeeping
        self.iteration = 0
    def _init_particles_on_boundary(self):
        """
        Place each particle at a random point on the boundary of the 2D box [lo,hi] x [lo,hi].
        Assumes 2D. For each particle choose one of 4 edges uniformly.
        """
        lo, hi = self.bounds
        # ensure 2D
        assert lo.shape[0] == 2 and hi.shape[0] == 2, "Boundary init assumes 2D bounds"

        for p in self.particles:
            side = int(self.rng.integers(0, 4))  # 0..3
            if side == 0:   # bottom edge (y = lo[1])
                x = self.rng.uniform(lo[0], hi[0])
                pos = np.array([x, lo[1]], dtype=float)
            elif side == 1: # right edge (x = hi[0])
                y = self.rng.uniform(lo[1], hi[1])
                pos = np.array([hi[0], y], dtype=float)
            elif side == 2: # top edge (y = hi[1])
                x = self.rng.uniform(lo[0], hi[0])
                pos = np.array([x, hi[1]], dtype=float)
            else:           # left edge (x = lo[0])
                y = self.rng.uniform(lo[1], hi[1])
                pos = np.array([lo[0], y], dtype=float)

            # assign and reset kinematic states
            p.x = pos
            p.v = np.zeros_like(p.v)
            p.a = np.zeros_like(p.a)
            p.dist_travelled = 0.0

            # recompute personal best at this initial position
            p.best_x = p.x.copy()
            p.best_signal = measure_signal(p.x, self.source_pos, S_s=self.S_s, alpha=self.alpha)


    def step(self) -> Tuple[bool, float]:
        """
        Perform one APSO iteration.
        Returns (found_source_flag, min_distance_to_source)
        """
        # 1) measure current positions and update personal bests based on measured signal (x_ib)
        for idx, p in enumerate(self.particles):
            s = measure_signal(p.x, self.source_pos, S_s=self.S_s, alpha=self.alpha)
            if s > p.best_signal:
                p.best_signal = s
                p.best_x = p.x.copy()

        # 2) update global best x_gb (argmax over personal best signals)
        best_particle = max(self.particles, key=lambda p: p.best_signal)
        self.gbest_signal = best_particle.best_signal
        self.gbest_x = best_particle.best_x.copy()

        # 3) APSO third-order updates (use R(0,c) ~ Uniform[0,c] as per paper)
        for p in self.particles:
            # scalar random samples R(0,c1) and R(0,c2)
            r1 = self.rng.uniform(0.0, self.c1)
            r2 = self.rng.uniform(0.0, self.c2)

            personal_term = (p.best_x - p.x)
            global_term = (self.gbest_x - p.x)

            a_new = self.w1 * p.a + r1 * personal_term + r2 * global_term
            v_new = self.w2 * p.v + a_new * self.T
            x_new = p.x + v_new * self.T

            # update distance travelled
            p.dist_travelled += np.linalg.norm(x_new - p.x)

            # assign new states
            p.a = a_new
            p.v = v_new
            # clip to bounds
            p.x = np.clip(x_new, self.bounds[0], self.bounds[1])

        self.iteration += 1

        # compute min distance to source (for termination check)
        min_dist = min(np.linalg.norm(p.x - self.source_pos) for p in self.particles)
        found = (min_dist <= self.termination_dist)
        return found, float(min_dist)

    def run_single(
        self,
        max_iter: int = 1000,
        speed: float = 10.0
    ) -> Tuple[float, int, float, List[np.ndarray]]:
        """
        Run APSO until the source is found or max_iter is reached.

        Returns:
        - time_to_find (seconds): distance travelled by the UAV that first reaches
            the source divided by constant speed.
        - iterations_used: number of iterations (waypoints) until detection.
        - swarm_distance_total: sum of distances travelled by ALL UAVs until detection
            (this is D_{ji} summed over j for this run).
        - trajectory history: list of NxD arrays (swarm positions per iteration).
        """

        # Store trajectories if needed for plotting / analysis
        traj_history: List[np.ndarray] = [
            np.vstack([p.x.copy() for p in self.particles])
        ]

        for it in range(1, max_iter + 1):
            found, _ = self.step()

            traj_history.append(
                np.vstack([p.x.copy() for p in self.particles])
            )

            if found:
                # --- Identify the UAV that actually reached the source ---
                finder = min(
                    self.particles,
                    key=lambda p: np.linalg.norm(p.x - self.source_pos)
                )

                # --- Time metric: physical time ---
                time_to_find = finder.dist_travelled / speed

                # --- Iteration metric ---
                iterations_used = it

                # --- Swarm distance metric (Eq. 24, per-run quantity SD_i) ---
                swarm_distance_total = sum(
                    p.dist_travelled for p in self.particles
                )

                return (
                    time_to_find,
                    iterations_used,
                    swarm_distance_total,
                    traj_history,
                )

        # -----------------------------
        # If source NOT found
        # -----------------------------
        swarm_distance_total = sum(
            p.dist_travelled for p in self.particles
        )

        # Upper-bound proxy for time (no finder UAV)
        time_to_find = swarm_distance_total / speed

        return (
            time_to_find,
            max_iter,
            swarm_distance_total,
            traj_history,
        )



    def run_monte_carlo(self, runs: int = 30, max_iter: int = 1000) -> dict:
        """
        Perform multiple independent Monte Carlo runs and compute metrics:
          - mu_Ts (average time to find),
          - mu_I (average iterations),
          - mu_SD (average swarm distance)
        Returns dictionary with metrics and raw arrays.
        """
        Ts_list = []
        I_list = []
        SD_list = []
        all_histories = []

        for r in range(runs):
            # Reinitialize swarm for each run
            self.particles = [Particle(self.dim, self.bounds[0], self.bounds[1]) for _ in range(self.N)]
            for p in self.particles:
                s = measure_signal(p.x, self.source_pos, S_s=self.S_s, alpha=self.alpha)
                p.best_signal = s
                p.best_x = p.x.copy()
            self.gbest_signal = max(p.best_signal for p in self.particles)
            self.gbest_x = max(self.particles, key=lambda p: p.best_signal).best_x.copy()
            # reset distances and iteration
            for p in self.particles:
                p.dist_travelled = 0.0
                p.v.fill(0.0)
                p.a.fill(0.0)
            self.iteration = 0

            Ts, I, SD, hist = self.run_single(max_iter=max_iter)
            Ts_list.append(Ts)
            I_list.append(I)
            SD_list.append(SD)
            all_histories.append(hist)

        mu_Ts = float(np.mean(Ts_list))
        mu_I = float(np.mean(I_list))
        mu_SD = float(np.mean(SD_list))

        return {
            "mu_Ts": mu_Ts,
            "mu_I": mu_I,
            "mu_SD": mu_SD,
            "Ts_list": Ts_list,
            "I_list": I_list,
            "SD_list": SD_list,
            "histories": all_histories,
        }


# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    # Domain and source
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    source = np.array([50.0, 50.0])

    # APSO hyperparameters (from paper)
    w1 = 0.675
    w2 = -0.285
    c1 = 1.193
    c2 = 1.193
    T = 1.0

    swarm_sizes = list(range(5, 31))  # n = 5 to 30
    runs = 10
    max_iter = 500

    avg_Ts = []
    avg_I = []
    avg_SD = []

    for n in swarm_sizes:
        print(f"\nRunning APSO for swarm size n = {n}")

        apso = APSO_SourceSeeker(
            objective=lambda x: 0.0,  # signal-based objective
            bounds=(lo, hi),
            source_pos=source,
            num_particles=n,
            w1=w1, w2=w2, c1=c1, c2=c2,
            T=T,
            S_s=1.0,
            alpha=0.01,
            termination_dist=0.1,
            seed=42
        )

        results = apso.run_monte_carlo(runs=runs, max_iter=max_iter)

        avg_Ts.append(results["mu_Ts"])
        avg_I.append(results["mu_I"])
        avg_SD.append(results["mu_SD"])

    # Convert to numpy arrays for safety
    avg_Ts = np.array(avg_Ts)
    avg_I = np.array(avg_I)
    avg_SD = np.array(avg_SD)

    plt.figure(figsize=(7, 5))
    plt.plot(swarm_sizes, avg_Ts, marker='o')
    plt.xlabel("Swarm size (n)")
    plt.ylabel("Average source seeking time μ(Tₛ) [s]")
    plt.title("APSO: Average Source Seeking Time vs Swarm Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_seeking_time_vs_swarm_size.png", dpi=300)
    plt.show()
    
    plt.figure(figsize=(7, 5))
    plt.plot(swarm_sizes, avg_I, marker='s', color='tab:orange')
    plt.xlabel("Swarm size (n)")
    plt.ylabel("Average number of iterations μ(I)")
    plt.title("APSO: Average Iterations vs Swarm Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_iterations_vs_swarm_size.png", dpi=300)
    plt.show()
    
    plt.figure(figsize=(7, 5))
    plt.plot(swarm_sizes, avg_SD, marker='^', color='tab:green')
    plt.xlabel("Swarm size (n)")
    plt.ylabel("Average swarm distance μ(SD)")
    plt.title("APSO: Average Swarm Distance vs Swarm Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_swarm_distance_vs_swarm_size.png", dpi=300)
    plt.show()

