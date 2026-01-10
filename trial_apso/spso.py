import numpy as np
from utils import measure_signal

class Particle:
    def __init__(self, id: int):
        self.id = id
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)

        self.best_position = np.zeros(2)
        self.best_signal = float('inf')

        self.dist_travelled = 0.0   # <-- ADD THIS



class SPSO:
    def __init__(self,
                 n_particles: int = 5,
                 side_length: float = 100.0,
                 omega: float = 0.721,
                 c1: float = 1.193,
                 c2: float = 1.193,
                 T: float = 1.0,
                 speed: float = 10.0):

        self.n = n_particles
        self.L = side_length
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.T = T
        self.speed = speed

        self.source = np.array([self.L / 2, self.L / 2])

        self.particles = [Particle(i) for i in range(self.n)]
        self._init_particles_on_boundary()

        self.global_best_position = None
        self.global_best_signal = float('inf')

        self.score_history = []
        self.min_distances = []

        self._initialize_bests()

    
    def _init_particles_on_boundary(self):
        for p in self.particles:
            side = np.random.randint(4)
            u = np.random.uniform(0, self.L)

            if side == 0:
                p.position = np.array([u, 0.0])
            elif side == 1:
                p.position = np.array([self.L, u])
            elif side == 2:
                p.position = np.array([u, self.L])
            else:
                p.position = np.array([0.0, u])

            p.velocity = np.zeros(2)
            p.best_position = p.position.copy()

    def _initialize_bests(self):
        for p in self.particles:
            s = measure_signal(p.position,self.source)
            p.best_signal = -s
            p.best_position = p.position.copy()

        self._update_global_best()


    def _update_global_best(self):
        best_idx = np.argmin([p.best_signal for p in self.particles])
        self.global_best_signal = self.particles[best_idx].best_signal
        self.global_best_position = self.particles[best_idx].best_position.copy()

    # -----------------------------
    # SPSO update
    # -----------------------------
    def step(self) -> bool:
        # Update personal bests
        for p in self.particles:
            s = measure_signal(p.position,self.source)
            stored = -s  # maximize signal
            if stored < p.best_signal:
                p.best_signal = stored
                p.best_position = p.position.copy()

        # Update global best
        self._update_global_best()
        self.score_history.append(-self.global_best_signal)

        # SPSO velocity + position update
        for p in self.particles:
            r1 = np.random.uniform(0.0, self.c1, size=2)
            r2 = np.random.uniform(0.0, self.c2, size=2)

            v_new = (
                self.omega * p.velocity
                + r1 * (p.best_position - p.position)
                + r2 * (self.global_best_position - p.position)
            )

            # Enforce constant speed
            norm_v = np.linalg.norm(v_new)
            if norm_v > 1e-12:
                p.velocity = (v_new / norm_v) * self.speed
            else:
                p.velocity = np.zeros(2)

            # Position update
            # Position update
            x_new = p.position + p.velocity * self.T

            # distance travelled (Euclidean)
            p.dist_travelled += np.linalg.norm(x_new - p.position)

            p.position = np.clip(x_new, 0.0, self.L)

        # Termination check
            distances = [np.linalg.norm(p.position - self.source) for p in self.particles]
            min_dist = min(distances)
            self.min_distances.append(min_dist)

            found = any(d <= 0.1 for d in distances)
            return found


    def run(self, max_iterations: int = 300):
        for k in range(1, max_iterations + 1):
            if self.step():
                # UAV that found the source
                finder = min(
                    self.particles,
                    key=lambda p: np.linalg.norm(p.position - self.source)
                )

                time_to_find = finder.dist_travelled / self.speed
                iterations_used = k
                swarm_distance = sum(p.dist_travelled for p in self.particles)

                return time_to_find, iterations_used, swarm_distance

        # If not found
        swarm_distance = sum(p.dist_travelled for p in self.particles)
        time_to_find = swarm_distance / self.speed

        return time_to_find, max_iterations, swarm_distance

import numpy as np
import matplotlib.pyplot as plt

def main():
    swarm_sizes = list(range(5, 31))
    runs = 10
    max_iter = 500

    avg_time = []
    avg_iters = []
    avg_swarm_dist = []

    for n in swarm_sizes:
        print(f"Running SPSO for swarm size n = {n}")

        times = []
        iters = []
        swarm_dists = []

        for _ in range(runs):
            spso = SPSO(
                n_particles=n,
                side_length=100.0,
                omega=0.721,
                c1=1.193,
                c2=1.193,
                T=1.0,
                speed=10.0
            )

            t, i, sd = spso.run(max_iterations=max_iter)

            times.append(t)
            iters.append(i)
            swarm_dists.append(sd)

        avg_time.append(np.mean(times))
        avg_iters.append(np.mean(iters))
        avg_swarm_dist.append(np.mean(swarm_dists))

    # -----------------------------
    # Plot 1: Avg source seeking time
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(swarm_sizes, avg_time, marker='o')
    plt.xlabel("Swarm size (n)")
    plt.ylabel("Average source seeking time μ(Tₛ) [s]")
    plt.title("SPSO: Average Source Seeking Time vs Swarm Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("spso_avg_time_vs_swarm_size.png", dpi=300)
    plt.show()

    # -----------------------------
    # Plot 2: Avg iterations
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(swarm_sizes, avg_iters, marker='s', color='tab:orange')
    plt.xlabel("Swarm size (n)")
    plt.ylabel("Average number of iterations μ(I)")
    plt.title("SPSO: Average Iterations vs Swarm Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("spso_avg_iterations_vs_swarm_size.png", dpi=300)
    plt.show()

    # -----------------------------
    # Plot 3: Avg swarm distance
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(swarm_sizes, avg_swarm_dist, marker='^', color='tab:green')
    plt.xlabel("Swarm size (n)")
    plt.ylabel("Average swarm distance μ(SD)")
    plt.title("SPSO: Average Swarm Distance vs Swarm Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("spso_avg_swarm_distance_vs_swarm_size.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

