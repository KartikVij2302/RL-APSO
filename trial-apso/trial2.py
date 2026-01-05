import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import time

class Drone:
    def __init__(self, search_space: Tuple[float, float], id: int, minimize: bool = False):
        """
        Initialize drone at a random position inside the search space box [0, W] x [0, H].
        'minimize' indicates whether lower objective values are better.
        """
        # Uniform initialization INSIDE the box (better for optimization benchmarks)
        self.position = np.array([
            np.random.uniform(0, search_space[0]),
            np.random.uniform(0, search_space[1])
        ])
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.best_position = self.position.copy()
        # Best signal/value initialization depends on optimization direction
        self.best_signal = float('inf') if minimize else float('-inf')
        self.id = id

    def update_best(self, signal_strength: float, minimize: bool = False):
        """
        Update personal best depending on whether we minimize or maximize.
        """
        if minimize:
            if signal_strength < self.best_signal:
                self.best_signal = signal_strength
                self.best_position = self.position.copy()
        else:
            if signal_strength > self.best_signal:
                self.best_signal = signal_strength
                self.best_position = self.position.copy()

class APSO:
    def __init__(self,
                 n_drones: int = 5,
                 search_space: Tuple[float, float] = (50, 50),
                 w1: float = 0.675,
                 w2: float = -0.285,
                 c1: float = 1.193,
                 c2: float = 1.193,
                 T: float = 1.0,
                 source_power: float = 100,
                 alpha: float = 0.01,
                 objective: str = "rastrigin",   # "rastrigin" or "source"
                 rastrigin_bounds: Tuple[float, float] = (-5.12, 5.12)):
        """
        objective:
            - "rastrigin" : evaluate Rastrigin after mapping real search space -> [-5.12,5.12]^2 (minimization)
            - "source"    : original source-detection objective (maximization)
        """
        self.n_drones = n_drones
        self.search_space = (float(search_space[0]), float(search_space[1]))
        self.w1 = w1
        self.w2 = w2
        self.c1 = c1
        self.c2 = c2
        self.T = T
        self.source_power = source_power
        self.alpha = alpha

        self.objective = objective.lower()
        self.minimize = (self.objective == "rastrigin")
        self.ras_bounds = (float(rastrigin_bounds[0]), float(rastrigin_bounds[1]))

        # Initialize source at center (used only if objective == "source")
        self.source_position = np.array([self.search_space[0] / 2.0, self.search_space[1] / 2.0])

        # Initialize drones (pass minimize flag to set their personal-best sentinel)
        self.drones = [Drone(self.search_space, i, minimize=self.minimize) for i in range(n_drones)]

        # Global best sentinel depends on minimize or maximize
        self.global_best_position = None
        self.global_best_signal = float('inf') if self.minimize else float('-inf')

        # For tracking performance
        self.score_history = []
        self.trajectory_history = []  # store positions of all drones each iteration
        # For source-objective compatibility
        self.min_distances = []

    # --- Rastrigin utilities ---
    def rastrigin(self, x: np.ndarray, A: float = 10.0) -> float:
        x = np.asarray(x)
        n = x.size
        return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

    def map_to_rastrigin_space(self, position: np.ndarray) -> np.ndarray:
        """
        Linearly map a point from the real search space [0, W] x [0, H] to
        the canonical Rastrigin box [ras_min, ras_max]^2.
        """
        pos = np.asarray(position, dtype=float)
        ras_min, ras_max = self.ras_bounds
        real_min = np.array([0.0, 0.0])
        real_max = np.array([self.search_space[0], self.search_space[1]])
        # Prevent division by zero
        scale = (ras_max - ras_min) / (real_max - real_min)
        return ras_min + (pos - real_min) * scale

    # --- Objective evaluation ---
    def measure_signal(self, position: np.ndarray) -> float:
        """
        Return objective value at 'position' in real coordinates.
        For Rastrigin mode -> returns Rastrigin(mapped position) (lower is better).
        For source mode -> returns source signal (higher is better).
        """
        if self.objective == "rastrigin":
            ras_pos = self.map_to_rastrigin_space(position)
            return self.rastrigin(ras_pos)
        else:
            # original Gaussian-like source signal model (maximization)
            distance = np.linalg.norm(position - self.source_position)
            return self.source_power * np.exp(-self.alpha * distance ** 2)

    # --- PSO update ---
    def update_drone(self, drone: Drone):
        # Stochastic coefficients in [0, c1] and [0, c2]
        r1 = np.random.uniform(0, self.c1)
        r2 = np.random.uniform(0, self.c2)

        # Use global best if available; otherwise fall back to personal best
        g_best = self.global_best_position if self.global_best_position is not None else drone.best_position

        drone.acceleration = (self.w1 * drone.acceleration +
                              r1 * (drone.best_position - drone.position) +
                              r2 * (g_best - drone.position))

        # Update velocity (integrate acceleration)
        drone.velocity = self.w2 * drone.velocity + drone.acceleration * self.T

        # Update position
        drone.position = drone.position + drone.velocity * self.T

        # Ensure drone stays within bounds [0, W] x [0, H]
        drone.position = np.clip(drone.position, [0.0, 0.0], [self.search_space[0], self.search_space[1]])

    def step(self) -> bool:
        current_positions = []
        # 1) Evaluate and update personal/global bests
        for drone in self.drones:
            score = self.measure_signal(drone.position)
            # Update personal best (works for both min and max)
            drone.update_best(score_strength := score, minimize=self.minimize)

            # Update global best depending on minimization/maximization
            if self.minimize:
                if score < self.global_best_signal:
                    self.global_best_signal = score
                    self.global_best_position = drone.position.copy()
            else:
                if score > self.global_best_signal:
                    self.global_best_signal = score
                    self.global_best_position = drone.position.copy()

            current_positions.append(drone.position.copy())

        # Record positions
        self.trajectory_history.append(np.array(current_positions))

        # 2) Move drones
        for drone in self.drones:
            self.update_drone(drone)

        # 3) Log performance (score history)
        self.score_history.append(self.global_best_signal)

        # if using source model, also track min distance and a stopping criterion
        if self.objective != "rastrigin":
            min_dist = min(np.linalg.norm(drone.position - self.source_position) for drone in self.drones)
            self.min_distances.append(min_dist)
            # Return True if source found (within tolerance)
            return min_dist < 0.1
        else:
            # For Rastrigin minimization, return True if global best close to zero
            return self.global_best_signal < 1e-6

    def run(self, max_iterations: int = 1000) -> Tuple[int, float]:
        start_time = time.time()
        for iteration in range(max_iterations):
            if self.step():
                elapsed_time = time.time() - start_time
                return iteration + 1, elapsed_time
        elapsed_time = time.time() - start_time
        return max_iterations, elapsed_time

    # --- Visualization ---
    def plot_performance(self):
        plt.figure(figsize=(10, 5))
        if self.minimize:
            # Plot best (min) value per iteration (log scale for readability)
            plt.plot(self.score_history, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Best Fitness (Rastrigin)')
            plt.yscale('log')
            plt.title('APSO Convergence (Rastrigin minimization)')
            plt.grid(True, which="both", ls="-", alpha=0.2)
        else:
            # Original source-search performance (min distance)
            plt.plot(self.min_distances, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Minimum Distance to Source')
            plt.title('APSO Search Performance (Source)')
            plt.grid(True)
        plt.show()

    def visualize_search(self, show_contour: bool = True):
        """
        Visualize the final drone positions and (optionally) the landscape.
        For Rastrigin objective, the contour plotted will be the Rastrigin landscape
        mapped onto the real search-space coordinates.
        """
        plt.figure(figsize=(8, 8))

        # Plot search space boundaries
        W, H = self.search_space
        plt.plot([0, W], [0, 0], 'k-')
        plt.plot([0, W], [H, H], 'k-')
        plt.plot([0, 0], [0, H], 'k-')
        plt.plot([W, W], [0, H], 'k-')

        if self.objective == "rastrigin" and show_contour:
            # Build grid in real coordinates and evaluate mapped Rastrigin
            nx = ny = 120
            xs = np.linspace(0.0, W, nx)
            ys = np.linspace(0.0, H, ny)
            X, Y = np.meshgrid(xs, ys)
            Z = np.zeros_like(X)
            pts = np.stack([X.ravel(), Y.ravel()], axis=1)
            # Map and evaluate in batches
            for idx, p in enumerate(pts):
                ras = self.map_to_rastrigin_space(p)
                Z.ravel()[idx] = self.rastrigin(ras)
            Z = Z.reshape(X.shape)
            contour = plt.contourf(X, Y, Z, levels=30, cmap='viridis')
            plt.colorbar(contour, label='Rastrigin Value (mapped)')

        # Plot source (if applicable)
        if self.objective != "rastrigin":
            plt.plot(self.source_position[0], self.source_position[1], 'r*', markersize=15, label='Source')

        # Plot drones (final positions)
        final_positions = self.trajectory_history[-1] if len(self.trajectory_history) > 0 else np.array([d.position for d in self.drones])
        plt.scatter(final_positions[:, 0], final_positions[:, 1], c='red', s=40, label='Drones', edgecolors='white')

        # Plot global best
        if self.global_best_position is not None:
            plt.scatter(self.global_best_position[0], self.global_best_position[1],
                        c='white', marker='*', s=200, label='Global Best', edgecolors='black')

        plt.xlabel('X Position (real space)')
        plt.ylabel('Y Position (real space)')
        if self.objective == "rastrigin":
            plt.title('Final Drone Positions on Mapped Rastrigin Landscape')
        else:
            plt.title('Final Drone Positions (Source Search)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

# --- Example usage in main() ---
def main():
    # Example: run Rastrigin in a 50x50 box (maps to [-5.12,5.12]^2 internally)
    apso = APSO(n_drones=30, search_space=(50, 50), objective="rastrigin")
    print("Running APSO (Rastrigin mapped from [0,50]^2) ...")
    iterations, elapsed_time = apso.run(max_iterations=300)
    print(f"Completed in {iterations} iterations, {elapsed_time:.4f} sec")
    print(f"Best score (Rastrigin) found: {apso.global_best_signal:.6f}")
    print(f"Best position (real-space): {apso.global_best_position}")
    # Visualize
    apso.plot_performance()
    apso.visualize_search(show_contour=True)

if __name__ == "__main__":
    main()
