import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

class Drone:
    def __init__(self, bounds: Tuple[float, float], id: int):
        # Initialize drone at random position within bounds
        self.position = np.random.uniform(bounds[0], bounds[1], 2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.best_position = self.position.copy()
        self.best_score = float('inf')
        self.id = id
        
    def update_best(self, score: float):
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()
            
class APSO:
    def __init__(self, 
                 n_drones: int = 5,
                 bounds: Tuple[float, float] = (-5.12, 5.12),
                 w1: float = 0.675,
                 w2: float = -0.285,
                 c1: float = 1.193,
                 c2: float = 1.193,
                 T: float = 1.0):
        
        self.n_drones = n_drones
        self.bounds = bounds
        self.w1 = w1
        self.w2 = w2
        self.c1 = c1
        self.c2 = c2
        self.T = T
        
        # Initialize drones
        self.drones = [Drone(bounds, i) for i in range(n_drones)]
        self.global_best_position = None
        self.global_best_score = float('inf')
        
        # For tracking performance
        self.score_history = []
        self.trajectory_history = [] # Store positions of all drones over time
        
    def evaluate_rastrigin(self, position: np.ndarray) -> float:
        """
        Rastrigin function: f(x) = An + sum(x_i^2 - A cos(2pi x_i))
        where A = 10, x_i in [-5.12, 5.12]
        Global minimum at x = 0, f(x) = 0
        """
        A = 10
        n = len(position)
        return A * n + np.sum(position**2 - A * np.cos(2 * np.pi * position))
    
    def update_drone(self, drone: Drone):
        # Update acceleration
        r1 = np.random.uniform(0, self.c1)
        r2 = np.random.uniform(0, self.c2)
        
        # If global best is not set yet (shouldn't happen after first step), use drone's best
        g_best = self.global_best_position if self.global_best_position is not None else drone.best_position
        
        drone.acceleration = (self.w1 * drone.acceleration + 
                            r1 * (drone.best_position - drone.position) +
                            r2 * (g_best - drone.position))
        
        # Update velocity
        drone.velocity = self.w2 * drone.velocity + drone.acceleration * self.T
        
        # Update position
        drone.position = drone.position + drone.velocity * self.T
        
        # Ensure drone stays within bounds
        drone.position = np.clip(drone.position, self.bounds[0], self.bounds[1])
    
    def step(self):
        current_positions = []
        # Update each drone's best position and global best
        for drone in self.drones:
            score = self.evaluate_rastrigin(drone.position)
            drone.update_best(score)
            
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = drone.position.copy()
            
            current_positions.append(drone.position.copy())
        
        self.trajectory_history.append(np.array(current_positions))
        
        # Update drone positions
        for drone in self.drones:
            self.update_drone(drone)
            
        self.score_history.append(self.global_best_score)
        
    def run(self, max_iterations: int = 100) -> Tuple[int, float]:
        start_time = time.time()
        
        for iteration in range(max_iterations):
            self.step()
            # Optional: Early stopping
            if self.global_best_score < 1e-6:
                elapsed_time = time.time() - start_time
                return iteration + 1, elapsed_time
                
        elapsed_time = time.time() - start_time
        return max_iterations, elapsed_time
    
    def plot_performance(self):
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Convergence
        plt.subplot(1, 2, 1)
        plt.plot(self.score_history, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Score (Log Scale)')
        plt.yscale('log')
        plt.title('APSO Convergence on Rastrigin Function')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Plot 2: Final State on Contour
        plt.subplot(1, 2, 2)
        x = np.linspace(self.bounds[0], self.bounds[1], 100)
        y = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = self.evaluate_rastrigin(np.array([X[i,j], Y[i,j]]))
                
        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Fitness Value')
        
        # Plot final positions
        final_positions = self.trajectory_history[-1]
        plt.scatter(final_positions[:, 0], final_positions[:, 1], c='red', s=30, label='Drones', edgecolors='white')
        
        # Plot global best
        if self.global_best_position is not None:
            plt.scatter(self.global_best_position[0], self.global_best_position[1], c='white', marker='*', s=200, label='Global Best', edgecolors='black')
            
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Final Drone Positions on Rastrigin Landscape')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_3d_landscape(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(self.bounds[0], self.bounds[1], 50)
        y = np.linspace(self.bounds[0], self.bounds[1], 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = self.evaluate_rastrigin(np.array([X[i,j], Y[i,j]]))
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Fitness')
        
        # Plot final positions in 3D
        final_positions = self.trajectory_history[-1]
        z_points = np.array([self.evaluate_rastrigin(p) for p in final_positions])
        ax.scatter(final_positions[:, 0], final_positions[:, 1], z_points, c='red', s=50, label='Drones')
        
        if self.global_best_position is not None:
            z_best = self.evaluate_rastrigin(self.global_best_position)
            ax.scatter(self.global_best_position[0], self.global_best_position[1], z_best, c='white', marker='*', s=200, label='Global Best', edgecolors='black')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Fitness')
        ax.set_title('3D View of Rastrigin Landscape and Final State')
        ax.legend()
        plt.show()

def run_monte_carlo(n_runs: int = 50):
    seeking_times = []
    iteration_counts = []
    
    print(f"\nStarting Monte Carlo simulation with {n_runs} runs...")
    
    for i in range(n_runs):
        # Re-initialize APSO for each run
        apso = APSO(n_drones=30, bounds=(-5.12, 5.12))
        iterations, elapsed_time = apso.run(max_iterations=200)
        
        seeking_times.append(elapsed_time)
        iteration_counts.append(iterations)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_runs} runs")
            
    avg_seeking_time = np.mean(seeking_times)
    avg_iterations = np.mean(iteration_counts)
    
    print(f"\nResults over {n_runs} runs:")
    print(f"Average Source Seeking Time (mu(Ts)): {avg_seeking_time:.4f} s")
    print(f"Average Number of Iterations (mu(I)): {avg_iterations:.2f}")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Source Seeking Time Distribution
    plt.subplot(1, 2, 1)
    plt.hist(seeking_times, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(avg_seeking_time, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_seeking_time:.4f}s')
    plt.xlabel('Source Seeking Time (s)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Source Seeking Time\n(N={n_runs})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Number of Iterations Distribution
    plt.subplot(1, 2, 2)
    plt.hist(iteration_counts, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.axvline(avg_iterations, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_iterations:.2f}')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Iterations\n(N={n_runs})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # Create APSO instance
    print("Initializing APSO for Rastrigin Function Optimization...")
    apso = APSO(n_drones=30, bounds=(-5.12, 5.12))
    
    # Run search
    print("Running optimization...")
    iterations, elapsed_time = apso.run(max_iterations=200)
    
    print(f"Search completed in {iterations} iterations")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Best Score Found: {apso.global_best_score:.6f}")
    print(f"Best Position: {apso.global_best_position}")
    
    # Plot results
    print("Generating plots...")
    apso.plot_performance()
    apso.plot_3d_landscape()
    
    # Run Monte Carlo Simulation
    run_monte_carlo(n_runs=50)

if __name__ == "__main__":
    main()
