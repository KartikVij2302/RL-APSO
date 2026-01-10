import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from trial_apso.run_simulation import monte_carlo_experiments_apso
from trial_apso.utils import TERMINATION_DIST, T
from novelty_4 import StandardPSO, NovelPPO_PSO, ObstacleEnvironment

# Define Constants
GOAL_POS = np.array([90.0, 90.0])
SP_TERMINATION_DIST = TERMINATION_DIST 
MAX_ITERS = 300

# Helper to run SPSO Monte Carlo
def run_spso_mc(n_drones, n_runs=50):
    env = ObstacleEnvironment()
    seeking_times = []
    iterations = []
    
    for _ in range(n_runs):
        pso = StandardPSO(env, num_particles=n_drones, max_iter=MAX_ITERS)
        _, traj = pso.run()
        
        # Analyze trajectory for termination
        found = False
        for k, particle_positions in enumerate(traj):
            # Check if any particle is within distance
            dists = np.linalg.norm(particle_positions - GOAL_POS, axis=1)
            if np.min(dists) <= SP_TERMINATION_DIST:
                iterations.append(k)
                seeking_times.append(k * T) # Assume 1 Iter = T (1.0s)
                found = True
                break
        
        if not found:
            iterations.append(MAX_ITERS)
            seeking_times.append(MAX_ITERS * T)
            
    return np.mean(seeking_times), np.mean(iterations)

# Helper for PPO-PSO
def run_ppopso_mc(n_drones, n_runs=50):
    env = ObstacleEnvironment()
    agent = NovelPPO_PSO(env, num_particles=n_drones, max_iter=MAX_ITERS)
    
    # Pre-train for 50 episodes
    # Use quiet mode or just let it print
    agent.pretrain(episodes=50) 
    
    seeking_times = []
    iterations = []
    
    for _ in range(n_runs):
        agent.reset_swarm()
        # agent.run() returns (val, traj). 
        # Note: In novelty_4.py, run() continues to update the agent (Online Learning).
        _, traj = agent.run()
        
        found = False
        for k, particle_positions in enumerate(traj):
            dists = np.linalg.norm(particle_positions - GOAL_POS, axis=1)
            if np.min(dists) <= SP_TERMINATION_DIST:
                iterations.append(k)
                seeking_times.append(k * T)
                found = True
                break
        
        if not found:
            iterations.append(MAX_ITERS)
            seeking_times.append(MAX_ITERS * T)
            
    return np.mean(seeking_times), np.mean(iterations)

def main():
    n_drones_list = [5, 10, 15, 20, 25, 30]
    n_runs = 40
    
    metrics = {
        'APSO': {'ts': [], 'iter': []},
        'PPO-PSO': {'ts': [], 'iter': []},
        'SPSO': {'ts': [], 'iter': []}
    }
    
    print(f"Starting Experiments with N={n_drones_list}, Runs={n_runs}")
    
    # Create results directory if not exists
    os.makedirs('results', exist_ok=True)
    
    for n in tqdm(n_drones_list, desc="Overall Progress"):
        print(f"\n\n--- Running Experiments for Swarm Size N={n} ---")

        # 1. APSO
        # monte_carlo_experiments_apso prints output
        res_apso = monte_carlo_experiments_apso(n_runs=n_runs, n_drones=n, side_length=100.0, max_iters=MAX_ITERS)
        metrics['APSO']['ts'].append(res_apso['mu_Ts'])
        metrics['APSO']['iter'].append(res_apso['mu_I'])
        # Ensure values explicitly match formulation
        print(f"[APSO] N={n}: µ(Ts) = {res_apso['mu_Ts']:.4f} s, µ(I) = {res_apso['mu_I']:.4f}")
        
        # 2. SPSO
        ts_spso, iter_spso = run_spso_mc(n, n_runs)
        metrics['SPSO']['ts'].append(ts_spso)
        metrics['SPSO']['iter'].append(iter_spso)
        print(f"[SPSO] N={n}: µ(Ts) = {ts_spso:.4f} s, µ(I) = {iter_spso:.4f}")
        
        # 3. PPO-PSO
        ts_ppo, iter_ppo = run_ppopso_mc(n, n_runs)
        metrics['PPO-PSO']['ts'].append(ts_ppo)
        metrics['PPO-PSO']['iter'].append(iter_ppo)
        print(f"[PPO-PSO] N={n}: µ(Ts) = {ts_ppo:.4f} s, µ(I) = {iter_ppo:.4f}")
    
    print("\nExperiments Complete. Generating Plots...")
        
    # Plotting
    # A. Source Seeking Time
    plt.figure(figsize=(10, 6))
    plt.plot(n_drones_list, metrics['APSO']['ts'], 'o-', label='APSO')
    plt.plot(n_drones_list, metrics['PPO-PSO']['ts'], 's-', label='PPO-PSO')
    plt.plot(n_drones_list, metrics['SPSO']['ts'], '^-', label='SPSO')
    plt.xlabel('Number of Drones (N)')
    plt.ylabel('Average Source Seeking Time (s)')
    plt.title('Average Source Seeking Time vs Swarm Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/avg_seeking_time_vs_n.png')
    
    # B. Average Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(n_drones_list, metrics['APSO']['iter'], 'o-', label='APSO')
    plt.plot(n_drones_list, metrics['PPO-PSO']['iter'], 's-', label='PPO-PSO')
    plt.plot(n_drones_list, metrics['SPSO']['iter'], '^-', label='SPSO')
    plt.xlabel('Number of Drones (N)')
    plt.ylabel('Average Number of Iterations')
    plt.title('Average Iterations vs Swarm Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/avg_iterations_vs_n.png')
    
    print("Plots saved to results/avg_seeking_time_vs_n.png and results/avg_iterations_vs_n.png")

if __name__ == "__main__":
    main()
