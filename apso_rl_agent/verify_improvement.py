import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Import your modules
from apso import APSO_SourceSeeker, measure_signal
from PPO import PPOAgent

# ---------------------------------------------------------
# 1. Helper to calculate State (Must match your Training Env)
# ---------------------------------------------------------
def get_rl_state(apso_instance, prev_signal, current_iter, max_iter):
    # 1. Swarm Diversity
    dists = [np.linalg.norm(p.x - apso_instance.gbest_x) for p in apso_instance.particles]
    diversity = np.mean(dists) if dists else 0.0
    
    # 2. Signal Change
    current_signal = apso_instance.gbest_signal
    signal_change = current_signal - prev_signal
    
    # 3. Normalized Time
    norm_iter = current_iter / max_iter
    
    return np.array([diversity, signal_change, norm_iter], dtype=np.float32)

# ---------------------------------------------------------
# 2. The RL-Guided Loop
# ---------------------------------------------------------
def run_rl_guided_apso(agent, n_runs=30, max_iter=500):
    """
    Runs APSO but asks the RL Agent for parameters (w1, w2, c1, c2) every step.
    """
    results = {
        "Ts": [], "I": [], "SD": [], "Success": []
    }
    
    # Configuration (Same as your training)
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    source = np.array([50.0, 50.0])
    
    for r in range(n_runs):
        # Initialize Standard APSO (Physics Engine)
        # We start with default params, but RL will overwrite them immediately
        apso = APSO_SourceSeeker(
            objective=lambda x: 0.0, bounds=(lo, hi), source_pos=source,
            num_particles=20, # Use same swarm size as training
            w1=0.675, w2=-0.285, c1=1.193, c2=1.193, # Initial placeholders
            S_s=1.0, alpha=0.01, termination_dist=0.1
        )
        
        prev_signal = apso.gbest_signal
        
        # Trackers
        step_params = [] # To visualize parameter evolution later if needed
        
        found = False
        iteration = 0
        
        for t in range(max_iter):
            # A. Get State
            state = get_rl_state(apso, prev_signal, t, max_iter)
            
            # B. Get Action from RL Agent (Deterministic for Evaluation)
            action, _ = agent.select_action(state)
            
            # C. Decode Action to Parameters (Same logic as env.step)
            # CLIPPED to ensure stability
            w1 = np.clip(0.6 + action[0] * 0.8, 0.1, 0.9)
            w2 = np.clip(0.4 + action[1] * 0.6, 0.1, 0.9)
            c1 = np.clip(1.0 + action[2] * 1.0, 0.1, 3.5)
            c2 = np.clip(1.0 + action[3] * 1.0, 0.1, 3.5)
            
            # D. OVERRIDE APSO Parameters
            apso.w1, apso.w2, apso.c1, apso.c2 = w1, w2, c1, c2
            step_params.append([w1, w2, c1, c2])
            
            # E. Step Physics
            prev_pos = np.array([p.x.copy() for p in apso.particles])
            found, min_dist = apso.step()
            
            # Update signal tracker
            prev_signal = apso.gbest_signal
            iteration += 1
            
            if found:
                break
        
        # Calculate Metrics for this run
        # 1. Swarm Distance
        total_sd = sum(p.dist_travelled for p in apso.particles)
        
        # 2. Time (Distance of finder / Speed) - Approx speed = 10.0
        # If found, use finder's distance. If not, use average.
        speed = 10.0
        if found:
            finder = min(apso.particles, key=lambda p: np.linalg.norm(p.x - source))
            time_s = finder.dist_travelled / speed
        else:
            time_s = max_iter # Penalty
            
        results["Ts"].append(time_s)
        results["I"].append(iteration)
        results["SD"].append(total_sd)
        results["Success"].append(1 if found else 0)

    return results

# ---------------------------------------------------------
# 3. Main Comparison Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # A. Load Trained Agent
    # Assuming 'agent' is available from your training script or loaded here
    # For this example, we re-initialize it (You should load weights!)
    state_dim = 3
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim, lr=0.0003)
    agent.load("apso_rl_agent/ppo_apso.pth") # <--- UNCOMMENT THIS AFTER SAVING YOUR MODEL
    
    N_RUNS = 50
    
    print(f"--- Running Comparative Analysis ({N_RUNS} runs) ---")
    
    # B. Run Baseline (Standard Fixed APSO)
    # Using the exact params from your paper/code:
    # w1=0.675, w2=-0.285, c1=1.193, c2=1.193
    print("1. Running Fixed Baseline...")
    
    # We use your existing APSO class for the baseline
    baseline_runner = APSO_SourceSeeker(
        objective=lambda x: 0.0, bounds=(np.zeros(2), np.ones(2)*100), source_pos=np.array([50,50]),
        num_particles=20, 
        w1=0.675, w2=-0.285, c1=1.193, c2=1.193, # Fixed Params
        T=1.0, termination_dist=0.1
    )
    # Your class has a built-in Monte Carlo runner, let's use it!
    base_results_raw = baseline_runner.run_monte_carlo(runs=N_RUNS, max_iter=500)
    
    base_metrics = {
        "Ts": np.mean(base_results_raw["Ts_list"]),
        "I": np.mean(base_results_raw["I_list"]),
        "SD": np.mean(base_results_raw["SD_list"])
    }

    # C. Run RL-Guided APSO
    print("2. Running RL-Guided Swarm...")
    rl_results_raw = run_rl_guided_apso(agent, n_runs=N_RUNS, max_iter=500)
    
    rl_metrics = {
        "Ts": np.mean(rl_results_raw["Ts"]),
        "I": np.mean(rl_results_raw["I"]),
        "SD": np.mean(rl_results_raw["SD"])
    }

    # D. Final Report
    print("\n" + "="*50)
    print(f"{'METRIC':<25} | {'FIXED APSO':<12} | {'RL-GUIDED':<12}")
    print("-" * 50)
    print(f"{'Avg Time (s)':<25} | {base_metrics['Ts']:<12.2f} | {rl_metrics['Ts']:<12.2f}")
    print(f"{'Avg Iterations':<25} | {base_metrics['I']:<12.2f} | {rl_metrics['I']:<12.2f}")
    print(f"{'Avg Swarm Dist (m)':<25} | {base_metrics['SD']:<12.2f} | {rl_metrics['SD']:<12.2f}")
    print("="*50)

    # E. Visualization
    metrics_names = ['Time (s)', 'Iterations', 'Swarm Dist (m)']
    fixed_vals = [base_metrics['Ts'], base_metrics['I'], base_metrics['SD']]
    rl_vals = [rl_metrics['Ts'], rl_metrics['I'], rl_metrics['SD']]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, fixed_vals, width, label='Fixed APSO')
    rects2 = ax.bar(x + width/2, rl_vals, width, label='RL-Guided')

    ax.set_ylabel('Value')
    ax.set_title('Performance Comparison: Fixed vs RL-Dynamic APSO')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    
    # Normalize y-axis log scale if Swarm Dist is huge compared to Time
    # ax.set_yscale('log') 

    plt.savefig("final_comparison.png")
    plt.show()