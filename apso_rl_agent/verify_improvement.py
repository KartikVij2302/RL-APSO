import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time

# reproducibility
SEED = 1234
np.random.seed(SEED)

# Import your modules
from apso import APSO_SourceSeeker, measure_signal, validate_apso_params
from PPO import PPOAgent

# ---------------------------------------------------------
# 1. Helper to calculate State (Must match your Training Env)
# ---------------------------------------------------------
def get_rl_state(apso_instance, prev_signal, current_iter, max_iter):
    """
    State vector MUST match what the policy was trained on.
    We include:
      - diversity
      - signal_change
      - normalized iteration
      - normalized apso params (w1,w2,c1,c2)
    => 7-dimensional state (float32)
    """
    # 1. Swarm Diversity
    dists = [np.linalg.norm(p.x - apso_instance.gbest_x) for p in apso_instance.particles]
    diversity = np.mean(dists) if dists else 0.0

    # 2. Signal Change
    current_signal = apso_instance.gbest_signal
    signal_change = current_signal - prev_signal

    # 3. Normalized Time
    norm_iter = current_iter / max(1, max_iter)
    avg_vel = np.mean([np.linalg.norm(p.v) for p in apso_instance.particles])
    # 4. APSO params (normalized)
    w1 = getattr(apso_instance, "w1", 0.0)
    w2 = getattr(apso_instance, "w2", 0.0)
    c1 = getattr(apso_instance, "c1", 1.0)
    c2 = getattr(apso_instance, "c2", 1.0)

    # Normalizations used during training:
    w1_n = np.clip(w1 / 2.0, -1.0, 1.0)   # assume w1 roughly in [-2,2]
    w2_n = np.clip(w2 / 2.0, -1.0, 1.0)   # assume w2 roughly in [-2,2]
    c1_n = np.clip(c1 / 5.0, 0.0, 1.0)    # c1 in [0,5]
    c2_n = np.clip(c2 / 5.0, 0.0, 1.0)    # c2 in [0,5]

    state = np.array([diversity, signal_change, norm_iter,avg_vel, w1_n, w2_n, c1_n, c2_n], dtype=np.float32)
    return state

# ---------------------------------------------------------
# 1b. Reusable mapping from action [-1,1]^4 -> APSO params
#     Must match training mapping exactly
# ---------------------------------------------------------
def map_action_to_params(action):
    """
    Maps action in [-1,1] to APSO params (w1,w2,c1,c2).
    Mapping used during training:
      w1: [-1.0, 1.5]
      w2: [-1.0, 1.0]
      c1: [0.01, 3.0]
      c2: [0.01, 3.0]
    """
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    w1 = -1.0 + (a[0] + 1.0) * (1.5 - (-1.0)) / 2.0   # [-1,1] -> [-1.0,1.5]
    w2 = -1.0 + (a[1] + 1.0) * (1.0 - (-1.0)) / 2.0   # [-1,1] -> [-1.0,1.0]
    c1 = 0.01 + (a[2] + 1.0) * (3.0 - 0.01) / 2.0     # [-1,1] -> [0.01,3.0]
    c2 = 0.01 + (a[3] + 1.0) * (3.0 - 0.01) / 2.0     # [-1,1] -> [0.01,3.0]
    return w1, w2, c1, c2

# ---------------------------------------------------------
# 2. The RL-Guided Loop (evaluation)
# ---------------------------------------------------------
def run_rl_guided_apso(agent, n_runs=30, max_iter=500, num_particles=20, source=None):
    """
    Runs APSO but asks the RL Agent for parameters (w1,w2,c1,c2) every step.
    Deterministic evaluation: we try to request deterministic action if the agent supports it.
    """
    results = {
        "run": [], "Ts": [], "I": [], "SD": [], "Success": [], "time_elapsed": []
    }

    # Configuration (Same as your training)
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    if source is None:
        source = np.array([50.0, 50.0])

    for r in range(n_runs):
        start_time = time.time()
        # Initialize APSO (physics)
        apso = APSO_SourceSeeker(
            objective=lambda x: 0.0, bounds=(lo, hi), source_pos=source,
            num_particles=num_particles,
            # initial placeholders (policy will overwrite immediately on first step)
            w1=0.675, w2=-0.285, c1=1.193, c2=1.193,
            S_s=1.0, alpha=0.01, termination_dist=0.1
        )

        prev_signal = apso.gbest_signal
        found = False
        iteration = 0

        # track total distance moved by particles (assumes particles have dist_travelled attr)
        for t in range(max_iter):
            state = get_rl_state(apso, prev_signal, t, max_iter)

            # get deterministic action if possible
            try:
                action, _ = agent.select_action(state, deterministic=True)
            except TypeError:
                # agent.select_action may not accept deterministic arg
                action, _ = agent.select_action(state)
            except Exception:
                # fallback: try without logprob
                try:
                    action = agent.select_action(state)
                    if isinstance(action, tuple):
                        action = action[0]
                except Exception as e:
                    print(f"[Warning] agent.select_action failed: {e}. Using zeros action.")
                    action = np.zeros(4, dtype=np.float32)

            # decode action into APSO params (same mapping as training env)
            w1, w2, c1, c2 = map_action_to_params(action)

            # clamp/validate to safe ranges
            # c1 = max(0.01, c1)
            # c2 = max(0.01, c2)

            try:
                validate_apso_params(w1, w2, c1, c2, getattr(apso, "T", 1.0))
                valid = True
            except Exception:
                valid = False

            if not valid:
                # if invalid, apply small perturbation towards safe defaults (safety)
                w1, w2, c1, c2 = 0.675, -0.285, 1.193, 1.193

            # apply params for next APSO iteration
            apso.w1 = w1
            apso.w2 = w2
            apso.c1 = c1
            apso.c2 = c2

            # Step physics
            try:
                found, min_dist = apso.step()
            except Exception as e:
                print(f"[Warning] apso.step() error: {e}")
                found = False
                min_dist = np.inf

            prev_signal = apso.gbest_signal
            iteration += 1

            if found:
                break

        # Calculate run metrics
        # total swarm distance (if your particle objects track dist_travelled)
        total_sd = 0.0
        try:
            total_sd = sum(getattr(p, "dist_travelled", 0.0) for p in apso.particles)
        except Exception:
            total_sd = 0.0

        # Time: use travel of finder particle / assumed speed, else penalty
        speed = 10.0
        if found:
            try:
                finder = min(apso.particles, key=lambda p: np.linalg.norm(p.x - source))
                time_s = getattr(finder, "dist_travelled", max_iter) / speed
            except Exception:
                time_s = 0.0
        else:
            time_s = float(max_iter)

        elapsed = time.time() - start_time

        results["run"].append(r)
        results["Ts"].append(time_s)
        results["I"].append(iteration)
        results["SD"].append(total_sd)
        results["Success"].append(1 if found else 0)
        results["time_elapsed"].append(elapsed)

    return results

# ---------------------------------------------------------
# 3. Baseline: try built-in Monte Carlo, else run manual baseline
# ---------------------------------------------------------
def run_fixed_baseline(n_runs=50, max_iter=500, num_particles=20, source=None):
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    if source is None:
        source = np.array([50.0, 50.0])

    baseline = APSO_SourceSeeker(
        objective=lambda x: 0.0, bounds=(lo, hi), source_pos=source,
        num_particles=num_particles,
        w1=0.675, w2=-0.285, c1=1.193, c2=1.193,
        T=1.0, termination_dist=0.1
    )

    # try run_monte_carlo if available and returns expected keys
    try:
        mc = baseline.run_monte_carlo(runs=n_runs, max_iter=max_iter)
        # Accept different key naming conventions
        if isinstance(mc, dict):
            # try common keys
            if "Ts_list" in mc and "I_list" in mc and "SD_list" in mc:
                return {
                    "Ts": list(mc["Ts_list"]),
                    "I": list(mc["I_list"]),
                    "SD": list(mc["SD_list"]),
                    "Success": list(mc.get("Success_list", [1]*n_runs))
                }
            # if keys differ, attempt guess
            if "Ts" in mc and "I" in mc and "SD" in mc:
                return {"Ts": list(mc["Ts"]), "I": list(mc["I"]), "SD": list(mc["SD"]), "Success": list(mc.get("Success", [1]*n_runs))}
        # if return format unexpected, fall back
        print("[Info] run_monte_carlo returned unexpected format; falling back to manual baseline.")
    except Exception:
        print("[Info] run_monte_carlo not available or failed; running manual baseline.")

    # Manual baseline simulation (same logic as RL loop but with fixed params)
    results = {"Ts": [], "I": [], "SD": [], "Success": []}
    for r in range(n_runs):
        apso = APSO_SourceSeeker(
            objective=lambda x: 0.0, bounds=(np.array([0.,0.]), np.array([100.,100.])),
            source_pos=source, num_particles=num_particles,
            w1=0.675, w2=-0.285, c1=1.193, c2=1.193, T=1.0, termination_dist=0.1
        )
        found = False
        prev_signal = apso.gbest_signal
        for t in range(max_iter):
            try:
                found, min_dist = apso.step()
            except Exception:
                found = False
                min_dist = np.inf
            prev_signal = apso.gbest_signal
            if found:
                break

        total_sd = sum(getattr(p, "dist_travelled", 0.0) for p in apso.particles)
        speed = 10.0
        if found:
            try:
                finder = min(apso.particles, key=lambda p: np.linalg.norm(p.x - source))
                time_s = getattr(finder, "dist_travelled", 0.0) / speed
            except Exception:
                time_s = 0.0
        else:
            time_s = float(max_iter)

        results["Ts"].append(time_s)
        results["I"].append(t+1)
        results["SD"].append(total_sd)
        results["Success"].append(1 if found else 0)

    return results

# ---------------------------------------------------------
# 4. Main Comparison Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # A. Load Trained Agent
    state_dim = 8   # must match get_rl_state output
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim, lr=0.0003)

    # try loading saved weights gracefully
    model_path = "apso_rl_agent/ppo_apso.pth"
    try:
        # try common load API
        try:
            agent.load(model_path)
            print(f"[Info] Loaded agent via agent.load('{model_path}').")
        except Exception:
            # some implementations expose 'load_state_dict' etc
            if hasattr(agent, "load_state_dict"):
                sd = torch.load(model_path, map_location="cpu")
                agent.load_state_dict(sd)
                print(f"[Info] Loaded agent via load_state_dict from '{model_path}'.")
            else:
                print("[Warning] Agent has no recognized load method; proceeding with uninitialized agent.")
    except Exception as e:
        print(f"[Warning] Failed to load agent from {model_path}: {e}. Proceeding with fresh agent (not ideal).")

    N_RUNS = 50
    MAX_ITER = 1000
    NUM_PARTICLES = 20
    SOURCE = np.array([50.0, 50.0])

    print(f"--- Running Comparative Analysis ({N_RUNS} runs) ---")

    # B. Run Baseline (Fixed APSO)
    print("1. Running Fixed Baseline...")
    baseline_results = run_fixed_baseline(n_runs=N_RUNS, max_iter=MAX_ITER, num_particles=NUM_PARTICLES, source=SOURCE)

    base_metrics = {
        "Ts_mean": np.mean(baseline_results["Ts"]),
        "Ts_std": np.std(baseline_results["Ts"]),
        "I_mean": np.mean(baseline_results["I"]),
        "SD_mean": np.mean(baseline_results["SD"]),
        "Success_rate": np.mean(baseline_results["Success"])
    }

    # C. Run RL-Guided APSO
    print("2. Running RL-Guided Swarm...")
    rl_results = run_rl_guided_apso(agent, n_runs=N_RUNS, max_iter=MAX_ITER, num_particles=NUM_PARTICLES, source=SOURCE)

    rl_metrics = {
        "Ts_mean": np.mean(rl_results["Ts"]),
        "Ts_std": np.std(rl_results["Ts"]),
        "I_mean": np.mean(rl_results["I"]),
        "SD_mean": np.mean(rl_results["SD"]),
        "Success_rate": np.mean(rl_results["Success"])
    }

    # D. Final Report
    print("\n" + "="*70)
    print(f"{'METRIC':<25} | {'FIXED APSO':<20} | {'RL-GUIDED':<20}")
    print("-" * 70)
    print(f"{'Avg Time (s)':<25} | {base_metrics['Ts_mean']:<8.2f} ±{base_metrics['Ts_std']:<6.2f} | {rl_metrics['Ts_mean']:<8.2f} ±{rl_metrics['Ts_std']:<6.2f}")
    print(f"{'Avg Iterations':<25} | {base_metrics['I_mean']:<20.2f} | {rl_metrics['I_mean']:<20.2f}")
    print(f"{'Avg Swarm Dist (m)':<25} | {base_metrics['SD_mean']:<20.2f} | {rl_metrics['SD_mean']:<20.2f}")
    print(f"{'Success Rate':<25} | {base_metrics['Success_rate']:<20.2f} | {rl_metrics['Success_rate']:<20.2f}")
    print("="*70)

    # E. Save per-run results to CSV for deeper analysis
    df_fixed = pd.DataFrame({
        "Ts": baseline_results["Ts"],
        "I": baseline_results["I"],
        "SD": baseline_results["SD"],
        "Success": baseline_results["Success"],
        "Type": "Fixed"
    })

    df_rl = pd.DataFrame({
        "Ts": rl_results["Ts"],
        "I": rl_results["I"],
        "SD": rl_results["SD"],
        "Success": rl_results["Success"],
        "Type": "RL"
    })

    df_all = pd.concat([df_fixed, df_rl], ignore_index=True)
    out_csv = "rl_vs_fixed_results.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"[Info] Per-run results saved to {out_csv}")

    # F. Visualization: bar + histograms
    # bar summary
    metrics_names = ['Avg Time (s)', 'Avg Iterations', 'Avg Swarm Dist (m)']
    fixed_vals = [base_metrics['Ts_mean'], base_metrics['I_mean'], base_metrics['SD_mean']]
    rl_vals = [rl_metrics['Ts_mean'], rl_metrics['I_mean'], rl_metrics['SD_mean']]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, fixed_vals, width, label='Fixed APSO')
    ax.bar(x + width/2, rl_vals, width, label='RL-Guided')
    ax.set_ylabel('Value')
    ax.set_title('Performance Comparison: Fixed vs RL-Dynamic APSO')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    plt.tight_layout()
    plt.savefig("final_comparison.png")
    print("[Info] Saved final comparison plot to final_comparison.png")

    # histograms for Ts
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(baseline_results["Ts"], bins=20)
    axes[0].set_title("Fixed APSO: Time (s) distribution")
    axes[1].hist(rl_results["Ts"], bins=20)
    axes[1].set_title("RL-Guided: Time (s) distribution")
    plt.tight_layout()
    plt.savefig("times_histograms.png")
    print("[Info] Saved times histograms to times_histograms.png")

    # G. Reward component analysis (from training)
    # Loads mean reward terms that were logged during RL-APSO training
    reward_stats_path = "apso_rl_agent/reward_component_means.npz"
    try:
        stats = np.load(reward_stats_path)
        time_cost_mean = float(stats["step_time_cost"])
        iter_penalty_mean = float(stats["iteration_penalty"])
        proximity_mean = float(stats["proximity_bonus"])
        success_mean = float(stats["success_bonus"])
        timeout_mean = float(stats["timeout_penalty"])

        print("\nReward component means from training (per step):")
        print(f"  Step time cost term (negative = time penalty): {time_cost_mean:.4f}")
        print(f"  Iteration penalty term: {iter_penalty_mean:.4f}")
        print(f"  Proximity bonus term: {proximity_mean:.4f}")
        print(f"  Success bonus term: {success_mean:.4f}")
        print(f"  Timeout penalty term: {timeout_mean:.4f}")
    except FileNotFoundError:
        print(f"[Info] Reward component means file not found at {reward_stats_path}. Run rl_enhanced_apso.py training first to generate it.")
    except Exception as e:
        print(f"[Warning] Failed to load reward component means from {reward_stats_path}: {e}")

    plt.show()
