import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import random
import sys

# reproducibility
SEED = 42
np.random.seed(SEED)

# Import your modules
from apso import APSO_SourceSeeker, validate_apso_params
from PPO import PPOAgent

# ---------------------------------------------------------
# 1. Helper to calculate State (Must match your Training Env)
# ---------------------------------------------------------
def get_rl_state(apso_instance, prev_signal, current_iter, max_iter,num_particles):
    """
    State vector MUST match what the policy was trained on.
    We include:
      - diversity
      - signal_change
      - normalized iteration
      - normalized apso params (w1,w2,c1,c2)
    => 9-dimensional state (float32)
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

    state = np.array([diversity, signal_change, norm_iter,avg_vel, w1_n, w2_n, c1_n, c2_n,float(num_particles)], dtype=np.float32)
    return state

# ---------------------------------------------------------
# 1b. Reusable mapping from action [-1,1]^4 -> APSO params
#     Must match training mapping in RLAPSOEnv exactly
# ---------------------------------------------------------
def map_action_to_params(apso_instance, action):
        """Map PPO action in [-1,1]^4 to APSO parameters.

        Mirrors RLAPSOEnv._map_action_to_params: parameters are updated
        multiplicatively around their current values using a fractional
        delta in [-0.2, 0.2].
        """
        delta_frac = 0.2
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        w1_cur = getattr(apso_instance, "w1", 0.675)
        w2_cur = getattr(apso_instance, "w2", -0.285)
        c1_cur = getattr(apso_instance, "c1", 1.193)
        c2_cur = getattr(apso_instance, "c2", 1.193)

        w1 = w1_cur * (1.0 + delta_frac * a[0])
        w2 = w2_cur * (1.0 + delta_frac * a[1])
        c1 = c1_cur * (1.0 + delta_frac * a[2])
        c2 = c2_cur * (1.0 + delta_frac * a[3])

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
            state = get_rl_state(apso, prev_signal, t, max_iter,num_particles)

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
            w1, w2, c1, c2 = map_action_to_params(apso, action)

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
# 3. Baseline:un manual baseline
# ---------------------------------------------------------
def run_fixed_baseline(n_runs=50, max_iter=500, num_particles=20, source=None):
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    if source is None:
        source = np.array([50.0, 50.0])

    # Manual baseline simulation (same logic as RL loop but with fixed params)
    results = {"Ts": [], "I": [], "SD": [], "Success": []}
    for r in range(n_runs):
        apso = APSO_SourceSeeker(
            objective=lambda x: 0.0, bounds=(lo, hi),
            source_pos=source, num_particles=num_particles,
            w1=0.675, w2=-0.285, c1=1.193, c2=1.193, T=1.0, termination_dist=0.1
        )
        found = False
        for t in range(max_iter):
            try:
                found, min_dist = apso.step()
            except Exception:
                found = False
                min_dist = np.inf
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
    # A. Set seeds for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # B. Load Trained Agent
    state_dim = 9   # must match get_rl_state output
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim, lr=0.0003)

    # Determine which trained model to load.
    # Usage examples (from workspace root):
    #   python apso_rl_agent/verify_improvement.py fixed
    #   python apso_rl_agent/verify_improvement.py random
    model_choice = "random"  # default
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("fixed", "fixed_source"):
            model_choice = "fixed"
        elif arg in ("random", "random_particles", "var_particles"):
            model_choice = "random"

    if model_choice == "fixed":
        model_path = "apso_rl_agent/latest_ppo_apso_fixed_source_2.pth"
    else:
        model_path = "apso_rl_agent/latest_ppo_apso_random_particles_2.pth"

    print(f"[Info] Using '{model_choice}' model from {model_path}")
    try:
        try:
            agent.load(model_path)
            print(f"[Info] Loaded agent via agent.load('{model_path}').")
        except Exception:
            if hasattr(agent, "load_state_dict"):
                sd = torch.load(model_path, map_location="cpu")
                agent.load_state_dict(sd)
                print(f"[Info] Loaded agent via load_state_dict from '{model_path}'.")
            else:
                print("[Warning] Agent has no recognized load method; proceeding with uninitialized agent.")
    except Exception as e:
        print(f"[Warning] Failed to load agent from {model_path}: {e}. Proceeding with fresh agent (not ideal).")

    # Evaluation configuration
    N_RUNS = 100
    MAX_ITER = 300
    NUM_PARTICLES = 20
    NUM_SOURCES = 25

    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])

    # C. Fixed-source comparison at (50, 50) over multiple runs
    fixed_source = np.array([50.0, 50.0])
    print("\n--- Fixed-source comparison at (50,50) over multiple runs ---")

    fixed_baseline = run_fixed_baseline(
        n_runs=N_RUNS,
        max_iter=MAX_ITER,
        num_particles=NUM_PARTICLES,
        source=fixed_source,
    )

    fixed_rl = run_rl_guided_apso(
        agent,
        n_runs=N_RUNS,
        max_iter=MAX_ITER,
        num_particles=NUM_PARTICLES,
        source=fixed_source,
    )

    # Per-run rows for CSV (same format as earlier rl_vs_fixed_results.csv)
    fixed_rows = []
    for r in range(N_RUNS):
        fixed_rows.append({
            "Ts": fixed_baseline["Ts"][r],
            "I": fixed_baseline["I"][r],
            "SD": fixed_baseline["SD"][r],
            "Success": fixed_baseline["Success"][r],
            "Type": "Fixed",
        })
        fixed_rows.append({
            "Ts": fixed_rl["Ts"][r],
            "I": fixed_rl["I"][r],
            "SD": fixed_rl["SD"][r],
            "Success": fixed_rl["Success"][r],
            "Type": "RL",
        })

    fixed_df = pd.DataFrame(fixed_rows)
    fixed_out_csv = "rl_vs_fixed_results.csv"
    fixed_df.to_csv(fixed_out_csv, index=False)

    print("[Info] Fixed-source results saved to rl_vs_fixed_results.csv")
    print("[Summary @ (50,50)]")
    print(f"  Fixed APSO:  Ts={np.mean(fixed_baseline['Ts']):.3f}, I={np.mean(fixed_baseline['I']):.2f}, SD={np.mean(fixed_baseline['SD']):.2f}")
    print(f"  RL-APSO:     Ts={np.mean(fixed_rl['Ts']):.3f}, I={np.mean(fixed_rl['I']):.2f}, SD={np.mean(fixed_rl['SD']):.2f}")

    # When using the "random" model, skip the random-source multi-source
    # experiments and only run the fixed-source + particle-count sweep.
    if model_choice != "random":
        # Generate random source locations (reproducible via SEED above)
        sources = np.random.uniform(lo, hi, size=(NUM_SOURCES, 2))

        print(f"--- Running Comparative Analysis over {NUM_SOURCES} random sources ---")
        print(f"Each configuration: {N_RUNS} runs, max_iter={MAX_ITER}, particles={NUM_PARTICLES}\n")

        # Containers for per-source averaged metrics
        base_Ts_mean, base_I_mean, base_SD_mean = [], [], []
        rl_Ts_mean, rl_I_mean, rl_SD_mean = [], [], []

        # Also collect full per-run results for optional CSV export
        all_rows = []

        for idx, src in enumerate(sources):
            print(f"Source {idx+1}/{NUM_SOURCES} at {src}")

            # Fixed APSO baseline
            baseline_results = run_fixed_baseline(
                n_runs=N_RUNS,
                max_iter=MAX_ITER,
                num_particles=NUM_PARTICLES,
                source=src,
            )

            # RL-guided APSO
            rl_results = run_rl_guided_apso(
                agent,
                n_runs=N_RUNS,
                max_iter=MAX_ITER,
                num_particles=NUM_PARTICLES,
                source=src,
            )

            # Aggregate metrics for this source
            base_Ts_mean.append(np.mean(baseline_results["Ts"]))
            base_I_mean.append(np.mean(baseline_results["I"]))
            base_SD_mean.append(np.mean(baseline_results["SD"]))

            rl_Ts_mean.append(np.mean(rl_results["Ts"]))
            rl_I_mean.append(np.mean(rl_results["I"]))
            rl_SD_mean.append(np.mean(rl_results["SD"]))

            # Extend rows for CSV: one row per run per method
            for r in range(N_RUNS):
                all_rows.append({
                    "source_id": idx + 1,
                    "source_x": src[0],
                    "source_y": src[1],
                    "run": r,
                    "Ts": baseline_results["Ts"][r],
                    "I": baseline_results["I"][r],
                    "SD": baseline_results["SD"][r],
                    "Success": baseline_results["Success"][r],
                    "Type": "Fixed",
                })
                all_rows.append({
                    "source_id": idx + 1,
                    "source_x": src[0],
                    "source_y": src[1],
                    "run": r,
                    "Ts": rl_results["Ts"][r],
                    "I": rl_results["I"][r],
                    "SD": rl_results["SD"][r],
                    "Success": rl_results["Success"][r],
                    "Type": "RL",
                })

        base_Ts_mean = np.array(base_Ts_mean)
        base_I_mean = np.array(base_I_mean)
        base_SD_mean = np.array(base_SD_mean)

        rl_Ts_mean = np.array(rl_Ts_mean)
        rl_I_mean = np.array(rl_I_mean)
        rl_SD_mean = np.array(rl_SD_mean)

        # Print simple tabular summary
        print("\n" + "=" * 90)
        print(f"{'Source':<8} | {'Location (x,y)':<25} | {'Fixed Ts':>10} | {'RL Ts':>10} | {'Fixed I':>10} | {'RL I':>10} | {'Fixed SD':>10} | {'RL SD':>10}")
        print("-" * 90)
        for i, src in enumerate(sources):
            print(
                f"{i+1:<8d} | "
                f"({src[0]:6.2f}, {src[1]:6.2f}) | "
                f"{base_Ts_mean[i]:10.2f} | {rl_Ts_mean[i]:10.2f} | "
                f"{base_I_mean[i]:10.2f} | {rl_I_mean[i]:10.2f} | "
                f"{base_SD_mean[i]:10.2f} | {rl_SD_mean[i]:10.2f}"
            )
        print("=" * 90)

        # Summary of how often RL-APSO outperforms fixed APSO (lower is better for all metrics)
        rl_better_Ts = np.sum(rl_Ts_mean < base_Ts_mean)
        rl_better_I = np.sum(rl_I_mean < base_I_mean)
        rl_better_SD = np.sum(rl_SD_mean < base_SD_mean)

        rl_better_all = np.sum(
            (rl_Ts_mean < base_Ts_mean)
            & (rl_I_mean < base_I_mean)
            & (rl_SD_mean < base_SD_mean)
        )

        print("\nRL-Guided APSO vs Fixed APSO (per-source averages):")
        print(f"  Time Ts: RL better on {rl_better_Ts}/{NUM_SOURCES} sources")
        print(f"  Iterations I: RL better on {rl_better_I}/{NUM_SOURCES} sources")
        print(f"  Swarm distance SD: RL better on {rl_better_SD}/{NUM_SOURCES} sources")
        print(f"  All three metrics: RL better on {rl_better_all}/{NUM_SOURCES} sources")

        # Save per-run results and source positions
        df_all = pd.DataFrame(all_rows)
        out_csv = "rl_vs_fixed_multi_source_results.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"[Info] Per-run multi-source results saved to {out_csv}")

        src_df = pd.DataFrame({
            "source_id": np.arange(1, NUM_SOURCES + 1),
            "source_x": sources[:, 0],
            "source_y": sources[:, 1],
        })
        src_df.to_csv("evaluated_sources.csv", index=False)
        print("[Info] Source locations saved to evaluated_sources.csv")

        # F. Visualization: comparison plots per source
        x_idx = np.arange(NUM_SOURCES)
        width = 0.35

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

        # 1) Average source seeking time
        axes[0].bar(x_idx - width / 2, base_Ts_mean, width, label="Fixed APSO")
        axes[0].bar(x_idx + width / 2, rl_Ts_mean, width, label="RL-Guided APSO")
        axes[0].set_title("Average Source Seeking Time per Source")
        axes[0].set_ylabel("Time (s)")

        # 2) Average iterations
        axes[1].bar(x_idx - width / 2, base_I_mean, width, label="Fixed APSO")
        axes[1].bar(x_idx + width / 2, rl_I_mean, width, label="RL-Guided APSO")
        axes[1].set_title("Average Iterations per Source")
        axes[1].set_ylabel("Iterations")

        # 3) Average swarm distance
        axes[2].bar(x_idx - width / 2, base_SD_mean, width, label="Fixed APSO")
        axes[2].bar(x_idx + width / 2, rl_SD_mean, width, label="RL-Guided APSO")
        axes[2].set_title("Average Swarm Distance per Source")
        axes[2].set_ylabel("Total swarm distance (m)")

        for ax in axes:
            ax.set_xlabel("Source index")
            ax.set_xticks(x_idx)
            ax.set_xticklabels([str(i + 1) for i in range(NUM_SOURCES)])
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        axes[0].legend(loc="best")

        plt.tight_layout()
        plt.savefig("multi_source_comparison.png", dpi=300)
        print("[Info] Saved multi-source comparison plot to multi_source_comparison.png")
    else:
        print("[Info] 'random' model mode: skipping random-source multi-source tests; running only particle-count sweep.")

    # H. Average source seeking time vs number of UAVs (fixed source at (50,50))
    #    Evaluate a total of 100 UAV-count settings, each an integer in [5, 15],
    #    distributed as uniformly as possible across the range.
    rng = np.random.default_rng(SEED)
    allowed_counts = np.arange(5, 16, 1)  # 5..15 inclusive
    reps = 100 // len(allowed_counts)
    remainder = 100 % len(allowed_counts)
    uav_counts = np.concatenate([
        np.repeat(allowed_counts, reps),
        rng.choice(allowed_counts, size=remainder, replace=False) if remainder > 0 else np.array([], dtype=int),
    ])
    rng.shuffle(uav_counts)
    uav_counts = np.sort(uav_counts)
    avg_Ts_fixed = []
    avg_Ts_rl = []

    fixed_source = np.array([50.0, 50.0])

    for n_uav in uav_counts:
        print(f"[Info] Evaluating average Ts for {n_uav} UAVs at source (50,50)")
        base_res = run_fixed_baseline(
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            num_particles=n_uav,
            source=fixed_source,
        )
        rl_res = run_rl_guided_apso(
            agent,
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            num_particles=n_uav,
            source=fixed_source,
        )

        avg_Ts_fixed.append(np.mean(base_res["Ts"]))
        avg_Ts_rl.append(np.mean(rl_res["Ts"]))

    avg_Ts_fixed = np.array(avg_Ts_fixed)
    avg_Ts_rl = np.array(avg_Ts_rl)

    # Summary statistics over the 100 UAV-count evaluations
    # (each element is the mean Ts across N_RUNS runs at that UAV count)
    fixed_mean = float(np.mean(avg_Ts_fixed))
    fixed_var = float(np.var(avg_Ts_fixed))
    fixed_std = float(np.std(avg_Ts_fixed))

    rl_mean = float(np.mean(avg_Ts_rl))
    rl_var = float(np.var(avg_Ts_rl))
    rl_std = float(np.std(avg_Ts_rl))

    print("\n=== UAV-count sweep summary over 100 evaluations (Ts) ===")
    print("[Fixed APSO]")
    print(f"  Mean: {fixed_mean:.6f}")
    print(f"  Variance: {fixed_var:.6f}")
    print(f"  Std Dev: {fixed_std:.6f}")
    print("[RL-APSO]")
    print(f"  Mean: {rl_mean:.6f}")
    print(f"  Variance: {rl_var:.6f}")
    print(f"  Std Dev: {rl_std:.6f}")

    plt.figure(figsize=(7, 5))
    plt.plot(uav_counts, avg_Ts_fixed, "o-", label="Fixed APSO")
    plt.plot(uav_counts, avg_Ts_rl, "s-", label="RL-Guided APSO")
    plt.xlabel("Number of UAVs (particles)")
    plt.ylabel("Average source seeking time Ts (s)")
    plt.title("Average source seeking time vs number of UAVs\nSource at (50,50)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("avg_Ts_vs_num_uavs.png", dpi=300)
    print("[Info] Saved plot of average Ts vs number of UAVs to avg_Ts_vs_num_uavs.png")

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
