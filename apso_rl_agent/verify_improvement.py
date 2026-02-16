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
# 3. Baseline:run manual baseline
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
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    state_dim = 9
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim, lr=0.0003)

    model_choice = "random"  # default
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("fixed", "fixed_source"):
            model_choice = "fixed"
        elif arg in ("random", "random_particles", "var_particles"):
            model_choice = "random"

    if model_choice == "fixed":
        model_path = "apso_rl_agent/models/latest_ppo_apso_fixed_source_4.pth"
    else:
        model_path = "apso_rl_agent/models/latest_ppo_apso_random_particles_4.pth"

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

    # Evaluation configuration (FIXED grid + FIXED source, VARIABLE swarm size)
    N_RUNS = 100
    MAX_ITER = 300

    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    fixed_source = np.array([50.0, 50.0])

    # Sweep swarm sizes (edit as needed)
    swarm_sizes = list(range(5, 31))

    rows = []
    mean_Ts_fixed = []
    mean_Ts_rl = []

    print("\n--- Fixed grid (100x100) & fixed source (50,50): swarm-size sweep ---")
    for n_particles in swarm_sizes:
        print(f"[Info] n_particles={n_particles}")

        base_res = run_fixed_baseline(
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            num_particles=n_particles,
            source=fixed_source,
        )
        rl_res = run_rl_guided_apso(
            agent,
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            num_particles=n_particles,
            source=fixed_source,
        )

        # record per-run rows for CSV
        for r in range(N_RUNS):
            rows.append({
                "n_particles": n_particles,
                "run": r,
                "Ts": base_res["Ts"][r],
                "I": base_res["I"][r],
                "SD": base_res["SD"][r],
                "Success": base_res["Success"][r],
                "Type": "Fixed",
            })
            rows.append({
                "n_particles": n_particles,
                "run": r,
                "Ts": rl_res["Ts"][r],
                "I": rl_res["I"][r],
                "SD": rl_res["SD"][r],
                "Success": rl_res["Success"][r],
                "Type": "RL",
            })

        mean_Ts_fixed.append(float(np.mean(base_res["Ts"])))
        mean_Ts_rl.append(float(np.mean(rl_res["Ts"])))

    out_csv = "rl_vs_fixed_fixedsource_swarm_sweep.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[Info] Saved sweep results to {out_csv}")

    plt.figure(figsize=(7, 5))
    plt.plot(swarm_sizes, mean_Ts_fixed, "o-", label="Fixed APSO")
    plt.plot(swarm_sizes, mean_Ts_rl, "s-", label="RL-Guided APSO")
    plt.xlabel("Swarm size (n_particles)")
    plt.ylabel("Mean source seeking time Ts (s)")
    plt.title("Fixed grid & fixed source: Ts vs swarm size")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fixedsource_Ts_vs_swarm_size.png", dpi=300)
    print("[Info] Saved plot to fixedsource_Ts_vs_swarm_size.png")

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
