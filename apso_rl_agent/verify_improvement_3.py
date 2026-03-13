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

from apso import APSO_SourceSeeker, validate_apso_params
from PPO import PPOAgent


# ---------------------------------------------------------
# Global Plot Style (publication quality)
# ---------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150
})

APSO_COLOR = "#1f77b4"
RL_COLOR = "#ff7f0e"


# ---------------------------------------------------------
# 1. Helper to calculate State
# ---------------------------------------------------------
def get_rl_state(apso_instance, prev_signal, current_iter, max_iter, num_particles):

    current_signal = apso_instance.gbest_signal
    signal_change = current_signal - prev_signal

    time_left = 1.0 - (current_iter / max(1, max_iter))
    avg_vel = np.mean([np.linalg.norm(p.v) for p in apso_instance.particles])

    w1 = getattr(apso_instance, "w1", 0.0)
    w2 = getattr(apso_instance, "w2", 0.0)
    c1 = getattr(apso_instance, "c1", 1.0)
    c2 = getattr(apso_instance, "c2", 1.0)

    w1_n = np.clip(w1 / 2.0, -1.0, 1.0)
    w2_n = np.clip(w2 / 2.0, -1.0, 1.0)
    c1_n = np.clip(c1 / 5.0, 0.0, 1.0)
    c2_n = np.clip(c2 / 5.0, 0.0, 1.0)

    num_particles_n = (float(num_particles) - 5.0) / 25.0

    state = np.array(
        [
            signal_change,
            time_left,
            avg_vel,
            w1_n,
            w2_n,
            c1_n,
            c2_n,
            num_particles_n,
        ],
        dtype=np.float32,
    )

    return state


# ---------------------------------------------------------
# 1b. Action → APSO parameters
# ---------------------------------------------------------
def map_action_to_params(apso_instance, action):

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
# 2. RL evaluation
# ---------------------------------------------------------
def run_rl_guided_apso(agent, n_runs=30, max_iter=500, num_particles=20, source=None, side_length=100.0):

    results = {"run": [], "Ts": [], "I": [], "SD": [], "Success": [], "time_elapsed": []}

    lo = np.array([0.0, 0.0])
    hi = np.array([float(side_length), float(side_length)])

    if source is None:
        half = float(side_length) * 0.5
        source = np.array([half, half])

    speed = 10.0

    print(f"[Info] Starting RL-APSO evaluation: runs={n_runs}, particles={num_particles}")

    for r in range(n_runs):

        start_time = time.time()

        apso = APSO_SourceSeeker(
            objective=lambda x: 0.0,
            bounds=(lo, hi),
            source_pos=source,
            num_particles=num_particles,
            w1=0.675,
            w2=-0.285,
            c1=1.193,
            c2=1.193,
            S_s=1.0,
            alpha=0.01,
            termination_dist=0.1,
        )

        prev_signal = apso.gbest_signal
        found = False
        iteration = 0

        for t in range(max_iter):

            state = get_rl_state(apso, prev_signal, t, max_iter, num_particles)

            try:
                action, _ = agent.select_action(state, deterministic=True)
            except TypeError:
                action, _ = agent.select_action(state)

            w1, w2, c1, c2 = map_action_to_params(apso, action)

            try:
                validate_apso_params(w1, w2, c1, c2, getattr(apso, "T", 1.0))
            except Exception:
                print("[Warning] Invalid APSO params, reverting to default.")
                w1, w2, c1, c2 = 0.675, -0.285, 1.193, 1.193

            apso.w1 = w1
            apso.w2 = w2
            apso.c1 = c1
            apso.c2 = c2

            found, min_dist = apso.step()

            prev_signal = apso.gbest_signal
            iteration += 1

            if found:
                break

        total_sd = sum(getattr(p, "dist_travelled", 0.0) for p in apso.particles)

        if found:
            finder = min(apso.particles, key=lambda p: np.linalg.norm(p.x - source))
            time_s = float(getattr(finder, "dist_travelled", 0.0)) / speed
        else:
            time_s = float(total_sd) / speed

        elapsed = time.time() - start_time

        results["run"].append(r)
        results["Ts"].append(time_s)
        results["I"].append(iteration)
        results["SD"].append(total_sd)
        results["Success"].append(1 if found else 0)
        results["time_elapsed"].append(elapsed)

    print("[Info] RL-APSO evaluation completed.")

    return results


# ---------------------------------------------------------
# 3. Baseline APSO
# ---------------------------------------------------------
def run_fixed_baseline(n_runs=50, max_iter=500, num_particles=20, source=None, side_length=100.0):

    print(f"[Info] Running baseline APSO: runs={n_runs}, particles={num_particles}")

    lo = np.array([0.0, 0.0])
    hi = np.array([float(side_length), float(side_length)])

    if source is None:
        source = np.array([side_length * 0.5, side_length * 0.5])

    results = {"Ts": [], "I": [], "SD": [], "Success": []}

    for r in range(n_runs):

        apso = APSO_SourceSeeker(
            objective=lambda x: 0.0,
            bounds=(lo, hi),
            source_pos=source,
            num_particles=num_particles,
            w1=0.675,
            w2=-0.285,
            c1=1.193,
            c2=1.193,
            T=1.0,
            termination_dist=0.1,
        )

        found = False

        for t in range(max_iter):

            try:
                found, min_dist = apso.step()
            except Exception:
                print("[Warning] APSO step error.")
                found = False

            if found:
                break

        total_sd = sum(getattr(p, "dist_travelled", 0.0) for p in apso.particles)

        speed = 10.0

        if found:
            finder = min(apso.particles, key=lambda p: np.linalg.norm(p.x - source))
            time_s = float(getattr(finder, "dist_travelled", 0.0)) / speed
        else:
            time_s = float(total_sd) / speed

        results["Ts"].append(time_s)
        results["I"].append(t + 1)
        results["SD"].append(total_sd)
        results["Success"].append(1 if found else 0)

    print("[Info] Baseline APSO completed.")

    return results


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    random.seed(SEED)
    torch.manual_seed(SEED)

    state_dim = 8
    action_dim = 4

    agent = PPOAgent(state_dim, action_dim, lr=0.0003)

    model_path = "models/latest_ppo_apso_random_particles_12.pth"

    print(f"[Info] Using model from {model_path}")

    try:
        agent.load(model_path)
        print(f"[Info] Loaded agent via agent.load('{model_path}')")
    except Exception as e:
        print(f"[Warning] Failed to load model: {e}")

    N_RUNS = 100
    MAX_ITER = 400

    fixed_source = np.array([50.0,50.0])
    swarm_sizes = list(range(5,31))

    mean_Ts_fixed=[]
    mean_Ts_rl=[]
    mean_I_fixed=[]
    mean_I_rl=[]
    mean_SD_fixed=[]
    mean_SD_rl=[]

    print("\n--- Fixed grid (100x100) & fixed source (50,50): swarm-size sweep ---")

    for n_particles in swarm_sizes:

        print(f"[Info] n_particles = {n_particles}")

        base_res=run_fixed_baseline(N_RUNS,MAX_ITER,n_particles,fixed_source)
        rl_res=run_rl_guided_apso(agent,N_RUNS,MAX_ITER,n_particles,fixed_source)

        mean_Ts_fixed.append(np.mean(base_res["Ts"]))
        mean_Ts_rl.append(np.mean(rl_res["Ts"]))
        mean_I_fixed.append(np.mean(base_res["I"]))
        mean_I_rl.append(np.mean(rl_res["I"]))
        mean_SD_fixed.append(np.mean(base_res["SD"]))
        mean_SD_rl.append(np.mean(rl_res["SD"]))

    print("[Info] Evaluation sweep finished.")

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------

    plt.figure(figsize=(7,5))
    plt.plot(swarm_sizes,mean_Ts_fixed,"o-",linewidth=2,label="APSO",color=APSO_COLOR)
    plt.plot(swarm_sizes,mean_Ts_rl,"s-",linewidth=2,label="RL-APSO",color=RL_COLOR)
    plt.xlabel("Swarm Size")
    plt.ylabel("Source Seeking Time $T_s$")
    plt.title("Source Seeking Time vs Swarm Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Ts_vs_swarm_size.png",dpi=300)
    print("[Info] Saved plot Ts_vs_swarm_size.png")

    plt.figure(figsize=(7,5))
    plt.plot(swarm_sizes,mean_I_fixed,"o-",linewidth=2,label="APSO",color=APSO_COLOR)
    plt.plot(swarm_sizes,mean_I_rl,"s-",linewidth=2,label="RL-APSO",color=RL_COLOR)
    plt.xlabel("Swarm Size")
    plt.ylabel("Iterations")
    plt.title("Iterations vs Swarm Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Iterations_vs_swarm_size.png",dpi=300)
    print("[Info] Saved plot Iterations_vs_swarm_size.png")

    plt.figure(figsize=(7,5))
    plt.plot(swarm_sizes,mean_SD_fixed,"o-",linewidth=2,label="APSO",color=APSO_COLOR)
    plt.plot(swarm_sizes,mean_SD_rl,"s-",linewidth=2,label="RL-APSO",color=RL_COLOR)
    plt.xlabel("Swarm Size")
    plt.ylabel("Swarm Distance $D_{swarm}$")
    plt.title("Swarm Distance vs Swarm Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig("SwarmDistance_vs_swarm_size.png",dpi=300)
    print("[Info] Saved plot SwarmDistance_vs_swarm_size.png")

    improvement=(np.array(mean_I_fixed)-np.array(mean_I_rl))/np.array(mean_I_fixed)*100

    plt.figure(figsize=(6,4))
    plt.plot(swarm_sizes,improvement,"o-",linewidth=2,color="#2ca02c")
    plt.xlabel("Swarm Size")
    plt.ylabel("Iteration Improvement (%)")
    plt.title("RL-APSO Improvement over APSO")
    plt.tight_layout()
    plt.savefig("rl_apso_improvement.png",dpi=300)
    print("[Info] Saved plot rl_apso_improvement.png")

    print("[Info] All plots saved successfully.")

    plt.show()