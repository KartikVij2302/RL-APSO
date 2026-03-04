import numpy as np
import random
from rl_apso import APSO_SourceSeeker
from PPO import PPOAgent
from rl_apso_error_term_approach import RLAPSOEnvErrorOnly  # <-- your new env file
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Utility: Run Standard APSO (fixed hyperparameters)
# ------------------------------------------------------------
UAV_SPEED = 10.0


def run_standard_apso(n_runs, num_particles, bounds, source, max_iter=1000):
    results = {"Ts": [], "I": [], "SD": [], "Success": []}

    for r in range(n_runs):
        apso = APSO_SourceSeeker(
            objective=lambda x: 0.0,
            bounds=bounds,
            source_pos=source,
            num_particles=num_particles,
            w1=0.675,
            w2=-0.285,
            c1=1.193,
            c2=1.193,
            T=1.0,
            S_s=1.0,
            alpha=0.01,
            termination_dist=0.1,
        )

        found = False
        iteration = 0
        min_dist = 0.0
        for t in range(max_iter):
            found, min_dist = apso.step()
            iteration += 1
            if found:
                break

        total_sd = sum(getattr(p, "dist_travelled", 0.0) for p in apso.particles)

        if found:
            finder = min(apso.particles, key=lambda p: np.linalg.norm(p.x - source))
            time_s = finder.dist_travelled / UAV_SPEED
        else:
            time_s = min_dist / UAV_SPEED

        results["Ts"].append(time_s)
        results["I"].append(iteration)
        results["SD"].append(total_sd)
        results["Success"].append(1 if found else 0)

    return results


# ------------------------------------------------------------
# Utility: Run RL Error-Term APSO
# ------------------------------------------------------------
def run_error_term_apso(agent, n_runs, num_particles, bounds, source, max_iter=500):
    results = {"Ts": [], "I": [], "SD": [], "Success": []}

    for r in range(n_runs):
        env = RLAPSOEnvErrorOnly(
            source_pos=source,
            bounds=bounds,
            num_particles=num_particles,
            max_iter=max_iter,
        )

        state = env.reset()
        found = False
        iteration = 0

        for t in range(max_iter):
            action, _ = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            iteration += 1
            if done:
                found = True
                break

        total_sd = sum(getattr(p, "dist_travelled", 0.0) for p in env.apso.particles)

        if found:
            finder = min(env.apso.particles, key=lambda p: np.linalg.norm(p.x - source))
            time_s = finder.dist_travelled / 10.0
        else:
            time_s = max_iter

        results["Ts"].append(time_s)
        results["I"].append(iteration)
        results["SD"].append(total_sd)
        results["Success"].append(1 if found else 0)

    return results


# ------------------------------------------------------------
# Helper: Print Results
# ------------------------------------------------------------
def print_results(title, baseline, rl):
    print("\n" + "=" * 70)
    print(title)
    print("-" * 70)

    def summarize(data):
        return (
            np.mean(data["Ts"]),
            np.mean(data["I"]),
            np.mean(data["SD"]),
            np.mean(data["Success"]),
        )

    base_vals = summarize(baseline)
    rl_vals = summarize(rl)

    print(f"{'METRIC':<25} | {'STANDARD APSO':<15} | {'ERROR-RL APSO':<15}")
    print("-" * 70)
    print(f"{'Avg Time (s)':<25} | {base_vals[0]:<15.2f} | {rl_vals[0]:<15.2f}")
    print(f"{'Avg Iterations':<25} | {base_vals[1]:<15.2f} | {rl_vals[1]:<15.2f}")
    print(f"{'Avg Swarm Dist':<25} | {base_vals[2]:<15.2f} | {rl_vals[2]:<15.2f}")
    print(f"{'Success Rate':<25} | {base_vals[3]:<15.2f} | {rl_vals[3]:<15.2f}")
    print("=" * 70)


# ------------------------------------------------------------
# Main Verification
# ------------------------------------------------------------
if __name__ == "__main__":

    # Load trained RL model
    state_dim = 8
    action_dim = 2
    agent = PPOAgent(state_dim, action_dim, lr=3e-4)
    agent.load("apso_rl_agent/models/latest_ppo_apso_error_term_random_particles.pth")

    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    bounds = (lo, hi)
    source = np.array([50.0, 50.0])

    # --------------------------------------------------------
    # TEST 1: Fixed Swarm Size
    # --------------------------------------------------------
    print("Running Test 1: Fixed Swarm Size (20)")
    baseline_fixed = run_standard_apso(
        n_runs=50, num_particles=20, bounds=bounds, source=source
    )

    rl_fixed = run_error_term_apso(
        agent, n_runs=50, num_particles=20, bounds=bounds, source=source
    )

    print_results("TEST 1: Fixed Swarm Size = 20", baseline_fixed, rl_fixed)

    # --------------------------------------------------------
    # TEST 2: Variable Swarm Size (100 runs)
    # --------------------------------------------------------
    print("Running Test 2: Variable Swarm Sizes (5–20)")

    baseline_var = {"Ts": [], "I": [], "SD": [], "Success": []}
    rl_var = {"Ts": [], "I": [], "SD": [], "Success": []}

    for run in range(100):
        N = random.randint(5, 20)

        base = run_standard_apso(1, N, bounds, source)
        rl = run_error_term_apso(agent, 1, N, bounds, source)

        for key in baseline_var:
            baseline_var[key].extend(base[key])
            rl_var[key].extend(rl[key])

    print_results("TEST 2: Variable Swarm Size (5–20)", baseline_var, rl_var)

    # -----------------------------
    # Structured evaluation: mean time vs N
    # -----------------------------

    # runs per N (increase for tighter error bars)
    runs_per_N = 100
    Ns = list(range(5, 21))

    mean_time_base = []
    std_time_base = []
    mean_time_rl = []
    std_time_rl = []

    print("\nStructured evaluation: computing mean Time vs number of particles...")

    for N in Ns:
        print(f"  Evaluating N = {N} (runs = {runs_per_N})...", end="", flush=True)

        # Baseline
        base_res = run_standard_apso(
            n_runs=runs_per_N,
            num_particles=N,
            bounds=bounds,
            source=source,
            max_iter=500,
        )
        base_times = np.array(base_res["Ts"], dtype=np.float32)
        mean_time_base.append(np.mean(base_times))
        std_time_base.append(np.std(base_times))

        # RL Error-Term APSO
        rl_res = run_error_term_apso(
            agent,
            n_runs=runs_per_N,
            num_particles=N,
            bounds=bounds,
            source=source,
            max_iter=500,
        )
        rl_times = np.array(rl_res["Ts"], dtype=np.float32)
        mean_time_rl.append(np.mean(rl_times))
        std_time_rl.append(np.std(rl_times))

        print(" done.")

    # Convert to arrays
    mean_time_base = np.array(mean_time_base)
    std_time_base = np.array(std_time_base)
    mean_time_rl = np.array(mean_time_rl)
    std_time_rl = np.array(std_time_rl)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        Ns,
        mean_time_base,
        yerr=std_time_base,
        fmt="-o",
        capsize=4,
        label="Standard APSO",
    )
    plt.errorbar(
        Ns,
        mean_time_rl,
        yerr=std_time_rl,
        fmt="-s",
        capsize=4,
        label="Error-Term RL-APSO",
    )
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.xlabel("Number of particles (N)")
    plt.ylabel("Average source-seeking time (s)")
    plt.title("Average Source-Seeking Time vs Swarm Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig("time_vs_particles.png", dpi=200)
    print("\nSaved plot to time_vs_particles.png")
    plt.show()
