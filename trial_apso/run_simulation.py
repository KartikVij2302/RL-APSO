# ---- Monte Carlo driver and analysis ----
import os
from .apso import APSO
from .spso import SPSO
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.01            # signal attenuation factor
SOURCE_POWER = 100.0    # signal amplitude at source
SPEED = 10.0            # constant UAV speed in m/s (as required)
T = 1.0                 # sampling interval (s); displacement per iter = SPEED * T
TERMINATION_DIST = 0.1  # termination threshold (m)

def monte_carlo_experiments_apso(n_runs: int = 50,
                            n_drones: int = 5,
                            side_length: float = 100.0,
                            max_iters: int = 300,
                            w1: float = 0.675,
                            w2: float = -0.285,
                            c1: float = 1.193,
                            c2: float = 1.193,
                            T_sample: float = T,
                            speed: float = SPEED):
    seeking_times: list[float] = []
    iterations_counts: list[int] = []
    all_score_histories: list[np.ndarray] = []
    success_count = 0

    for seed in range(n_runs):
        np.random.seed(seed)
        apso = APSO(n_drones=n_drones, side_length=side_length, w1=w1, w2=w2,
                    c1=c1, c2=c2, T_sample=T_sample, speed=speed, objective='source')
        iters, sim_time = apso.run(max_iterations=max_iters)
        seeking_times.append(sim_time)    # T_si (simulated)
        iterations_counts.append(iters)   # W_si
        all_score_histories.append(np.array(apso.score_history))
        if len(apso.min_distances) > 0 and apso.min_distances[-1] <= TERMINATION_DIST:
            success_count += 1

        print(f"Run {seed+1}/{n_runs}: iters={iters}, sim_time={sim_time:.3f}s, best_signal={(-apso.global_best_signal):.6e}")

    # Metrics (equations 22, 23)
    mu_Ts = float(np.mean(seeking_times))
    mu_I = float(np.mean(iterations_counts))
    success_rate = success_count / float(n_runs)

    print("\n=== Monte Carlo Summary ===")
    print(f"Runs: {n_runs}, Drones: {n_drones}, Side: {side_length} m, Max iters: {max_iters}")
    print(f"Average source seeking time µ(Ts): {mu_Ts:.6f} s")
    print(f"Average number of iterations µ(I): {mu_I:.3f}")
    print(f"Success rate: {success_rate*100.0:.1f}%")

    # --- Plot 1: Distribution of seeking times ---
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.hist(seeking_times, bins=15, edgecolor='black', alpha=0.8)
    plt.axvline(mu_Ts, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {mu_Ts:.3f}s')
    plt.xlabel('Source Seeking Time (simulated seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Source Seeking Time (N={n_runs})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/seeking_time_hist.png')
    plt.close()

    # --- Plot 2: Distribution of iterations ---
    max_count = max(iterations_counts) if iterations_counts else 1
    bin_width = max(1, max(1, max_count // 15))
    plt.figure(figsize=(8,6))
    plt.hist(iterations_counts, bins=range(0, max_iters + 10, bin_width), edgecolor='black', alpha=0.8)
    plt.axvline(mu_I, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {mu_I:.2f}')
    plt.xlabel('Number of Iterations (Waypoints) to Find Source')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Iterations (N={n_runs})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/iterations_hist.png')
    plt.close()

    # --- Plot 3: Mean convergence curve (global best over iterations) ---
    if any(len(h) > 0 for h in all_score_histories):
        max_len = max(len(h) for h in all_score_histories)
        padded = np.array([np.pad(h, (0, max_len - len(h)), 'edge') for h in all_score_histories])
        mean_curve = padded.mean(axis=0)
        std_curve = padded.std(axis=0)
        iters = np.arange(1, max_len + 1)
        plt.figure(figsize=(10,6))
        lower = np.maximum(mean_curve - std_curve, 1e-12)
        upper = mean_curve + std_curve
        plt.fill_between(iters, lower, upper, alpha=0.2, label='±1 std')
        plt.plot(iters, mean_curve, '-', linewidth=2, label='Mean global best (signal)')
        if np.all(mean_curve > 0):
            plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Best Signal (mean)')
        plt.title('Mean Convergence Curve (mean ± std) over Monte Carlo runs')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/mean_convergence.png')
        plt.close()
    else:
        print("No score histories to plot.")

    return {
        'seeking_times': seeking_times,
        'iterations': iterations_counts,
        'mu_Ts': mu_Ts,
        'mu_I': mu_I,
        'success_rate': success_rate
    }


def monte_carlo_experiments_spso(n_runs: int = 50,
                                 n_particles: int = 5,
                                 side_length: float = 100.0,
                                 max_iters: int = 300,
                                 omega: float = 0.721,
                                 c1: float = 1.193,
                                 c2: float = 1.193,
                                 T: float = T,
                                 speed: float = SPEED):
    seeking_times: list[float] = []
    iterations_counts: list[int] = []
    all_score_histories: list[np.ndarray] = []
    success_count = 0

    for seed in range(n_runs):
        np.random.seed(seed)
        spso = SPSO(n_particles=n_particles, side_length=side_length, omega=omega,
                    c1=c1, c2=c2, T=T, speed=speed)
        iters, sim_time = spso.run(max_iterations=max_iters)
        seeking_times.append(sim_time)
        iterations_counts.append(iters)
        all_score_histories.append(np.array(spso.score_history))
        if len(spso.min_distances) > 0 and spso.min_distances[-1] <= TERMINATION_DIST:
            success_count += 1

        print(f"Run {seed+1}/{n_runs}: iters={iters}, sim_time={sim_time:.3f}s, best_signal={(-spso.global_best_signal):.6e}")

    # Metrics
    mu_Ts = float(np.mean(seeking_times))
    mu_I = float(np.mean(iterations_counts))
    success_rate = success_count / float(n_runs)

    print("\n=== SPSO Monte Carlo Summary ===")
    print(f"Runs: {n_runs}, Particles: {n_particles}, Side: {side_length} m, Max iters: {max_iters}")
    print(f"Average source seeking time µ(Ts): {mu_Ts:.6f} s")
    print(f"Average number of iterations µ(I): {mu_I:.3f}")
    print(f"Success rate: {success_rate*100.0:.1f}%")

    # --- Plot 1: Distribution of seeking times ---
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.hist(seeking_times, bins=15, edgecolor='black', alpha=0.8)
    plt.axvline(mu_Ts, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {mu_Ts:.3f}s')
    plt.xlabel('Source Seeking Time (simulated seconds)')
    plt.ylabel('Frequency')
    plt.title(f'SPSO: Distribution of Source Seeking Time (N={n_runs})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/spso_seeking_time_hist.png')
    plt.close()

    # --- Plot 2: Distribution of iterations ---
    max_count = max(iterations_counts) if iterations_counts else 1
    bin_width = max(1, max(1, max_count // 15))
    plt.figure(figsize=(8,6))
    plt.hist(iterations_counts, bins=range(0, max_iters + 10, bin_width), edgecolor='black', alpha=0.8)
    plt.axvline(mu_I, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {mu_I:.2f}')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Frequency')
    plt.title(f'SPSO: Distribution of Iterations (N={n_runs})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/spso_iterations_hist.png')
    plt.close()

    # --- Plot 3: Mean convergence curve ---
    if any(len(h) > 0 for h in all_score_histories):
        max_len = max(len(h) for h in all_score_histories)
        padded = np.array([np.pad(h, (0, max_len - len(h)), 'edge') for h in all_score_histories])
        mean_curve = padded.mean(axis=0)
        std_curve = padded.std(axis=0)
        iters = np.arange(1, max_len + 1)
        plt.figure(figsize=(10,6))
        lower = np.maximum(mean_curve - std_curve, 1e-12)
        upper = mean_curve + std_curve
        plt.fill_between(iters, lower, upper, alpha=0.2, label='±1 std')
        plt.plot(iters, mean_curve, '-', linewidth=2, label='Mean global best (signal)')
        if np.all(mean_curve > 0):
            plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Best Signal (mean)')
        plt.title('SPSO: Mean Convergence Curve (mean ± std)')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/spso_mean_convergence.png')
        plt.close()
    else:
        print("No SPSO score histories to plot.")

    return {
        'seeking_times': seeking_times,
        'iterations': iterations_counts,
        'mu_Ts': mu_Ts,
        'mu_I': mu_I,
        'success_rate': success_rate
    }
       
    
