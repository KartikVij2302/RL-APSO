import os
from apso_rl_agent.apso import APSO_FunctionOptimizer, validate_apso_params
from apso_rl_agent.rl_enhanced_apso import set_global_seed
import numpy as np
import matplotlib.pyplot as plt


current_dir = os.path.dirname(os.path.abspath(__file__))

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

def ackley(x: np.ndarray) -> float:
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return float(term1 + term2 + a + np.exp(1.0))

def griewank(x: np.ndarray) -> float:
    sum_sq = np.sum(x ** 2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(1.0 + sum_sq / 4000.0 - prod_cos)


def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2.0 * np.pi * x)))


BENCHMARKS = {
    "Sphere": (sphere, (-100.0, 100.0)),
    "Rosenbrock": (rosenbrock, (-30.0, 30.0)),
    "Ackley": (ackley, (-32.0, 32.0)),
    "Griewank": (griewank, (-600.0, 600.0)),
    "Rastrigin": (rastrigin, (-5.12, 5.12)),
}


def _map_action_to_params_for_function(apso, action: np.ndarray):
    """Map PPO action in [-1,1]^4 to APSO parameters (evaluation).

    This mirrors the mapping used in RLAPSOEnv._map_action_to_params so
    the trained policy sees a consistent control interface.
    """

    delta_frac = 0.2
    a = np.clip(action, -1.0, 1.0)

    w1_cur = getattr(apso, "w1", 0.675)
    w2_cur = getattr(apso, "w2", -0.285)
    c1_cur = getattr(apso, "c1", 1.193)
    c2_cur = getattr(apso, "c2", 1.193)

    w1 = w1_cur * (1.0 + delta_frac * a[0])
    w2 = w2_cur * (1.0 + delta_frac * a[1])
    c1 = c1_cur * (1.0 + delta_frac * a[2])
    c2 = c2_cur * (1.0 + delta_frac * a[3])

    return float(w1), float(w2), float(c1), float(c2)


def evaluate_rl_apso_on_function(
    agent: PPOAgent,
    func,
    bounds,
    dim: int = 30,
    num_particles: int = 20,
    max_iter: int = 300,
    runs: int = 30,
    termination_tol: float = 1e-8,
):
    """Evaluate a trained RL-APSO controller on a given test function.

    The controller adjusts APSO parameters (w1, w2, c1, c2) while
    APSO_FunctionOptimizer performs the actual search to minimise
    ``func`` over the hyper-rectangle defined by ``bounds``.
    """
    lo, hi = bounds
    lo_vec = np.full(dim, lo, dtype=float)
    hi_vec = np.full(dim, hi, dtype=float)

    best_values = []
    iterations = []
    histories = []  # per-run convergence curves (best value vs iteration)

    for _ in range(runs):
        apso = APSO_FunctionOptimizer(
            objective=func,
            bounds=(lo_vec, hi_vec),
            num_particles=num_particles,
            w1=0.675,
            w2=-0.285,
            c1=1.193,
            c2=1.193,
            T=1.0,
            termination_tol=termination_tol,
        )

        prev_signal = -apso.gbest_value
        run_history = []

        for it in range(max_iter):
            # Construct state in the same format used during training
            current_signal = -apso.gbest_value
            signal_change = current_signal - prev_signal

            time_left = 1.0 - (it / max_iter)
            avg_vel = np.mean([np.linalg.norm(p.v) for p in apso.particles])

            w1 = getattr(apso, "w1", 0.0)
            w2 = getattr(apso, "w2", 0.0)
            c1 = getattr(apso, "c1", 1.0)
            c2 = getattr(apso, "c2", 1.0)

            num_particles_n = (num_particles - 5.0) / 25.0
            # maps 5->0, 30->1

            state = np.array(
                [
                    signal_change,
                    time_left,
                    avg_vel,
                    np.clip(w1 / 2.0, -1.0, 1.0),
                    np.clip(w2 / 2.0, -1.0, 1.0),
                    np.clip(c1 / 5.0, 0.0, 1.0),
                    np.clip(c2 / 5.0, 0.0, 1.0),
                    num_particles_n,
                ],
                dtype=np.float32,
            )

            action, _ = agent.select_action(state)

            # Map action to APSO parameters and validate stability
            w1_new, w2_new, c1_new, c2_new = _map_action_to_params_for_function(apso, action)
            try:
                validate_apso_params(w1_new, w2_new, c1_new, c2_new, getattr(apso, "T", 1.0))
                apso.w1 = w1_new
                apso.w2 = w2_new
                apso.c1 = c1_new
                apso.c2 = c2_new
            except Exception:
                # If invalid, fall back to fixed stable parameters
                apso.w1 = 0.675
                apso.w2 = -0.285
                apso.c1 = 1.193
                apso.c2 = 1.193

            found, gbest_val = apso.step()
            prev_signal = -gbest_val

            run_history.append(gbest_val)

            if found:
                break

        # Pad history so all runs have length max_iter
        if len(run_history) < max_iter:
            run_history.extend([run_history[-1]] * (max_iter - len(run_history)))

        best_values.append(apso.gbest_value)
        iterations.append(len(run_history))
        histories.append(run_history)

    return {
        "best_values": np.array(best_values),
        "iterations": np.array(iterations),
        "histories": np.array(histories),  # shape: (runs, max_iter)
    }


def evaluate_fixed_apso_on_function(
    func,
    bounds,
    dim: int = 30,
    num_particles: int = 20,
    max_iter: int = 300,
    runs: int = 30,
    termination_tol: float = 1e-8,
):
    """Evaluate fixed-parameter APSO (no RL) on a test function.

    Uses the same APSO_FunctionOptimizer but keeps (w1, w2, c1, c2)
    fixed at the stable reference values for all iterations.
    Returns per-run final best values, iterations, and convergence
    histories suitable for plotting.
    """

    lo, hi = bounds
    lo_vec = np.full(dim, lo, dtype=float)
    hi_vec = np.full(dim, hi, dtype=float)

    best_values = []
    iterations = []
    histories = []

    for _ in range(runs):
        apso = APSO_FunctionOptimizer(
            objective=func,
            bounds=(lo_vec, hi_vec),
            num_particles=num_particles,
            w1=0.675,
            w2=-0.285,
            c1=1.193,
            c2=1.193,
            T=1.0,
            termination_tol=termination_tol,
        )

        run_history = []

        for it in range(max_iter):
            found, gbest_val = apso.step()
            run_history.append(gbest_val)

            if found:
                break

        if len(run_history) < max_iter:
            run_history.extend([run_history[-1]] * (max_iter - len(run_history)))

        best_values.append(apso.gbest_value)
        iterations.append(len(run_history))
        histories.append(run_history)

    return {
        "best_values": np.array(best_values),
        "iterations": np.array(iterations),
        "histories": np.array(histories),
    }


def evaluate_trained_rl_apso_on_benchmarks(
    runs_per_function: int = 30,
    dim: int = 30,
    num_particles: int = 20,
    max_iter: int = 300,
    termination_tol: float = 1e-8,
):
    """Evaluate a trained PPO-based RL-APSO on standard benchmarks.

    For each function in BENCHMARKS this reports the mean / std of the
    best function value found and the average number of iterations
    required over ``runs_per_function`` independent runs.
    """
    set_global_seed(42)
    state_dim = 8
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim, lr=3e-4)

    model_path = os.path.join(current_dir, "ppo_apso.pth")
    if not os.path.exists(model_path):
        print(f"[Error] Trained PPO model not found at {model_path}. Run training first.")
        return

    agent.load(model_path)
    print(f"Loaded trained PPO agent from {model_path}")

    print("\n=== RL-Enhanced APSO Benchmarking ===")
    print(f"Dimensionality: {dim}, Particles: {num_particles}, Max iter: {max_iter}")
    print(f"Runs per function: {runs_per_function}\n")

    for name, (func, (lo, hi)) in BENCHMARKS.items():
        print(f"--- {name} ---")

        # RL-guided APSO
        rl_stats = evaluate_rl_apso_on_function(
            agent,
            func,
            (lo, hi),
            dim=dim,
            num_particles=num_particles,
            max_iter=max_iter,
            runs=runs_per_function,
            termination_tol=termination_tol,
        )

        # Fixed-parameter APSO baseline
        fixed_stats = evaluate_fixed_apso_on_function(
            func,
            (lo, hi),
            dim=dim,
            num_particles=num_particles,
            max_iter=max_iter,
            runs=runs_per_function,
            termination_tol=termination_tol,
        )

        rl_vals = rl_stats["best_values"]
        rl_iters = rl_stats["iterations"]
        fixed_vals = fixed_stats["best_values"]
        fixed_iters = fixed_stats["iterations"]

        print(f"  Bounds: [{lo}, {hi}]")
        print(f"  RL-APSO   mean best f(x): {rl_vals.mean():.6e} ± {rl_vals.std():.2e}")
        print(f"  Fixed APSO mean best f(x): {fixed_vals.mean():.6e} ± {fixed_vals.std():.2e}")
        print(f"  RL-APSO   mean iterations: {rl_iters.mean():.2f}")
        print(f"  Fixed APSO mean iterations: {fixed_iters.mean():.2f}")

        # Convergence plots: function value vs number of iterations
        rl_hist_mean = rl_stats["histories"].mean(axis=0)
        fixed_hist_mean = fixed_stats["histories"].mean(axis=0)

        plt.figure(figsize=(8, 5))
        plt.semilogy(rl_hist_mean, label="RL-APSO")
        plt.semilogy(fixed_hist_mean, label="Fixed APSO", linestyle="--")
        plt.xlabel("Iteration")
        plt.ylabel("Best f(x) (log scale)")
        plt.title(f"Convergence on {name}")
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.legend()
        fname = f"convergence_{name.lower()}.png".replace(" ", "_")
        plt.tight_layout()
        plt.savefig(fname)
        print(f"  Saved convergence plot to {fname}\n")