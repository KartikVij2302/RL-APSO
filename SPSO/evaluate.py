"""Evaluate RL-modified SPSO vs baseline SPSO.

Metrics (Monte Carlo averages over N runs):
  - Average source seeking time μ(Ts): time from start until the source is located
	by one or more UAVs.
  - Average number of iterations μ(I): average number of waypoints/iterations
	generated until locating the source.

This script compares:
  1) Baseline SPSO with fixed (c1, c2)
  2) RL-modified SPSO where a PPO agent adapts (c1, c2) online each iteration

Run (from repo root):
  python SPSO/evaluate.py --model-path models/ppo_spso_c1c2.pth
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


# Make repo root importable even when running `python SPSO/evaluate.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from apso_rl_agent.PPO import PPOAgent

try:
	# when executed as a module: python -m SPSO.evaluate
	from .spso import SPSO
except Exception:  # pragma: no cover
	# when executed as a script: python SPSO/evaluate.py
	from spso import SPSO


@dataclass
class EvalConfig:
	side_length: float = 100.0
	n_particles: int = 10
	max_iter: int = 300
	T: float = 1.0
	speed: float = 10.0
	omega: float = 0.721

	# baseline parameters
	c1_baseline: float = 1.193
	c2_baseline: float = 1.193

	# RL action mapping (must match training)
	delta_frac: float = 0.2
	c_min: float = 0.05
	c_max: float = 5.0

	termination_dist: float = 0.1


def _policy_action(agent: PPOAgent, state: np.ndarray, deterministic: bool) -> np.ndarray:
	"""Return action in [-1,1]^2. Deterministic uses actor mean; otherwise samples."""
	if deterministic:
		with torch.no_grad():
			s = torch.as_tensor(state, dtype=torch.float32)
			a = agent.policy_old.actor(s)
		return a.cpu().numpy()

	action, _logprob = agent.select_action(state)
	return np.asarray(action, dtype=np.float32)


def _spso_state(spso: SPSO, prev_best_signal: float, current_iter: int, max_iter: int) -> np.ndarray:
	"""State matches RLSPOSOEnv._get_state (7-D)."""
	positions = np.array([p.position for p in spso.particles])
	gbest = spso.global_best_position

	dists = np.linalg.norm(positions - gbest[None, :], axis=1)
	diversity = float(np.mean(dists)) if len(dists) else 0.0

	current_best_signal = float(-spso.global_best_signal)
	best_signal_change = current_best_signal - float(prev_best_signal)

	time_left = 1.0 - (current_iter / max(1, max_iter))

	vels = np.array([np.linalg.norm(p.velocity) for p in spso.particles])
	avg_vel = float(np.mean(vels)) if len(vels) else 0.0

	c1_norm = float(np.clip(float(spso.c1) / 5.0, 0.0, 1.0))
	c2_norm = float(np.clip(float(spso.c2) / 5.0, 0.0, 1.0))
	n_particles_norm = float(np.clip((spso.n - 5.0) / 25.0, 0.0, 1.0))

	return np.array(
		[diversity, best_signal_change, time_left, avg_vel, c1_norm, c2_norm, n_particles_norm],
		dtype=np.float32,
	)


def _apply_rl_c1c2(cfg: EvalConfig, spso: SPSO, action: np.ndarray) -> Tuple[float, float]:
	a = np.asarray(action, dtype=float).reshape(-1)
	a = np.clip(a, -1.0, 1.0)

	c1_cur, c2_cur = float(spso.c1), float(spso.c2)
	c1 = c1_cur * (1.0 + cfg.delta_frac * float(a[0]))
	c2 = c2_cur * (1.0 + cfg.delta_frac * float(a[1]))

	if (not np.isfinite(c1)) or (not np.isfinite(c2)) or c1 <= 0.0 or c2 <= 0.0:
		c1, c2 = c1_cur, c2_cur

	c1 = float(np.clip(c1, cfg.c_min, cfg.c_max))
	c2 = float(np.clip(c2, cfg.c_min, cfg.c_max))

	spso.c1 = c1
	spso.c2 = c2
	return c1, c2


def run_baseline(cfg: EvalConfig, sources: np.ndarray, seed: int) -> List[Dict]:
	rows: List[Dict] = []
	for run_idx, source in enumerate(sources):
		np.random.seed(seed + run_idx)
		spso = SPSO(
			n_particles=cfg.n_particles,
			side_length=cfg.side_length,
			omega=cfg.omega,
			c1=cfg.c1_baseline,
			c2=cfg.c2_baseline,
			T=cfg.T,
			speed=cfg.speed,
		)
		spso.source = np.array(source, dtype=float)

		found = False
		iterations_used = cfg.max_iter
		for k in range(1, cfg.max_iter + 1):
			if spso.step():
				found = True
				iterations_used = k
				break

		# Source seeking time definition: time duration until first detection.
		# With constant sampling interval T, Ts = I * T.
		Ts = float(iterations_used) * float(cfg.T)

		rows.append(
			{
				"run": int(run_idx),
				"method": "SPSO",
				"Ts": Ts,
				"I": int(iterations_used),
				"success": int(found),
				"source_x": float(source[0]),
				"source_y": float(source[1]),
				"n_particles": int(cfg.n_particles),
			}
		)
	return rows


def run_rl_modified(cfg: EvalConfig, sources: np.ndarray, model_path: str, seed: int, deterministic: bool) -> List[Dict]:
	agent = PPOAgent(state_dim=7, action_dim=2, lr=3e-4)
	agent.load(model_path)

	rows: List[Dict] = []
	for run_idx, source in enumerate(sources):
		np.random.seed(seed + run_idx)
		torch.manual_seed(seed + run_idx)

		spso = SPSO(
			n_particles=cfg.n_particles,
			side_length=cfg.side_length,
			omega=cfg.omega,
			c1=cfg.c1_baseline,
			c2=cfg.c2_baseline,
			T=cfg.T,
			speed=cfg.speed,
		)
		spso.source = np.array(source, dtype=float)

		prev_best_signal = float(-spso.global_best_signal)
		found = False
		iterations_used = cfg.max_iter

		for t in range(cfg.max_iter):
			state = _spso_state(spso, prev_best_signal=prev_best_signal, current_iter=t, max_iter=cfg.max_iter)
			action = _policy_action(agent, state, deterministic=deterministic)
			_apply_rl_c1c2(cfg, spso, action)

			if spso.step():
				found = True
				iterations_used = t + 1
				break

			prev_best_signal = float(-spso.global_best_signal)

		Ts = float(iterations_used) * float(cfg.T)
		rows.append(
			{
				"run": int(run_idx),
				"method": "RL-SPSO",
				"Ts": Ts,
				"I": int(iterations_used),
				"success": int(found),
				"source_x": float(source[0]),
				"source_y": float(source[1]),
				"n_particles": int(cfg.n_particles),
			}
		)
	return rows


def summarize(rows: List[Dict], method: str) -> Dict:
	r = [x for x in rows if x["method"] == method]
	Ts = np.array([x["Ts"] for x in r], dtype=float)
	I = np.array([x["I"] for x in r], dtype=float)
	success = np.array([x["success"] for x in r], dtype=int)

	# Monte Carlo mean definitions
	out = {
		"method": method,
		"mu_Ts": float(np.mean(Ts)) if len(Ts) else float("nan"),
		"mu_I": float(np.mean(I)) if len(I) else float("nan"),
		"success_rate": float(np.mean(success)) if len(success) else float("nan"),
	}
	return out


def write_csv(path: str, rows: List[Dict]) -> None:
	if not rows:
		return
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	fieldnames = list(rows[0].keys())
	with open(path, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for row in rows:
			w.writerow(row)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument("--model-path", type=str, required=True, help="Path to trained RL-SPSO PPO model")
	p.add_argument("--n-runs", type=int, default=100)
	p.add_argument("--max-iter", type=int, default=300)
	p.add_argument("--n-particles", type=int, default=10)
	p.add_argument("--side-length", type=float, default=100.0)
	p.add_argument("--seed", type=int, default=42)

	p.add_argument("--fixed-source", nargs=2, type=float, default=[50.0, 50.0])
	p.add_argument("--random-source", action="store_true", help="Sample a random source per run uniformly in the grid")
	p.add_argument("--deterministic", action="store_true", help="Use policy mean action (no sampling)")

	p.add_argument("--out-csv", type=str, default="results/spso_vs_rlspso_eval.csv")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	cfg = EvalConfig(
		side_length=float(args.side_length),
		n_particles=int(args.n_particles),
		max_iter=int(args.max_iter),
	)

	rng = np.random.default_rng(int(args.seed))
	if args.random_source:
		sources = rng.uniform(low=0.0, high=cfg.side_length, size=(int(args.n_runs), 2))
	else:
		src = np.array([float(args.fixed_source[0]), float(args.fixed_source[1])], dtype=float)
		sources = np.repeat(src[None, :], repeats=int(args.n_runs), axis=0)

	baseline_rows = run_baseline(cfg, sources=sources, seed=int(args.seed))
	rl_rows = run_rl_modified(
		cfg,
		sources=sources,
		model_path=str(args.model_path),
		seed=int(args.seed),
		deterministic=bool(args.deterministic),
	)
	all_rows = baseline_rows + rl_rows
	write_csv(str(args.out_csv), all_rows)

	s_baseline = summarize(all_rows, "SPSO")
	s_rl = summarize(all_rows, "RL-SPSO")

	print("\n=== SPSO vs RL-SPSO Evaluation ===")
	print(f"runs N = {int(args.n_runs)} | grid = {cfg.side_length:.1f}x{cfg.side_length:.1f} | swarm = {cfg.n_particles} | max_iter = {cfg.max_iter}")
	if args.random_source:
		print("source: random per run")
	else:
		print(f"source: fixed at ({args.fixed_source[0]:.2f}, {args.fixed_source[1]:.2f})")
	print(f"policy: {'deterministic' if args.deterministic else 'stochastic'}")
	print(f"saved: {args.out_csv}")

	print("\nMetric definitions:")
	print("  μ(Ts) = (1/N) * Σ Ts_i, Ts_i = I_i * T")
	print("  μ(I)  = (1/N) * Σ I_i")

	print("\nResults:")
	print(f"  SPSO     | μ(Ts)={s_baseline['mu_Ts']:.3f}s | μ(I)={s_baseline['mu_I']:.2f} | success={s_baseline['success_rate']*100:.1f}%")
	print(f"  RL-SPSO  | μ(Ts)={s_rl['mu_Ts']:.3f}s | μ(I)={s_rl['mu_I']:.2f} | success={s_rl['success_rate']*100:.1f}%")


if __name__ == "__main__":
	main()

