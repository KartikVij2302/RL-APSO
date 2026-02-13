"""RL integration for the traditional SPSO source-seeking problem.

This script trains a PPO agent to adapt SPSO hyperparameters (c1, c2)
online while the swarm searches for a single source.

Run (from repo root):
	python SPSO/rl_spso.py --train

Optional evaluation:
	python SPSO/rl_spso.py --eval --model-path models/ppo_spso_c1c2.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# Make repo root importable even when running `python SPSO/rl_spso.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from apso_rl_agent.PPO import PPOAgent

# local import (works when file executed by path)
from spso import SPSO


@dataclass
class RLSPOSOConfig:
	side_length: float = 100.0
	n_particles: int = 10
	max_iter: int = 300
	speed: float = 10.0
	omega: float = 0.721
	c1_init: float = 1.193
	c2_init: float = 1.193
	termination_dist: float = 0.1

	# Action mapping: multiplicative delta per step.
	delta_frac: float = 0.2
	c_min: float = 0.05
	c_max: float = 5.0

	# Reward shaping (mirrors RLAPSOEnv, scaled for SPSO)
	alpha_time: float = 50.0
	beta_iter: float = 1.0
	gamma_close: float = 20.0
	success_bonus: float = 300.0
	timeout_penalty: float = -20.0
	proximity_decay: float = 2.0

	# Running reward normalization
	rn_beta: float = 0.999
	reward_clip: float = 200.0


class RLSPOSOEnv:
	"""Environment wrapper around `SPSO` where the agent controls (c1, c2)."""

	def __init__(self, cfg: RLSPOSOConfig, seed: int | None = None):
		self.cfg = cfg
		self.rng = np.random.default_rng(seed)

		self.spso: SPSO | None = None
		self.current_iter = 0

		self.prev_best_signal = 0.0

		# Reward normalization state
		self.reward_rmean = 0.0
		self.reward_rvar = 1.0

		# Logging buffers (optional analysis)
		self.step_time_cost_terms = []
		self.iteration_penalty_terms = []
		self.proximity_bonus_terms = []
		self.success_bonus_terms = []
		self.timeout_penalty_terms = []

	@property
	def map_diag(self) -> float:
		return float(np.sqrt(2.0) * self.cfg.side_length)

	def reset(self, source_pos: np.ndarray | None = None, n_particles: int | None = None) -> np.ndarray:
		if n_particles is None:
			n_particles = self.cfg.n_particles

		self.spso = SPSO(
			n_particles=int(n_particles),
			side_length=float(self.cfg.side_length),
			omega=float(self.cfg.omega),
			c1=float(self.cfg.c1_init),
			c2=float(self.cfg.c2_init),
			T=1.0,
			speed=float(self.cfg.speed),
		)

		# Override/randomize the source location (SPSO defaults to center)
		if source_pos is None:
			lo, hi = 0.0, self.cfg.side_length
			source_pos = self.rng.uniform(low=lo, high=hi, size=2)
		self.spso.source = np.array(source_pos, dtype=float)

		self.current_iter = 0
		self.prev_best_signal = float(-self.spso.global_best_signal)
		return self._get_state()

	def _get_state(self) -> np.ndarray:
		assert self.spso is not None

		# 7-D state:
		# [diversity, best_signal_change, time_left, avg_vel, c1_norm, c2_norm, n_particles_norm]
		positions = np.array([p.position for p in self.spso.particles])
		gbest = self.spso.global_best_position

		dists = np.linalg.norm(positions - gbest[None, :], axis=1)
		diversity = float(np.mean(dists)) if len(dists) else 0.0

		current_best_signal = float(-self.spso.global_best_signal)
		best_signal_change = current_best_signal - self.prev_best_signal

		time_left = 1.0 - (self.current_iter / max(1, self.cfg.max_iter))

		vels = np.array([np.linalg.norm(p.velocity) for p in self.spso.particles])
		avg_vel = float(np.mean(vels)) if len(vels) else 0.0

		c1 = float(self.spso.c1)
		c2 = float(self.spso.c2)
		c1_norm = float(np.clip(c1 / 5.0, 0.0, 1.0))
		c2_norm = float(np.clip(c2 / 5.0, 0.0, 1.0))

		# Map [5..30] -> [0..1] if you sweep swarm sizes; otherwise stable indicator.
		n_particles_norm = float(np.clip((self.spso.n - 5.0) / 25.0, 0.0, 1.0))

		return np.array(
			[diversity, best_signal_change, time_left, avg_vel, c1_norm, c2_norm, n_particles_norm],
			dtype=np.float32,
		)

	def _map_action_to_c1c2(self, action: np.ndarray) -> Tuple[float, float, bool, float]:
		"""Action is expected in [-1, 1]^2; returns (c1, c2, valid, penalty)."""
		assert self.spso is not None
		a = np.asarray(action, dtype=float).reshape(-1)
		if a.size != 2 or not np.all(np.isfinite(a)):
			return float(self.spso.c1), float(self.spso.c2), False, -8.0

		a = np.clip(a, -1.0, 1.0)
		c1_cur, c2_cur = float(self.spso.c1), float(self.spso.c2)
		c1 = c1_cur * (1.0 + self.cfg.delta_frac * float(a[0]))
		c2 = c2_cur * (1.0 + self.cfg.delta_frac * float(a[1]))

		valid = True
		penalty = 0.0
		if (not np.isfinite(c1)) or (not np.isfinite(c2)) or c1 <= 0.0 or c2 <= 0.0:
			valid = False
			penalty = -8.0
			c1, c2 = c1_cur, c2_cur

		c1 = float(np.clip(c1, self.cfg.c_min, self.cfg.c_max))
		c2 = float(np.clip(c2, self.cfg.c_min, self.cfg.c_max))
		return c1, c2, valid, penalty

	def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
		assert self.spso is not None

		# 1) Apply RL-controlled hyperparameters
		c1, c2, valid_params, invalid_penalty = self._map_action_to_c1c2(action)
		self.spso.c1 = c1
		self.spso.c2 = c2

		# 2) Run SPSO physics
		prev_pos = np.array([p.position.copy() for p in self.spso.particles])
		found = self.spso.step()
		curr_pos = np.array([p.position for p in self.spso.particles])

		# 3) Compute reward components (similar spirit to RLAPSOEnv)
		step_dist = float(np.sum(np.linalg.norm(curr_pos - prev_pos, axis=1)))
		mean_step_dist = step_dist / max(1, self.spso.n)
		step_time = mean_step_dist / max(1e-9, self.spso.speed)

		min_dist = float(
			min(np.linalg.norm(curr_pos - self.spso.source[None, :], axis=1))
		)
		min_dist_norm = min_dist / (self.map_diag + 1e-6)

		time_cost_term = -self.cfg.alpha_time * np.log1p(step_time)

		frac = self.current_iter / max(1, self.cfg.max_iter)
		iteration_term = -self.cfg.beta_iter * np.exp(frac)

		proximity_term = self.cfg.gamma_close * np.exp(-self.cfg.proximity_decay * min_dist_norm)

		reward = time_cost_term + iteration_term + proximity_term + invalid_penalty

		success_term = 0.0
		timeout_term = 0.0
		done = False
		if found:
			success_term = self.cfg.success_bonus
			reward += success_term
			done = True

		self.current_iter += 1
		if self.current_iter >= self.cfg.max_iter:
			done = True
			timeout_term = self.cfg.timeout_penalty
			reward += timeout_term

		# Logging
		self.step_time_cost_terms.append(float(time_cost_term))
		self.iteration_penalty_terms.append(float(iteration_term))
		self.proximity_bonus_terms.append(float(proximity_term))
		self.success_bonus_terms.append(float(success_term))
		self.timeout_penalty_terms.append(float(timeout_term))

		# 4) Normalize + clip reward (same style as RLAPSOEnv)
		old_mean = self.reward_rmean
		self.reward_rmean = self.cfg.rn_beta * self.reward_rmean + (1.0 - self.cfg.rn_beta) * float(reward)
		self.reward_rvar = self.cfg.rn_beta * self.reward_rvar + (1.0 - self.cfg.rn_beta) * (float(reward) - old_mean) ** 2
		r_std = float(np.sqrt(self.reward_rvar) + 1e-6)
		reward_norm = float(np.clip((float(reward) - self.reward_rmean) / r_std, -self.cfg.reward_clip, self.cfg.reward_clip))

		# update state trackers
		current_best_signal = float(-self.spso.global_best_signal)
		self.prev_best_signal = current_best_signal

		info = {
			"found": bool(found),
			"min_dist": float(min_dist),
			"c1": float(self.spso.c1),
			"c2": float(self.spso.c2),
			"valid_params": bool(valid_params),
			"reward_raw": float(reward),
		}
		return self._get_state(), reward_norm, done, info


def train(cfg: RLSPOSOConfig, episodes: int, model_path: str, seed: int | None = 0) -> None:
	env = RLSPOSOEnv(cfg, seed=seed)

	state_dim = 7
	action_dim = 2
	agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-4)

	os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

	for ep in range(1, episodes + 1):
		state = env.reset()
		ep_return = 0.0
		done = False

		for _ in range(cfg.max_iter):
			action, logprob = agent.select_action(state)
			next_state, reward, done, _info = env.step(action)
			agent.store(state, action, logprob, reward, done)
			ep_return += float(reward)
			state = next_state
			if done:
				break

		agent.update()

		if ep % 10 == 0:
			print(f"[train] episode={ep:5d}  return={ep_return:9.3f}  iters={env.current_iter:4d}  last_c1={env.spso.c1:.3f} last_c2={env.spso.c2:.3f}")

		if ep % 100 == 0:
			agent.save(model_path)
			print(f"[train] saved checkpoint -> {model_path}")

	agent.save(model_path)
	print(f"[train] done, saved -> {model_path}")


def evaluate(cfg: RLSPOSOConfig, model_path: str, episodes: int = 30, seed: int | None = 1) -> None:
	env = RLSPOSOEnv(cfg, seed=seed)
	agent = PPOAgent(state_dim=7, action_dim=2, lr=1e-4)
	agent.load(model_path)

	times = []
	iters = []
	founds = 0

	for _ in range(episodes):
		state = env.reset()
		for _ in range(cfg.max_iter):
			action, _lp = agent.select_action(state)
			state, _r, done, info = env.step(action)
			if done:
				if info.get("found", False):
					founds += 1
				break

		# episode time proxy: swarm distance / speed
		swarm_distance = float(sum(p.dist_travelled for p in env.spso.particles))
		times.append(swarm_distance / max(1e-9, cfg.speed))
		iters.append(env.current_iter)

	print(
		f"[eval] episodes={episodes} found={founds}/{episodes} "
		f"mean_time={np.mean(times):.3f}s mean_iters={np.mean(iters):.2f}"
	)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument("--train", action="store_true", help="Train PPO to control SPSO c1/c2")
	p.add_argument("--eval", action="store_true", help="Evaluate a trained PPO controller")
	p.add_argument("--episodes", type=int, default=12000)
	p.add_argument("--max-iter", type=int, default=300)
	p.add_argument("--n-particles", type=int, default=10)
	p.add_argument("--side-length", type=float, default=100.0)
	p.add_argument("--model-path", type=str, default="models/ppo_spso_c1c2.pth")
	p.add_argument("--seed", type=int, default=0)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	cfg = RLSPOSOConfig(
		side_length=float(args.side_length),
		n_particles=int(args.n_particles),
		max_iter=int(args.max_iter),
	)

	if args.train == args.eval:
		raise SystemExit("Pass exactly one of --train or --eval")

	if args.train:
		train(cfg=cfg, episodes=int(args.episodes), model_path=str(args.model_path), seed=int(args.seed))
	else:
		evaluate(cfg=cfg, model_path=str(args.model_path), episodes=max(1, int(args.episodes) // 10), seed=int(args.seed))


if __name__ == "__main__":
	main()
