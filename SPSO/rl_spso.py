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
import torch
import torch.nn as nn
import torch.optim as optim


# Make repo root importable even when running `python SPSO/rl_spso.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from apso_rl_agent.PPO import PPOAgent

# local import (works when file executed by path)
from spso import SPSO


def compare_standard_vs_rl(
	cfg: RLSPSOConfig,
	model_path: str,
	episodes: int = 100,
	seed: int | None = 0,
	algo: str = "ppo",
	swarm_size_low: int = 5,
	swarm_size_high: int = 31,
) -> Dict[str, float]:
	"""Compare standard SPSO vs RL-enhanced SPSO.

	Computes averages over `episodes` for:
	- source seeking time
	- iterations used
	- total swarm distance travelled

	Results are stratified by swarm size N.
	
	For each N in [swarm_size_low, swarm_size_high-1], runs `episodes` trials and reports
	per-N mean ± std for:
	- source seeking time
	- iterations used
	- total swarm distance travelled

	Returns a dict with both baselines' means.
	"""
	algo = str(algo).lower()
	if not os.path.exists(model_path):
		raise SystemExit(f"Model file not found: {model_path}")
	if int(swarm_size_low) >= int(swarm_size_high):
		raise ValueError("swarm_size_low must be < swarm_size_high")

	# Per-N buffers
	std_times_by_n: dict[int, list[float]] = {n: [] for n in range(int(swarm_size_low), int(swarm_size_high))}
	std_iters_by_n: dict[int, list[float]] = {n: [] for n in range(int(swarm_size_low), int(swarm_size_high))}
	std_swarm_dists_by_n: dict[int, list[float]] = {n: [] for n in range(int(swarm_size_low), int(swarm_size_high))}
	rl_times_by_n: dict[int, list[float]] = {n: [] for n in range(int(swarm_size_low), int(swarm_size_high))}
	rl_iters_by_n: dict[int, list[float]] = {n: [] for n in range(int(swarm_size_low), int(swarm_size_high))}
	rl_swarm_dists_by_n: dict[int, list[float]] = {n: [] for n in range(int(swarm_size_low), int(swarm_size_high))}

	# Build RL agent + env
	env = RLSPSOEnv(cfg, seed=seed)
	if algo == "ppo":
		agent: object = PPOAgent(state_dim=7, action_dim=2, lr=3e-4)
	elif algo == "ddpg":
		agent = DDPGAgent(state_dim=7, action_dim=2, seed=int(seed or 0))
	elif algo == "td3":
		agent = TD3Agent(state_dim=7, action_dim=2, seed=int(seed or 0))
	else:
		raise ValueError(f"Unknown algo '{algo}'. Use ppo, ddpg, or td3.")
	agent.load(model_path)  # type: ignore[attr-defined]

	base_seed = int(seed or 0)
	runs_per_n = int(episodes)
	for n_particles in range(int(swarm_size_low), int(swarm_size_high)):
		for run_idx in range(runs_per_n):
			# SPSO uses global numpy RNG; seed it per-run for repeatability.
			# (and so standard and RL runs share the same initial boundary placement).
			ep_seed = base_seed + (n_particles * 10_000) + run_idx
			np.random.seed(ep_seed)

			# --- Standard SPSO run ---
			spso = SPSO(
				n_particles=int(n_particles),
				side_length=float(cfg.side_length),
				omega=float(cfg.omega),
				c1=float(cfg.c1_init),
				c2=float(cfg.c2_init),
				T=1.0,
				speed=float(cfg.speed),
			)
			t, it_used, swarm_dist, found = spso.run(max_iterations=int(cfg.max_iter))
			if found:
				std_times_by_n[n_particles].append(float(t))
				std_iters_by_n[n_particles].append(float(it_used))
				std_swarm_dists_by_n[n_particles].append(float(swarm_dist))

			# --- RL-enhanced SPSO run ---
			np.random.seed(ep_seed)
			state = env.reset(n_particles=int(n_particles))
			info: Dict = {}
			for _ in range(int(cfg.max_iter)):
				if algo == "ppo":
					action, _lp = agent.select_action(state)  # type: ignore[attr-defined]
				else:
					action = agent.select_action(state, noise=False)  # type: ignore[attr-defined]
				state, _r, done, info = env.step(action)
				if done:
					break

			assert env.spso is not None
			swarm_distance = float(sum(p.dist_travelled for p in env.spso.particles))
			found = bool(info.get("found", False))
			if found:
				finder = min(env.spso.particles, key=lambda p: np.linalg.norm(p.position - env.spso.source))
				time_to_find = float(finder.dist_travelled) / max(1e-9, float(cfg.speed))
			else:
				time_to_find = swarm_distance / max(1e-9, float(cfg.speed))

			if found:
				rl_times_by_n[n_particles].append(float(time_to_find))
				rl_iters_by_n[n_particles].append(float(env.current_iter))
				rl_swarm_dists_by_n[n_particles].append(float(swarm_distance))

	def _mean_std(x: list[float]) -> tuple[float, float]:
		if not x:
			return float("nan"), float("nan")
		arr = np.asarray(x, dtype=float)
		mean = float(np.mean(arr))
		std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
		return mean, std

	results: Dict[str, float] = {
		"runs_per_n": int(runs_per_n),
		"swarm_size_low": int(swarm_size_low),
		"swarm_size_high": int(swarm_size_high),
	}

	print(
		"[compare] "
		f"algo={algo} runs_per_n={runs_per_n} swarm_size=[{int(swarm_size_low)}..{int(swarm_size_high)-1}] max_iter={cfg.max_iter}"
	)
	print(
		"N | standard: time(mean±std) iters(mean±std) dist(mean±std) || rl: time(mean±std) iters(mean±std) dist(mean±std)"
	)
	for n in range(int(swarm_size_low), int(swarm_size_high)):
		std_mt, std_st = _mean_std(std_times_by_n[n])
		std_mi, std_si = _mean_std(std_iters_by_n[n])
		std_md, std_sd = _mean_std(std_swarm_dists_by_n[n])
		rl_mt, rl_st = _mean_std(rl_times_by_n[n])
		rl_mi, rl_si = _mean_std(rl_iters_by_n[n])
		rl_md, rl_sd = _mean_std(rl_swarm_dists_by_n[n])

		# Store per-N summary in flat dict (easy to save as JSON/CSV later)
		results[f"N{n}_std_mean_time"] = std_mt
		results[f"N{n}_std_std_time"] = std_st
		results[f"N{n}_std_mean_iters"] = std_mi
		results[f"N{n}_std_std_iters"] = std_si
		results[f"N{n}_std_mean_swarm_dist"] = std_md
		results[f"N{n}_std_std_swarm_dist"] = std_sd
		results[f"N{n}_rl_mean_time"] = rl_mt
		results[f"N{n}_rl_std_time"] = rl_st
		results[f"N{n}_rl_mean_iters"] = rl_mi
		results[f"N{n}_rl_std_iters"] = rl_si
		results[f"N{n}_rl_mean_swarm_dist"] = rl_md
		results[f"N{n}_rl_std_swarm_dist"] = rl_sd

		print(
			f"{n:2d} | "
			f"std: {std_mt:7.3f}±{std_st:6.3f}  {std_mi:7.2f}±{std_si:6.2f}  {std_md:9.3f}±{std_sd:8.3f} || "
			f"rl: {rl_mt:7.3f}±{rl_st:6.3f}  {rl_mi:7.2f}±{rl_si:6.2f}  {rl_md:9.3f}±{rl_sd:8.3f}"
		)

	return results


class ReplayBuffer:
	def __init__(self, state_dim: int, action_dim: int, capacity: int, seed: int = 0):
		self.capacity = int(capacity)
		self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
		self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
		self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
		self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
		self.done = np.zeros((self.capacity, 1), dtype=np.float32)
		self.ptr = 0
		self.size = 0
		self.rng = np.random.default_rng(seed)

	def add(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray, d: bool) -> None:
		idx = self.ptr
		self.state[idx] = s
		self.action[idx] = a
		self.reward[idx] = float(r)
		self.next_state[idx] = s2
		self.done[idx] = 1.0 if d else 0.0
		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
		bs = min(int(batch_size), self.size)
		idx = self.rng.integers(0, self.size, size=bs)
		s = torch.as_tensor(self.state[idx], dtype=torch.float32)
		a = torch.as_tensor(self.action[idx], dtype=torch.float32)
		r = torch.as_tensor(self.reward[idx], dtype=torch.float32)
		s2 = torch.as_tensor(self.next_state[idx], dtype=torch.float32)
		d = torch.as_tensor(self.done[idx], dtype=torch.float32)
		return s, a, r, s2, d


class MLPActor(nn.Module):
	def __init__(self, state_dim: int, action_dim: int):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, action_dim),
			nn.Tanh(),
		)

	def forward(self, s: torch.Tensor) -> torch.Tensor:
		return self.net(s)


class MLPCritic(nn.Module):
	def __init__(self, state_dim: int, action_dim: int):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim + action_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

	def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
		return self.net(torch.cat([s, a], dim=-1))


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
	with torch.no_grad():
		for tp, sp in zip(target.parameters(), source.parameters()):
			tp.data.mul_(1.0 - tau)
			tp.data.add_(tau * sp.data)


class DDPGAgent:
	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		lr_actor: float = 1e-4,
		lr_critic: float = 1e-3,
		gamma: float = 0.99,
		tau: float = 0.005,
		act_noise: float = 0.1,
		seed: int = 0,
	):
		self.actor = MLPActor(state_dim, action_dim)
		self.actor_target = MLPActor(state_dim, action_dim)
		self.critic = MLPCritic(state_dim, action_dim)
		self.critic_target = MLPCritic(state_dim, action_dim)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.critic_target.load_state_dict(self.critic.state_dict())

		self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
		self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

		self.gamma = float(gamma)
		self.tau = float(tau)
		self.act_noise = float(act_noise)
		self.rng = np.random.default_rng(seed)

	def select_action(self, state: np.ndarray, noise: bool = True) -> np.ndarray:
		with torch.no_grad():
			s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
			a = self.actor(s).squeeze(0).cpu().numpy()
		if noise:
			a = a + self.rng.normal(0.0, self.act_noise, size=a.shape)
		return np.clip(a, -1.0, 1.0).astype(np.float32)

	def update(self, replay: ReplayBuffer, batch_size: int) -> None:
		s, a, r, s2, d = replay.sample(batch_size)

		with torch.no_grad():
			a2 = self.actor_target(s2)
			q_target = self.critic_target(s2, a2)
			y = r + self.gamma * (1.0 - d) * q_target

		q = self.critic(s, a)
		critic_loss = nn.MSELoss()(q, y)
		self.critic_opt.zero_grad()
		critic_loss.backward()
		self.critic_opt.step()

		# actor update
		actor_loss = -self.critic(s, self.actor(s)).mean()
		self.actor_opt.zero_grad()
		actor_loss.backward()
		self.actor_opt.step()

		_soft_update(self.actor_target, self.actor, self.tau)
		_soft_update(self.critic_target, self.critic, self.tau)

	def save(self, path: str) -> None:
		os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
		torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

	def load(self, path: str) -> None:
		ckpt = torch.load(path, map_location="cpu")
		self.actor.load_state_dict(ckpt["actor"])
		self.critic.load_state_dict(ckpt["critic"])
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.critic_target.load_state_dict(self.critic.state_dict())


class TD3Agent:
	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		lr_actor: float = 1e-4,
		lr_critic: float = 1e-3,
		gamma: float = 0.99,
		tau: float = 0.005,
		act_noise: float = 0.1,
		policy_noise: float = 0.2,
		noise_clip: float = 0.5,
		policy_delay: int = 2,
		seed: int = 0,
	):
		self.actor = MLPActor(state_dim, action_dim)
		self.actor_target = MLPActor(state_dim, action_dim)
		self.critic1 = MLPCritic(state_dim, action_dim)
		self.critic2 = MLPCritic(state_dim, action_dim)
		self.critic1_target = MLPCritic(state_dim, action_dim)
		self.critic2_target = MLPCritic(state_dim, action_dim)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.critic1_target.load_state_dict(self.critic1.state_dict())
		self.critic2_target.load_state_dict(self.critic2.state_dict())

		self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
		self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr_critic)
		self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr_critic)

		self.gamma = float(gamma)
		self.tau = float(tau)
		self.act_noise = float(act_noise)
		self.policy_noise = float(policy_noise)
		self.noise_clip = float(noise_clip)
		self.policy_delay = int(policy_delay)
		self.total_it = 0
		self.rng = np.random.default_rng(seed)

	def select_action(self, state: np.ndarray, noise: bool = True) -> np.ndarray:
		with torch.no_grad():
			s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
			a = self.actor(s).squeeze(0).cpu().numpy()
		if noise:
			a = a + self.rng.normal(0.0, self.act_noise, size=a.shape)
		return np.clip(a, -1.0, 1.0).astype(np.float32)

	def update(self, replay: ReplayBuffer, batch_size: int) -> None:
		self.total_it += 1
		s, a, r, s2, d = replay.sample(batch_size)

		with torch.no_grad():
			noise = torch.clamp(
				torch.randn_like(a) * self.policy_noise,
				-self.noise_clip,
				self.noise_clip,
			)
			a2 = torch.clamp(self.actor_target(s2) + noise, -1.0, 1.0)
			q1_t = self.critic1_target(s2, a2)
			q2_t = self.critic2_target(s2, a2)
			q_t = torch.min(q1_t, q2_t)
			y = r + self.gamma * (1.0 - d) * q_t

		q1 = self.critic1(s, a)
		q2 = self.critic2(s, a)
		loss1 = nn.MSELoss()(q1, y)
		loss2 = nn.MSELoss()(q2, y)

		self.critic1_opt.zero_grad()
		loss1.backward()
		self.critic1_opt.step()

		self.critic2_opt.zero_grad()
		loss2.backward()
		self.critic2_opt.step()

		if self.total_it % self.policy_delay == 0:
			actor_loss = -self.critic1(s, self.actor(s)).mean()
			self.actor_opt.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()

			_soft_update(self.actor_target, self.actor, self.tau)
			_soft_update(self.critic1_target, self.critic1, self.tau)
			_soft_update(self.critic2_target, self.critic2, self.tau)

	def save(self, path: str) -> None:
		os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
		torch.save(
			{
				"actor": self.actor.state_dict(),
				"critic1": self.critic1.state_dict(),
				"critic2": self.critic2.state_dict(),
			},
			path,
		)

	def load(self, path: str) -> None:
		ckpt = torch.load(path, map_location="cpu")
		self.actor.load_state_dict(ckpt["actor"])
		self.critic1.load_state_dict(ckpt["critic1"])
		self.critic2.load_state_dict(ckpt["critic2"])
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.critic1_target.load_state_dict(self.critic1.state_dict())
		self.critic2_target.load_state_dict(self.critic2.state_dict())


@dataclass
class RLSPSOConfig:
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


class RLSPSOEnv:
	"""Environment wrapper around `SPSO` where the agent controls (c1, c2)."""

	def __init__(self, cfg: RLSPSOConfig, seed: int | None = None):
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

		# Keep source location fixed (SPSO defaults to center) unless explicitly provided.
		if source_pos is not None:
			self.spso.set_source(source_pos)

		self.current_iter = 0
		self.prev_best_signal = float(-self.spso.get_best_local_signal())
		return self._get_state()

	def _get_state(self) -> np.ndarray:
		assert self.spso is not None

		# 7-D state:
		# [diversity, best_signal_change, time_left, avg_vel, c1_norm, c2_norm, n_particles_norm]
		diversity = self.spso.get_mean_local_best_distance()

		current_best_signal = float(-self.spso.get_best_local_signal())
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

		# Scale reward by swarm size for this SPSO run.
		n_scale = float(self.spso.n)
		time_cost_term *= n_scale
		iteration_term *= n_scale
		proximity_term *= n_scale
		invalid_penalty *= n_scale
		reward = time_cost_term + iteration_term + proximity_term + invalid_penalty

		success_term = 0.0
		timeout_term = 0.0
		done = False
		if found:
			success_term = self.cfg.success_bonus * n_scale
			reward += success_term
			done = True

		self.current_iter += 1
		if self.current_iter >= self.cfg.max_iter:
			done = True
			timeout_term = self.cfg.timeout_penalty * n_scale
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
		current_best_signal = float(-self.spso.get_best_local_signal())
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


def train(cfg: RLSPSOConfig, episodes: int, model_path: str, seed: int | None = 0, algo: str = "ppo") -> None:
	env = RLSPSOEnv(cfg, seed=seed)

	state_dim = 7
	action_dim = 2
	algo = algo.lower()

	if algo == "ppo":
		agent: object = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=2e-4)
	elif algo == "ddpg":
		agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, seed=int(seed or 0))
	elif algo == "td3":
		agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, seed=int(seed or 0))
	else:
		raise ValueError(f"Unknown algo '{algo}'. Use ppo, ddpg, or td3.")

	replay = None
	if algo in {"ddpg", "td3"}:
		replay = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=200_000, seed=int(seed or 0))
		batch_size = 256
		start_steps = 2000
		updates_per_step = 1

	os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

	for ep in range(1, episodes + 1):
		# Randomize swarm size per episode (matches n_particles_norm mapping in state).
		n_particles = int(env.rng.integers(low=5, high=31))
		state = env.reset(n_particles=n_particles)
		ep_return = 0.0
		done = False

		for t in range(cfg.max_iter):
			if algo == "ppo":
				action, logprob = agent.select_action(state)  # type: ignore[attr-defined]
				next_state, reward, done, _info = env.step(action)
				agent.store(state, action, logprob, reward, done)  # type: ignore[attr-defined]
			else:
				assert replay is not None
				# random exploration for initial steps
				if replay.size < start_steps:
					action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
				else:
					action = agent.select_action(state, noise=True)  # type: ignore[attr-defined]
				next_state, reward, done, _info = env.step(action)
				replay.add(state, action, float(reward), next_state, bool(done))
				# learning
				if replay.size >= batch_size:
					for _ in range(updates_per_step):
						agent.update(replay, batch_size=batch_size)  # type: ignore[attr-defined]

			ep_return += float(reward)
			state = next_state
			if done:
				break

		if algo == "ppo":
			agent.update()  # type: ignore[attr-defined]

		if ep % 10 == 0:
			print(f"[train] episode={ep:5d}  return={ep_return:9.3f}  iters={env.current_iter:4d}  last_c1={env.spso.c1:.3f} last_c2={env.spso.c2:.3f}")

		if ep % 100 == 0:
			agent.save(model_path)  # type: ignore[attr-defined]
			print(f"[train] saved checkpoint -> {model_path}")

	agent.save(model_path)  # type: ignore[attr-defined]
	print(f"[train] done, saved -> {model_path}")


def evaluate(cfg: RLSPSOConfig, model_path: str, episodes: int = 30, seed: int | None = 1, algo: str = "ppo") -> None:
	env = RLSPSOEnv(cfg, seed=seed)
	algo = algo.lower()
	if algo == "ppo":
		agent: object = PPOAgent(state_dim=7, action_dim=2, lr=3e-4)
	elif algo == "ddpg":
		agent = DDPGAgent(state_dim=7, action_dim=2, seed=int(seed or 0))
	elif algo == "td3":
		agent = TD3Agent(state_dim=7, action_dim=2, seed=int(seed or 0))
	else:
		raise ValueError(f"Unknown algo '{algo}'. Use ppo, ddpg, or td3.")

	agent.load(model_path)  # type: ignore[attr-defined]

	times = []
	iters = []
	founds = 0

	for _ in range(episodes):
		state = env.reset()
		for _ in range(cfg.max_iter):
			if algo == "ppo":
				action, _lp = agent.select_action(state)  # type: ignore[attr-defined]
			else:
				action = agent.select_action(state, noise=False)  # type: ignore[attr-defined]
			state, _r, done, info = env.step(action)
			if done:
				if info.get("found", False):
					founds += 1
				break

		# episode time proxy: swarm distance / speed
		swarm_distance = float(sum(p.dist_travelled for p in env.spso.particles))
		times.append(swarm_distance / cfg.speed)
		iters.append(env.current_iter)

	print(
		f"[eval] episodes={episodes} found={founds}/{episodes} "
		f"mean_time={np.mean(times):.3f}s mean_iters={np.mean(iters):.2f}"
	)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "ddpg", "td3"], help="RL algorithm to use")
	p.add_argument("--train", action="store_true", help="Train PPO to control SPSO c1/c2")
	p.add_argument("--eval", action="store_true", help="Evaluate a trained PPO controller")
	p.add_argument("--compare", action="store_true", help="Compare standard SPSO vs RL-enhanced SPSO")
	p.add_argument("--episodes", type=int, default=12000)
	p.add_argument("--compare-episodes", type=int, default=100, help="Number of runs per N for --compare")
	p.add_argument("--max-iter", type=int, default=300)
	p.add_argument("--n-particles", type=int, default=10)
	p.add_argument("--side-length", type=float, default=100.0)
	p.add_argument("--model-path", type=str, default="models/ppo_spso_c1c2.pth")
	p.add_argument("--seed", type=int, default=0)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	cfg = RLSPSOConfig(
		side_length=float(args.side_length),
		n_particles=int(args.n_particles),
		max_iter=int(args.max_iter),
	)

	mode_count = int(bool(args.train)) + int(bool(args.eval)) + int(bool(args.compare))
	if mode_count != 1:
		raise SystemExit("Pass exactly one of --train, --eval, or --compare")

	if args.train:
		train(cfg=cfg, episodes=int(args.episodes), model_path=str(args.model_path), seed=int(args.seed), algo=str(args.algo))
	elif args.eval:
		evaluate(cfg=cfg, model_path=str(args.model_path), episodes=max(1, int(args.episodes) // 10), seed=int(args.seed), algo=str(args.algo))
	else:
		compare_standard_vs_rl(
			cfg=cfg,
			model_path=str(args.model_path),
			episodes=int(args.compare_episodes),
			seed=int(args.seed),
			algo=str(args.algo),
		)


if __name__ == "__main__":
	main()
