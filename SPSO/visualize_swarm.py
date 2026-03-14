"""Pygame visualization for SPSO and RL-enhanced SPSO swarm source seeking.

Runs two separate simulations with identical initial conditions:
1) Baseline SPSO with fixed (c1, c2)
2) RL-SPSO where a PPO policy adapts (c1, c2) online

Example:
  python SPSO/visualize_swarm.py --model-path models/ppo_spso_2.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "imageio is required for mp4 export. Install with: pip install imageio imageio-ffmpeg"
    ) from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for coefficient plot export. Install with: pip install matplotlib"
    ) from exc

try:
    import pygame
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "pygame is required for visualization. Install with: pip install pygame"
    ) from exc


# Make repo root importable when running: python SPSO/visualize_swarm.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from apso_rl_agent.PPO import PPOAgent

try:
    from .spso import SPSO
except Exception:  # pragma: no cover
    from spso import SPSO


@dataclass
class SimConfig:
    side_length: float = 100.0
    n_particles: int = 10
    max_iter: int = 300
    T: float = 1.0
    speed: float = 10.0
    omega: float = 0.721
    c1_baseline: float = 1.193
    c2_baseline: float = 1.193
    delta_frac: float = 0.2
    c_min: float = 0.05
    c_max: float = 5.0


@dataclass
class PlaybackConfig:
    width: int = 1320
    height: int = 860
    fps: int = 60
    iterations_per_second: float = 6.0
    pause_between_runs: float = 2.0
    trail_length: int = 90


@dataclass(frozen=True)
class VizTheme:
    bg: tuple[int, int, int] = (26, 30, 36)
    vignette: tuple[int, int, int, int] = (4, 6, 8, 70)
    world_fill: tuple[int, int, int] = (44, 50, 58)
    grid: tuple[int, int, int] = (76, 86, 98)
    world_border: tuple[int, int, int] = (118, 132, 148)
    source_ring: tuple[int, int, int] = (237, 148, 61)
    source_outer: tuple[int, int, int] = (244, 171, 94)
    source_inner: tuple[int, int, int] = (255, 232, 181)
    baseline_body: tuple[int, int, int] = (91, 174, 255)
    rl_body: tuple[int, int, int] = (95, 234, 167)
    particle_center: tuple[int, int, int] = (236, 242, 248)
    heading: tuple[int, int, int] = (211, 221, 230)
    panel_fill: tuple[int, int, int] = (31, 35, 43)
    panel_border: tuple[int, int, int] = (118, 132, 148)
    panel_text: tuple[int, int, int] = (225, 232, 240)


@dataclass
class SimTrace:
    name: str
    algo: str
    run_index: int
    positions: List[np.ndarray]
    source: np.ndarray
    found: bool
    found_iter: int
    c1_hist: List[float]
    c2_hist: List[float]
    min_dist_hist: List[float]

    @property
    def total_steps(self) -> int:
        return max(0, len(self.positions) - 1)


def _policy_action(agent: PPOAgent, state: np.ndarray, deterministic: bool) -> np.ndarray:
    if deterministic:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32)
            a = agent.policy_old.actor(s)
        return a.cpu().numpy()

    action, _logprob = agent.select_action(state)
    return np.asarray(action, dtype=np.float32)


def _spso_state(spso: SPSO, prev_best_signal: float, current_iter: int, max_iter: int) -> np.ndarray:
    diversity = spso.get_mean_local_best_distance()
    current_best_signal = float(-spso.get_best_local_signal())
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


def _apply_rl_c1c2(cfg: SimConfig, spso: SPSO, action: np.ndarray) -> None:
    a = np.asarray(action, dtype=float).reshape(-1)
    a = np.clip(a, -1.0, 1.0)

    c1_cur, c2_cur = float(spso.c1), float(spso.c2)
    c1 = c1_cur * (1.0 + cfg.delta_frac * float(a[0]))
    c2 = c2_cur * (1.0 + cfg.delta_frac * float(a[1]))

    if (not np.isfinite(c1)) or (not np.isfinite(c2)) or c1 <= 0.0 or c2 <= 0.0:
        c1, c2 = c1_cur, c2_cur

    spso.c1 = float(np.clip(c1, cfg.c_min, cfg.c_max))
    spso.c2 = float(np.clip(c2, cfg.c_min, cfg.c_max))


def _capture_positions(spso: SPSO) -> np.ndarray:
    return np.array([p.position.copy() for p in spso.particles], dtype=np.float32)


def run_baseline_trace(cfg: SimConfig, source: np.ndarray, seed: int) -> SimTrace:
    np.random.seed(seed)

    spso = SPSO(
        n_particles=cfg.n_particles,
        side_length=cfg.side_length,
        omega=cfg.omega,
        c1=cfg.c1_baseline,
        c2=cfg.c2_baseline,
        T=cfg.T,
        speed=cfg.speed,
    )
    spso.set_source(source)

    positions = [_capture_positions(spso)]
    c1_hist = [float(spso.c1)]
    c2_hist = [float(spso.c2)]
    min_dist_hist = [float(min(np.linalg.norm(p.position - spso.source) for p in spso.particles))]

    found = False
    found_iter = cfg.max_iter
    for k in range(1, cfg.max_iter + 1):
        found = bool(spso.step())
        positions.append(_capture_positions(spso))
        c1_hist.append(float(spso.c1))
        c2_hist.append(float(spso.c2))
        min_dist_hist.append(float(min(np.linalg.norm(p.position - spso.source) for p in spso.particles)))
        if found:
            found_iter = k
            break

    return SimTrace(
        name="SPSO Baseline",
        algo="baseline",
        run_index=0,
        positions=positions,
        source=np.asarray(source, dtype=np.float32),
        found=found,
        found_iter=int(found_iter),
        c1_hist=c1_hist,
        c2_hist=c2_hist,
        min_dist_hist=min_dist_hist,
    )


def run_rl_trace(cfg: SimConfig, source: np.ndarray, model_path: str, seed: int, deterministic: bool) -> SimTrace:
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = PPOAgent(state_dim=7, action_dim=2, lr=3e-4)
    agent.load(model_path)

    spso = SPSO(
        n_particles=cfg.n_particles,
        side_length=cfg.side_length,
        omega=cfg.omega,
        c1=cfg.c1_baseline,
        c2=cfg.c2_baseline,
        T=cfg.T,
        speed=cfg.speed,
    )
    spso.set_source(source)

    positions = [_capture_positions(spso)]
    c1_hist = [float(spso.c1)]
    c2_hist = [float(spso.c2)]
    min_dist_hist = [float(min(np.linalg.norm(p.position - spso.source) for p in spso.particles))]

    prev_best_signal = float(-spso.get_best_local_signal())
    found = False
    found_iter = cfg.max_iter

    for t in range(cfg.max_iter):
        state = _spso_state(spso, prev_best_signal=prev_best_signal, current_iter=t, max_iter=cfg.max_iter)
        action = _policy_action(agent, state, deterministic=deterministic)
        _apply_rl_c1c2(cfg, spso, action)

        found = bool(spso.step())
        positions.append(_capture_positions(spso))
        c1_hist.append(float(spso.c1))
        c2_hist.append(float(spso.c2))
        min_dist_hist.append(float(min(np.linalg.norm(p.position - spso.source) for p in spso.particles)))

        if found:
            found_iter = t + 1
            break

        prev_best_signal = float(-spso.get_best_local_signal())

    return SimTrace(
        name="RL-Enhanced SPSO",
        algo="rl",
        run_index=0,
        positions=positions,
        source=np.asarray(source, dtype=np.float32),
        found=found,
        found_iter=int(found_iter),
        c1_hist=c1_hist,
        c2_hist=c2_hist,
        min_dist_hist=min_dist_hist,
    )


def _world_to_screen(xy: np.ndarray, L: float, rect: pygame.Rect) -> tuple[int, int]:
    x = rect.left + (float(xy[0]) / L) * rect.width
    y = rect.bottom - (float(xy[1]) / L) * rect.height
    return int(round(x)), int(round(y))


def _draw_grid(surface: pygame.Surface, rect: pygame.Rect, spacing: int, color: tuple[int, int, int]) -> None:
    for x in range(rect.left, rect.right + 1, spacing):
        pygame.draw.line(surface, color, (x, rect.top), (x, rect.bottom), 1)
    for y in range(rect.top, rect.bottom + 1, spacing):
        pygame.draw.line(surface, color, (rect.left, y), (rect.right, y), 1)


def _draw_scan_rings(surface: pygame.Surface, center: tuple[int, int], base_r: int, color: tuple[int, int, int], t: float) -> None:
    for i in range(3):
        r = base_r + int((t * 30 + i * 16) % 44)
        pygame.draw.circle(surface, color, center, r, 1)


def _lerp_positions(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return a * (1.0 - alpha) + b * alpha


def _draw_hud(
    surface: pygame.Surface,
    body_font: pygame.font.Font,
    trace: SimTrace,
    step_index: int,
    total_runs: int,
    theme: VizTheme,
) -> None:
    panel = pygame.Rect(38, 38, 350, 172)
    pygame.draw.rect(surface, theme.panel_fill, panel, border_radius=12)
    pygame.draw.rect(surface, theme.panel_border, panel, width=2, border_radius=12)

    def blit_line(text: str, y: int, color: tuple[int, int, int] | None = None) -> int:
        c = theme.panel_text if color is None else color
        srf = body_font.render(text, True, c)
        surface.blit(srf, (panel.left + 18, y))
        return y + 30

    y = panel.top + 16
    y = blit_line(f"run: {trace.run_index + 1}/{total_runs} ({trace.name})", y)
    y = blit_line(f"min_dist: {trace.min_dist_hist[min(step_index, len(trace.min_dist_hist)-1)]:.3f}", y)
    y = blit_line(f"found: {'YES' if trace.found else 'NO'}", y)
    blit_line(f"iterations_to_found: {trace.found_iter if trace.found else 'timeout'}", y)


def _render_scene(
    surface: pygame.Surface,
    body_font: pygame.font.Font,
    sim_cfg: SimConfig,
    pb_cfg: PlaybackConfig,
    trace: SimTrace,
    base_step: int,
    alpha: float,
    sim_time_s: float,
    total_runs: int,
    theme: VizTheme,
) -> None:
    world_rect = pygame.Rect(28, 28, 880, 804)
    next_step = min(base_step + 1, trace.total_steps)
    pos_a = trace.positions[base_step]
    pos_b = trace.positions[next_step]
    pos_now = _lerp_positions(pos_a, pos_b, alpha)

    # Background layers
    surface.fill(theme.bg)
    vignette = pygame.Surface((pb_cfg.width, pb_cfg.height), pygame.SRCALPHA)
    pygame.draw.rect(vignette, theme.vignette, vignette.get_rect(), width=0)
    surface.blit(vignette, (0, 0))

    pygame.draw.rect(surface, theme.world_fill, world_rect, border_radius=14)
    _draw_grid(surface, world_rect, spacing=40, color=theme.grid)
    pygame.draw.rect(surface, theme.world_border, world_rect, width=2, border_radius=14)

    # Source marker
    src_xy = _world_to_screen(trace.source, sim_cfg.side_length, world_rect)
    _draw_scan_rings(surface, src_xy, base_r=10, color=theme.source_ring, t=sim_time_s)
    pygame.draw.circle(surface, theme.source_outer, src_xy, 7)
    pygame.draw.circle(surface, theme.source_inner, src_xy, 3)

    # Particles + heading vectors
    particle_count = pos_now.shape[0]
    body_color = theme.baseline_body if trace.algo == "baseline" else theme.rl_body
    prev_for_heading = trace.positions[max(0, base_step - 1)]

    for pi in range(particle_count):
        pxy = _world_to_screen(pos_now[pi], sim_cfg.side_length, world_rect)
        pygame.draw.circle(surface, body_color, pxy, 6)
        pygame.draw.circle(surface, theme.particle_center, pxy, 2)

        heading = pos_now[pi] - prev_for_heading[pi]
        norm = float(np.linalg.norm(heading))
        if norm > 1e-9:
            h = heading / norm
            arrow_world = pos_now[pi] + h * 2.5
            arr = _world_to_screen(arrow_world, sim_cfg.side_length, world_rect)
            pygame.draw.line(surface, theme.heading, pxy, arr, 2)

    _draw_hud(
        surface=surface,
        body_font=body_font,
        trace=trace,
        step_index=base_step,
        total_runs=total_runs,
        theme=theme,
    )


def save_simulation_mp4(sim_cfg: SimConfig, pb_cfg: PlaybackConfig, traces: List[SimTrace], output_path: str) -> None:
    pygame.init()
    surface = pygame.Surface((pb_cfg.width, pb_cfg.height))
    body_font = pygame.font.SysFont("dejavusansmono", 21)
    theme = VizTheme()
    total_runs = max((t.run_index for t in traces), default=0) + 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with imageio.get_writer(output_path, fps=pb_cfg.fps, codec="libx264", quality=8) as writer:
        for run_idx, trace in enumerate(traces):
            run_duration_s = trace.total_steps / max(1e-6, pb_cfg.iterations_per_second)
            run_frames = max(1, int(np.ceil(run_duration_s * pb_cfg.fps)))

            for f in range(run_frames):
                sim_time_s = min(run_duration_s, f / max(1e-6, float(pb_cfg.fps)))
                iter_pos = sim_time_s * pb_cfg.iterations_per_second
                base_step = int(np.floor(iter_pos))
                alpha = float(iter_pos - base_step)
                if base_step >= trace.total_steps:
                    base_step = trace.total_steps
                    alpha = 0.0

                _render_scene(
                    surface=surface,
                    body_font=body_font,
                    sim_cfg=sim_cfg,
                    pb_cfg=pb_cfg,
                    trace=trace,
                    base_step=base_step,
                    alpha=alpha,
                    sim_time_s=sim_time_s,
                    total_runs=total_runs,
                    theme=theme,
                )
                frame = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
                writer.append_data(frame)

            if run_idx < len(traces) - 1 and pb_cfg.pause_between_runs > 0.0:
                pause_frames = max(1, int(np.ceil(pb_cfg.pause_between_runs * pb_cfg.fps)))
                for _ in range(pause_frames):
                    _render_scene(
                        surface=surface,
                        body_font=body_font,
                        sim_cfg=sim_cfg,
                        pb_cfg=pb_cfg,
                        trace=trace,
                        base_step=trace.total_steps,
                        alpha=0.0,
                        sim_time_s=run_duration_s,
                        total_runs=total_runs,
                        theme=theme,
                    )
                    frame = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
                    writer.append_data(frame)

    pygame.quit()


def save_coefficients_plot(traces: List[SimTrace], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plt.figure(figsize=(11, 6), dpi=130)

    grouped: dict[str, List[SimTrace]] = {"baseline": [], "rl": []}
    for trace in traces:
        if trace.algo in grouped:
            grouped[trace.algo].append(trace)

    for algo, group in grouped.items():
        if not group:
            continue

        max_len = max(len(t.c1_hist) for t in group)
        c1_mat = np.full((len(group), max_len), np.nan, dtype=np.float32)
        c2_mat = np.full((len(group), max_len), np.nan, dtype=np.float32)

        for i, t in enumerate(group):
            c1_mat[i, : len(t.c1_hist)] = t.c1_hist
            c2_mat[i, : len(t.c2_hist)] = t.c2_hist

        xs = np.arange(max_len)
        c1_mean = np.nanmean(c1_mat, axis=0)
        c2_mean = np.nanmean(c2_mat, axis=0)

        label_prefix = "SPSO" if algo == "baseline" else "RL-SPSO"
        plt.plot(xs, c1_mean, linewidth=2.2, label=f"{label_prefix} c1")
        plt.plot(xs, c2_mean, linewidth=2.2, linestyle="--", label=f"{label_prefix} c2")

    plt.xlabel("Iteration")
    plt.ylabel("Coefficient Value")
    plt.title("SPSO Hyperparameters Over Time")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, format="jpg")
    plt.close()


def play_visualization(sim_cfg: SimConfig, pb_cfg: PlaybackConfig, traces: List[SimTrace]) -> None:
    pygame.init()
    pygame.display.set_caption("SPSO Robotics Visualization")
    screen = pygame.display.set_mode((pb_cfg.width, pb_cfg.height))
    clock = pygame.time.Clock()

    body_font = pygame.font.SysFont("dejavusansmono", 21)
    theme = VizTheme()
    total_runs = max((t.run_index for t in traces), default=0) + 1

    run_idx = 0
    sim_time_s = 0.0
    paused = False
    auto_switch_wait = 0.0

    running = True
    while running:
        dt = clock.tick(pb_cfg.fps) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    sim_time_s = 0.0
                    auto_switch_wait = 0.0
                elif ev.key == pygame.K_TAB:
                    run_idx = (run_idx + 1) % len(traces)
                    sim_time_s = 0.0
                    auto_switch_wait = 0.0
                elif ev.key == pygame.K_RIGHT and paused:
                    sim_time_s += 1.0 / max(1e-6, pb_cfg.iterations_per_second)

        trace = traces[run_idx]
        if not paused:
            sim_time_s += dt

        iter_pos = sim_time_s * pb_cfg.iterations_per_second
        base_step = int(np.floor(iter_pos))
        alpha = float(iter_pos - base_step)

        if base_step >= trace.total_steps:
            base_step = trace.total_steps
            alpha = 0.0
            if not paused:
                auto_switch_wait += dt
                if auto_switch_wait >= pb_cfg.pause_between_runs:
                    run_idx = (run_idx + 1) % len(traces)
                    sim_time_s = 0.0
                    auto_switch_wait = 0.0

        _render_scene(
            surface=screen,
            body_font=body_font,
            sim_cfg=sim_cfg,
            pb_cfg=pb_cfg,
            trace=trace,
            base_step=base_step,
            alpha=alpha,
            sim_time_s=sim_time_s,
            total_runs=total_runs,
            theme=theme,
        )

        pygame.display.flip()

    pygame.quit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True, help="Path to trained PPO model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-particles", type=int, default=10)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--side-length", type=float, default=100.0)
    p.add_argument("--speed", type=float, default=10.0)
    p.add_argument("--omega", type=float, default=0.721)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--c1", type=float, default=1.193)
    p.add_argument("--c2", type=float, default=1.193)

    p.add_argument("--fixed-source", nargs=2, type=float, default=[50.0, 50.0])
    p.add_argument("--random-source", action="store_true")
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--iters-per-second", type=float, default=6.0)
    p.add_argument("--pause-between-runs", type=float, default=5.0)
    p.add_argument("--trail-length", type=int, default=90)
    p.add_argument("--mc-runs", type=int, default=5)
    p.add_argument("--mp4-path", type=str, default="results/swarm_visualization.mp4")
    p.add_argument("--coeff-plot-path", type=str, default="results/spso_coefficients.jpg")
    p.add_argument("--no-display", action="store_true", help="Skip pygame window and only export files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise SystemExit(f"Model not found: {args.model_path}")

    sim_cfg = SimConfig(
        side_length=float(args.side_length),
        n_particles=int(args.n_particles),
        max_iter=int(args.max_iter),
        T=float(args.T),
        speed=float(args.speed),
        omega=float(args.omega),
        c1_baseline=float(args.c1),
        c2_baseline=float(args.c2),
    )
    pb_cfg = PlaybackConfig(
        fps=int(args.fps),
        iterations_per_second=float(args.iters_per_second),
        pause_between_runs=float(args.pause_between_runs),
        trail_length=int(args.trail_length),
    )

    mc_runs = max(1, int(args.mc_runs))
    traces: List[SimTrace] = []

    print(f"Running Monte Carlo simulations: {mc_runs} runs")
    for run_idx in range(mc_runs):
        run_seed = int(args.seed) + run_idx
        rng = np.random.default_rng(run_seed)

        if bool(args.random_source):
            source = rng.uniform(low=0.0, high=sim_cfg.side_length, size=(2,)).astype(np.float32)
        else:
            source = np.array([float(args.fixed_source[0]), float(args.fixed_source[1])], dtype=np.float32)

        baseline = run_baseline_trace(sim_cfg, source=source, seed=run_seed)
        baseline.run_index = run_idx
        baseline.name = "SPSO Baseline"

        rl_trace = run_rl_trace(
            sim_cfg,
            source=source,
            model_path=str(args.model_path),
            seed=run_seed,
            deterministic=bool(args.deterministic),
        )
        rl_trace.run_index = run_idx
        rl_trace.name = "RL-Enhanced SPSO"

        traces.extend([baseline, rl_trace])

        print(
            f"  run {run_idx + 1}/{mc_runs} seed={run_seed} source=({source[0]:.2f}, {source[1]:.2f}) "
            f"| baseline found={baseline.found} iter={baseline.found_iter} "
            f"| rl found={rl_trace.found} iter={rl_trace.found_iter}"
        )

    save_simulation_mp4(sim_cfg, pb_cfg, traces=traces, output_path=str(args.mp4_path))
    save_coefficients_plot(traces=traces, output_path=str(args.coeff_plot_path))

    print(f"Saved simulation video: {args.mp4_path}")
    print(f"Saved coefficient plot: {args.coeff_plot_path}")

    if not bool(args.no_display):
        play_visualization(sim_cfg, pb_cfg, traces=traces)


if __name__ == "__main__":
    main()
