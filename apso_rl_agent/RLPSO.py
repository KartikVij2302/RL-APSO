from pso import PSO_SourceSeeker
import numpy as np


class RLPSOEnv:

    def __init__(self, source_pos, bounds, num_particles=10, max_iter=300):

        self.source_pos = np.array(source_pos)
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter

        self.pso = None
        self.current_iter = 0

        self.prev_gbest = None

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------
    def reset(self, source_pos=None, num_particles=None):

        if num_particles is not None:
            self.num_particles = int(num_particles)

        if source_pos is not None:
            self.source_pos = np.array(source_pos)
        else:
            lo, hi = self.bounds
            self.source_pos = np.random.uniform(low=lo, high=hi)

        self.pso = PSO_SourceSeeker(
            bounds=self.bounds,
            source_pos=self.source_pos,
            num_particles=self.num_particles,
            w=0.7,
            c1=1.5,
            c2=1.5,
            termination_dist=0.1,
        )

        self.current_iter = 0

        self.prev_gbest = getattr(self.pso, "gbest_signal", 0.0)

        return self._get_state()

    # ---------------------------------------------------------
    # STATE
    # ---------------------------------------------------------
    def _get_state(self):

        gbest = float(getattr(self.pso, "gbest_signal", 0.0))

        avg_vel = float(np.mean(
            [np.linalg.norm(p.v) for p in self.pso.particles]
        ))

        time_left = 1.0 - (self.current_iter / self.max_iter)

        w = getattr(self.pso, "w", 0.7)
        c1 = getattr(self.pso, "c1", 1.5)
        c2 = getattr(self.pso, "c2", 1.5)

        num_particles_n = (self.num_particles - 5.0) / 25.0

        state = np.array(
            [
                gbest,
                avg_vel,
                time_left,
                np.clip(w / 2.0, 0, 1),
                np.clip(c1 / 5.0, 0, 1),
                np.clip(c2 / 5.0, 0, 1),
                num_particles_n,
            ],
            dtype=np.float32,
        )

        return state

    # ---------------------------------------------------------
    # ACTION → PARAMETERS
    # ---------------------------------------------------------
    def _map_action_to_params(self, action):

        delta = 0.2
        a = np.clip(action, -1, 1)

        w_cur = getattr(self.pso, "w", 0.7)
        c1_cur = getattr(self.pso, "c1", 1.5)
        c2_cur = getattr(self.pso, "c2", 1.5)

        w = w_cur * (1 + delta * a[0])
        c1 = c1_cur * (1 + delta * a[1])
        c2 = c2_cur * (1 + delta * a[2])

        # Keep PSO parameters in numerically stable ranges during long rollouts.
        w = np.clip(w, 0.2, 1.2)
        c1 = np.clip(c1, 0.1, 3.5)
        c2 = np.clip(c2, 0.1, 3.5)

        return w, c1, c2

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------
    def step(self, action):

        w, c1, c2 = self._map_action_to_params(action)

        self.pso.w = float(w)
        self.pso.c1 = float(c1)
        self.pso.c2 = float(c2)

        found, min_dist = self.pso.step()

        new_gbest = getattr(self.pso, "gbest_signal", 0.0)

        # ---------------------------------------------------------
        # SPARSE REWARD
        # ---------------------------------------------------------

        if new_gbest > self.prev_gbest:
            reward = 1.0
        else:
            reward = -1.0

        self.prev_gbest = new_gbest

        # ---------------------------------------------------------
        # TERMINATION
        # ---------------------------------------------------------

        self.current_iter += 1

        done = False

        if found:
            done = True

        if self.current_iter >= self.max_iter:
            done = True

        return self._get_state(), reward, done, True
        
