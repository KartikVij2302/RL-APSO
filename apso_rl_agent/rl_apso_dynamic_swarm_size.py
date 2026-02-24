from apso import APSO_SourceSeeker, validate_apso_params
import numpy as np

class RLAPSOEnv:
    def __init__(self, source_pos, bounds, num_particles=10, max_iter=300):
        self.source_pos = np.array(source_pos)
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        self.apso = None
        self.current_iter = 0
        self.prev_signal = 0.0
        self.prev_gbest_dist = 0.0

        # Running statistics for reward normalization
        self.reward_rmean = 0.0
        self.reward_rvar = 1.0
        self.reward_count = 1e-4
        self.RN_BETA = 0.999
        self.REWARD_CLIP = 200.0

        # Logging buffers for reward components across all training steps
        # (aligned with time-to-source and iteration objectives)
        self.step_time_cost_terms = []     # negative cost proportional to travel time per step
        self.iteration_penalty_terms = []  # negative cost per iteration
        self.proximity_bonus_terms = []    # positive reward for being close to source
        self.success_bonus_terms = []
        self.timeout_penalty_terms = []

    def reset(self, source_pos=None, num_particles=None):
        """Reset the environment for a new episode.

        Parameters
        ----------
        source_pos : array-like, optional
            If provided, overrides the source location for this episode.
            Otherwise, the source is sampled uniformly within bounds.
        num_particles : int, optional
            If provided, overrides the swarm size for this episode.
            This allows training over varying swarm sizes while
            keeping the same underlying environment implementation.
        """

        # Optionally override swarm size for this episode
        if num_particles is not None:
            self.num_particles = int(num_particles)

        # --- Randomize or override source position ---
        if source_pos is not None:
            self.source_pos = np.array(source_pos)
        else:
            lo, hi = self.bounds
            self.source_pos = np.random.uniform(low=lo, high=hi)

        # Initialize APSO with standard stable parameters
        self.apso = APSO_SourceSeeker(
            objective=lambda x: 0.0,
            bounds=self.bounds,
            source_pos=self.source_pos,
            num_particles=self.num_particles,
            w1=0.675, w2=-0.285, c1=1.193, c2=1.193, T=1.0,
            S_s=1.0, alpha=0.01, termination_dist=0.1
        )

        self.current_iter = 0
        self.prev_signal = getattr(self.apso, 'gbest_signal', 0.0)
        self.prev_gbest_dist = np.linalg.norm(self.apso.gbest_x - self.source_pos)

        return self._get_state()


    def get_reward_component_means(self):
        """Return mean value of each reward component over all steps seen so far."""
        def _mean(arr):
            return float(np.mean(arr)) if arr else 0.0

        return {
            "step_time_cost": _mean(self.step_time_cost_terms),
            "iteration_penalty": _mean(self.iteration_penalty_terms),
            "proximity_bonus": _mean(self.proximity_bonus_terms),
            "success_bonus": _mean(self.success_bonus_terms),
            "timeout_penalty": _mean(self.timeout_penalty_terms),
        }

    def _get_state(self):
        # 8-Dim State: [SigChange, TimeLeft, AvgVel, w1, w2, c1, c2, num_particles]

        # 1. Signal Change
        current_signal = getattr(self.apso, 'gbest_signal', 0.0)
        signal_change = current_signal - self.prev_signal
        
        # 2. Time Remaining (Normalized 1.0 -> 0.0)
        time_left = 1.0 - (self.current_iter / self.max_iter)
        
        # 3. Average Velocity (Crucial for sensing "Energy")
        avg_vel = np.mean([np.linalg.norm(p.v) for p in self.apso.particles])
        
        # 4-7. Current Params (Normalized)
        w1 = getattr(self.apso, 'w1', 0.0)
        w2 = getattr(self.apso, 'w2', 0.0)
        c1 = getattr(self.apso, 'c1', 1.0)
        c2 = getattr(self.apso, 'c2', 1.0)
        num_particles_n = (self.num_particles - 5.0) / 25.0   
        # maps 5->0, 30->1

        state = np.array([
            signal_change,
            time_left,
            avg_vel,
            np.clip(w1/2, -1, 1),
            np.clip(w2/2, -1, 1),
            np.clip(c1/5, 0, 1),
            np.clip(c2/5, 0, 1),
            num_particles_n,
        ], dtype=np.float32)
        
        return state

    def _map_action_to_params(self, action):
    # action in [-1,1]^4 interpreted as fractional deltas in [-0.2, 0.2]
        delta_frac = 0.2
        a = np.clip(action, -1.0, 1.0)
        # compute current params
        w1_cur = getattr(self.apso, 'w1', 0.675)
        w2_cur = getattr(self.apso, 'w2', -0.285)
        c1_cur = getattr(self.apso, 'c1', 1.193)
        c2_cur = getattr(self.apso, 'c2', 1.193)

        w1 = w1_cur * (1.0 + delta_frac * a[0])
        w2 = w2_cur * (1.0 + delta_frac * a[1])
        c1 = c1_cur * (1.0 + delta_frac * a[2])
        c2 = c2_cur * (1.0 + delta_frac * a[3])

        return w1, w2, c1, c2


    def step(self, action):
        # --- 1. APPLY PARAMS WITH STABILITY CHECK ---
        w1, w2, c1, c2 = self._map_action_to_params(action)

        # Validate APSO stability for the proposed parameters.
        # If invalid, keep previous APSO parameters but apply a penalty.
        valid_params = True
        invalid_param_penalty = 0.0
        try:
            validate_apso_params(w1, w2, c1, c2, self.apso.T)
            # Only assign if parameters satisfy stability criteria
            self.apso.w1 = float(w1)
            self.apso.w2 = float(w2)
            self.apso.c1 = float(c1)
            self.apso.c2 = float(c2)
        except Exception:
            # Mark action as invalid and add a modest penalty; APSO continues
            # with its previous stable parameters so exploration isn't overly discouraged.
            valid_params = False
            invalid_param_penalty = -80.0
        # --- 2. RUN PHYSICS ---
        prev_pos_matrix = np.array([p.x.copy() for p in self.apso.particles])
        try:
            found, min_dist = self.apso.step()
        except Exception:
            found = False
            min_dist = 1000.0

        # --- 3. CALCULATE REWARD ---
        # 1. Base Clock Penalty (PASSIVE)
        # Higher than before to force faster iterations mu(I)
        base_clock_penalty = -5.0 

        # 2. Exponential Pressure (ACTIVE)
        # Becomes the dominant penalty after 50% of max_iter
        time_pressure = -10.0 * np.exp(2.0 * (self.current_iter / self.max_iter) - 1.0)
        curr_pos_matrix = np.array([p.x for p in self.apso.particles])
        step_dist = np.sum(np.linalg.norm(curr_pos_matrix - prev_pos_matrix, axis=1))
        # 3. Aggressive Progress Reward
        # Directly targets mu(Ts) by rewarding high-speed approach
        dist_delta = self.prev_gbest_dist - min_dist
        progress_reward = 75.0 * dist_delta # Reward meters gained, penalize meters lost

        # 4. Efficiency Constraint
        # Penalize total swarm movement mu(SD) to keep paths straight
        fuel_penalty = -0.01 * step_dist

        # 5. Massive Time-Sensitive Success Payout
        success_bonus = 1500.0 * np.cos((np.pi / 2) * (self.current_iter / self.max_iter)) if found else 0.0

        reward = base_clock_penalty + time_pressure + progress_reward + fuel_penalty + success_bonus + invalid_param_penalty
        success_term  = 0.0
        timeout_term = 0.0
        done = False
        if found:
            success_term = 500.0
            reward += success_term
            done = True
         
        # Update trackers
        self.current_iter += 1
        
        if self.current_iter >= self.max_iter:
            done = True
            timeout_term = -20.0  # Timeout penalty
            reward += timeout_term
        beta_iter = 1.0
        frac = self.current_iter / max(1, self.max_iter)
        iteration_term = -beta_iter * np.exp(frac)
        gamma_close = 10.0
        map_diag = getattr(self, "map_diag", 100.0)   # set map_diag on reset() if you randomize map size
        min_dist_norm = min_dist / (map_diag + 1e-6)
        proximity_term = gamma_close * np.exp(-1.0 * min_dist_norm)
        # Log individual reward components for analysis
        self.step_time_cost_terms.append(time_pressure)
        self.iteration_penalty_terms.append(iteration_term)
        self.proximity_bonus_terms.append(proximity_term)
        self.success_bonus_terms.append(success_term)
        self.timeout_penalty_terms.append(timeout_term)

        # --- 4. RUNNING REWARD NORMALIZATION + CLIPPING ---
        # update running mean/var (Welford-ish exponential)
        old_mean = self.reward_rmean
        self.reward_rmean = self.RN_BETA * self.reward_rmean + (1 - self.RN_BETA) * reward
        self.reward_rvar = self.RN_BETA * self.reward_rvar + (1 - self.RN_BETA) * (reward - old_mean) ** 2
        r_std = np.sqrt(self.reward_rvar) + 1e-6

        # normalize and clip
        reward_norm = float(np.clip((reward - self.reward_rmean) / r_std, -self.REWARD_CLIP, self.REWARD_CLIP))

        return self._get_state(), reward_norm, done, valid_params