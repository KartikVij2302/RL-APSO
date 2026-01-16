import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure we can import modules from the same directory

from apso import APSO_SourceSeeker, validate_apso_params
from PPO import PPOAgent

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


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

    def reset(self):
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
        # 8-Dim State: [Diversity, SigChange, TimeLeft, AvgVel, w1, w2, c1, c2]
        
        # 1. Diversity
        dists = [np.linalg.norm(p.x - self.apso.gbest_x) for p in self.apso.particles]
        diversity = np.mean(dists) if dists else 0.0
        
        # 2. Signal Change
        current_signal = getattr(self.apso, 'gbest_signal', 0.0)
        signal_change = current_signal - self.prev_signal
        
        # 3. Time Remaining (Normalized 1.0 -> 0.0)
        time_left = 1.0 - (self.current_iter / self.max_iter)
        
        # 4. Average Velocity (Crucial for sensing "Energy")
        avg_vel = np.mean([np.linalg.norm(p.v) for p in self.apso.particles])
        
        # 5-8. Current Params (Normalized)
        w1 = getattr(self.apso, 'w1', 0.0)
        w2 = getattr(self.apso, 'w2', 0.0)
        c1 = getattr(self.apso, 'c1', 1.0)
        c2 = getattr(self.apso, 'c2', 1.0)
        
        state = np.array([
            diversity, signal_change, time_left, avg_vel,
            np.clip(w1/2, -1, 1), np.clip(w2/2, -1, 1), 
            np.clip(c1/5, 0, 1), np.clip(c2/5, 0, 1)
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
            # Mark action as invalid and add a fixed penalty; APSO continues
            # with its previous stable parameters.
            valid_params = False
            invalid_param_penalty = -50.0
        # --- 2. RUN PHYSICS ---
        prev_pos_matrix = np.array([p.x.copy() for p in self.apso.particles])
        try:
            found, min_dist = self.apso.step()
        except Exception:
            found = False
            min_dist = 1000.0

        # --- 3. CALCULATE REWARD ---
        # Reward is designed to directly encode the two objectives:
        # (1) minimise physical source seeking time, approximated via
        #     per-step travel distance at constant UAV speed;
        # (2) minimise the number of iterations until detection.

        # Sum distance moved by all particles this step
        curr_pos_matrix = np.array([p.x for p in self.apso.particles])
        step_dist = np.sum(np.linalg.norm(curr_pos_matrix - prev_pos_matrix, axis=1))
        # Convert to an average per-UAV distance and then to time using
        # the constant waypoint speed v = 10 m/s.
        UAV_SPEED = 10.0
        mean_step_dist = step_dist / self.num_particles
        step_time = mean_step_dist / UAV_SPEED  # seconds per waypoint move (average UAV)

        # A. Time cost term (log-shaped): larger per-step times get more penalty,
        #    but the increase is sublinear (diminishing returns).
        alpha_time = 15.0
        time_cost_term = -alpha_time * np.log1p(step_time)   # log(1 + step_time)

        # B. Iteration penalty (exp-shaped): later iterations incur higher cost
        #    than earlier ones.
        beta_iter = 1.0
        frac = self.current_iter / self.max_iter             # in [0,1]
        iteration_term = -beta_iter * np.exp(frac)           # in [-e, -1]

        # C. Proximity bonus (exp-shaped): reward increases as the swarm's
        #    closest UAV approaches the source (smaller min_dist).
        #    Use the min_dist returned from APSO step.
        gamma_close = 1.0
        proximity_term = gamma_close * np.exp(-0.1 * min_dist)

        # D. Success Bonus / Timeout Penalty (only at terminal steps)
        success_term = 0.0
        timeout_term = 0.0

        reward = time_cost_term + iteration_term + proximity_term + invalid_param_penalty

        done = False
        if found:
            success_term = 300.0
            reward += success_term
            done = True
        
        # Update trackers
        self.current_iter += 1
        
        if self.current_iter >= self.max_iter:
            done = True
            timeout_term = -20.0  # Timeout penalty
            reward += timeout_term

        # Log individual reward components for analysis
        self.step_time_cost_terms.append(time_cost_term)
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

def run_rl_apso_training():
    # Configuration
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    source = np.array([50.0, 50.0]) 

    num_particles = 20
    max_iter = 300

    # Init Env and Agent
    env = RLAPSOEnv(source, (lo, hi), num_particles, max_iter)

    state_dim = 8   # changed to include current APSO params
    action_dim = 4
    lr = 3e-4
    agent = PPOAgent(state_dim, action_dim, lr=lr)

    num_episodes = 4000
    rewards_history = []

    print(f"Starting RL-APSO Training for {num_episodes} episodes...")

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        valid_actions = 0

        for t in range(max_iter):
            action, logprob = agent.select_action(state)

            next_state, reward, done, valid = env.step(action)
            if valid: valid_actions += 1

            agent.store(state, action, logprob, reward, done)

            state = next_state
            ep_reward += reward

            if done:
                break

        agent.update()
        rewards_history.append(ep_reward)

        if (ep + 1) % 10 == 0:
            avg_rew = np.mean(rewards_history[-10:])
            print(f"Episode {ep+1}/{num_episodes} | Avg Reward: {avg_rew:.4f} | Valid Actions: {valid_actions}/{t+1}")

    agent.save(os.path.join(current_dir, "ppo_apso.pth"))
    print("Model saved to ppo_apso.pth")

    # Save mean reward component statistics for offline analysis
    component_means = env.get_reward_component_means()
    np.savez(os.path.join(current_dir, "reward_component_means.npz"), **component_means)
    print("Saved reward component means to reward_component_means.npz")

    # Plotting
    try:
        plt.figure()
        plt.plot(rewards_history)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("RL-APSO Training Performance")
        plt.savefig("rl_apso_training.png")
        print("Training plot saved to rl_apso_training.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    run_rl_apso_training()
