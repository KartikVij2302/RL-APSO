import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure we can import modules from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from apso import APSO_SourceSeeker, validate_apso_params
from PPO import PPOAgent

import numpy as np
import os
import sys

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from apso import APSO_SourceSeeker, validate_apso_params
from PPO import PPOAgent

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
        a = np.clip(action, -1.0, 1.0)
        # Dynamic ranges allowing for negative inertia (braking) and high social force
        w1 = -0.5 + (a[0] + 1.0) * (1.5 - (-0.5)) / 2.0  # [-0.5, 1.5]
        w2 = -1.0 + (a[1] + 1.0) * (1.0 - (-1.0)) / 2.0  # [-1.0, 1.0]
        c1 = 0.01 + (a[2] + 1.0) * (4.0 - 0.01) / 2.0    # [0.01, 4.0]
        c2 = 0.01 + (a[3] + 1.0) * (4.0 - 0.01) / 2.0    # [0.01, 4.0]
        return w1, w2, c1, c2

    def step(self, action):
        # --- 1. APPLY PARAMS IMMEDIATELY (Causal Control) ---
        w1, w2, c1, c2 = self._map_action_to_params(action)
        
        # Force validity via clipping instead of rejecting (keeps physics running)
        self.apso.w1 = np.clip(w1, -0.9, 1.5)
        self.apso.w2 = np.clip(w2, -0.9, 0.9)
        self.apso.c1 = max(0.1, c1)
        self.apso.c2 = max(0.1, c2)
        
        # --- 2. RUN PHYSICS ---
        prev_pos_matrix = np.array([p.x.copy() for p in self.apso.particles])
        try:
            found, min_dist = self.apso.step()
        except Exception:
            found = False
            min_dist = 1000.0

        # --- 3. CALCULATE REWARD ---
        reward = 0.0
        
        # A. Signal Improvement (Scaled)
        current_signal = getattr(self.apso, 'gbest_signal', 0.0)
        improvement = current_signal - self.prev_signal
        reward += improvement * 50.0
        
        # B. Distance Heuristic (Guide to source)
        curr_gbest_dist = np.linalg.norm(self.apso.gbest_x - self.source_pos)
        dist_improvement = self.prev_gbest_dist - curr_gbest_dist
        reward += dist_improvement * 1.0 # +1 point per meter closer
        
        # C. Fuel Cost (Efficiency)
        # Sum distance moved by all particles
        curr_pos_matrix = np.array([p.x for p in self.apso.particles])
        step_dist = np.sum(np.linalg.norm(curr_pos_matrix - prev_pos_matrix, axis=1))
        reward -= 0.5 * step_dist # Small penalty per meter moved
        
        # D. Time Penalty (CRITICAL: Forces speed)
        reward -= 1 # Penalty per time step
        
        # E. Success Bonus
        done = False
        if found:
            reward += 100.0
            done = True
        
        # Update trackers
        self.prev_signal = current_signal
        self.prev_gbest_dist = curr_gbest_dist
        self.current_iter += 1
        
        if self.current_iter >= self.max_iter:
            done = True
            reward -= 20.0 # Timeout penalty
            
        return self._get_state(), float(reward), done, True

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
    lr = 0.0003
    agent = PPOAgent(state_dim, action_dim, lr=lr)

    num_episodes = 3000
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
