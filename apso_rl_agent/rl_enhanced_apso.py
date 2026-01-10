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

class RLAPSOEnv:
    """
    RL Environment for tuning APSO parameters (w1, w2, c1, c2).
    """
    def __init__(self, source_pos, bounds, num_particles=10, max_iter=500):
        self.source_pos = np.array(source_pos)
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        self.apso = None
        self.current_iter = 0
        self.prev_signal = 0.0
        
    def reset(self):
        # Initialize APSO
        # Using default values initially
        self.apso = APSO_SourceSeeker(
            objective=lambda x: 0.0,
            bounds=self.bounds,
            source_pos=self.source_pos,
            num_particles=self.num_particles,
            w1=0.675, w2=-0.285, c1=1.193, c2=1.193, T=1.0,
            S_s=1.0, alpha=0.01, termination_dist=0.1
        )
        self.current_iter = 0
        self.prev_signal = self.apso.gbest_signal
        
        return self._get_state()

    def _get_state(self):
        # State: [Swarm Diversity, Convergence Speed, Current Iteration]
        
        # 1. Swarm Diversity: Avg distance of particles from Global Best
        dists = []
        for p in self.apso.particles:
            d = np.linalg.norm(p.x - self.apso.gbest_x)
            dists.append(d)
        diversity = np.mean(dists) if dists else 0.0
        
        # 2. Convergence Speed: Rate of change in global best signal
        # S_gb(t) - S_gb(t-1)
        # Using current gbest minus previous step gbest
        current_signal = self.apso.gbest_signal
        signal_change = current_signal - self.prev_signal
        
        # 3. Current Iteration (normalized)
        norm_iter = self.current_iter / self.max_iter
        
        # Return state vector
        state = np.array([diversity, signal_change, norm_iter], dtype=np.float32)
        return state

    def step(self, action):
        """
        Apply action (APSO params), separate physics, calculate reward.
        action: np.array of shape (4,) from PPO actor (values in [-1, 1])
        """
        # Centered around standard values to improve stability of initial exploration
        # w1 range around 0.6: [-0.4, 1.6] if width is 1.0 (multiplier 1.0)
        # Let's use multipliers to give reasonable range
        
        w1 = 0.6 + action[0] * 0.8
        w2 = 0.4 + action[1] * 0.6
        c1 = 1.0 + action[2] * 1.0
        c2 = 1.0 + action[3] * 1.0
        
        # Ensure c1, c2 > 0
        c1 = max(0.01, c1)
        c2 = max(0.01, c2)
        
        T = self.apso.T
        
        # Validate parameters
        valid = False
        try:
            validate_apso_params(w1, w2, c1, c2, T)
            valid = True
        except ValueError:
            valid = False
            
        reward = 0.0
        done = False
        found = False
        
        if valid:
            # Apply parameters
            self.apso.w1 = w1
            self.apso.w2 = w2
            self.apso.c1 = c1
            self.apso.c2 = c2
            
            # Step physics
            found, min_dist = self.apso.step()
            
            # Post-step signal
            current_signal = self.apso.gbest_signal
            
            # Reward = Improvement
            improvement = current_signal - self.prev_signal
            reward = improvement * 50.0 
            
            self.prev_signal = current_signal
            
            # Bonus for finding source
            if found:
                reward += 100.0
                done = True
                
        else:
            # Invalid parameters: Penalty
            reward = -5.0
            # We treat this as a wasted step without physics update
            # But we increment iteration
            pass

        self.current_iter += 1
        if self.current_iter >= self.max_iter:
            done = True
            
        next_state = self._get_state()
        
        return next_state, reward, done, valid

def run_rl_apso_training():
    # Configuration
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    source = np.array([50.0, 50.0]) 
    
    num_particles = 20
    max_iter = 200
    
    # Init Env and Agent
    env = RLAPSOEnv(source, (lo, hi), num_particles, max_iter)
    
    state_dim = 3
    action_dim = 4
    lr = 0.0003
    agent = PPOAgent(state_dim, action_dim, lr=lr)
    
    num_episodes = 500
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
