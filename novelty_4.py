import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

# ==========================================
# 1. Swarm Environment (Obstacles & Goal)
# ==========================================
class ObstacleEnvironment:
    def __init__(self):
        self.bounds = (-10, 100)
        self.goal = np.array([90.0, 90.0])
        
        # Define Obstacles (x, y, radius)
        self.obstacles = [
            (30, 30, 15),
            (60, 60, 15),
            (20, 70, 10),
            (70, 20, 10),
            (50, 50, 10)
        ]

    def cost_function(self, x):
        """
        Fitness = Distance to Goal + Obstacle Penalty
        """
        # 1. Distance to Goal
        dist_to_goal = np.linalg.norm(x - self.goal)
        
        # 2. Obstacle Penalty
        penalty = 0
        for ox, oy, r in self.obstacles:
            obs_pos = np.array([ox, oy])
            dist_to_obs = np.linalg.norm(x - obs_pos)
            
            # If inside obstacle or safety margin
            if dist_to_obs < r:
                # Massive penalty that increases the deeper you are in
                penalty += 1000 * (r - dist_to_obs)
        
        return dist_to_goal + penalty

# ==========================================
# 2. Standard PSO (Baseline)
# ==========================================
class StandardPSO:
    def __init__(self, env, num_particles=30, max_iter=500):
        self.env = env
        self.dim = 2 # 2D Space
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = env.bounds

        # Initialize close to start (0,0)
        self.X = np.random.uniform(0, 10, (num_particles, self.dim))
        self.V = np.random.uniform(-1, 1, (num_particles, self.dim))
        
        self.pbest_pos = self.X.copy()
        self.pbest_val = np.array([self.env.cost_function(x) for x in self.X])
        
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)

        self.w = 0.729
        self.c1 = 1.494
        self.c2 = 1.494
        
        self.history_traj = [self.X.copy()] # Track positions for plotting

    def run(self):
        for i in tqdm(range(self.max_iter), desc="Standard PSO"):
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            
            self.V = (self.w * self.V) + \
                     (self.c1 * r1 * (self.pbest_pos - self.X)) + \
                     (self.c2 * r2 * (self.gbest_pos - self.X))
            
            # Velocity Clamp (simulate max robot speed)
            self.V = np.clip(self.V, -5, 5)
            
            self.X = self.X + self.V
            self.X = np.clip(self.X, self.bounds[0], self.bounds[1])

            current_vals = np.array([self.env.cost_function(x) for x in self.X])
            
            # Update PBest
            improved = current_vals < self.pbest_val
            self.pbest_pos[improved] = self.X[improved]
            self.pbest_val[improved] = current_vals[improved]

            # Update GBest
            if np.min(current_vals) < self.gbest_val:
                self.gbest_val = np.min(current_vals)
                self.gbest_pos = self.X[np.argmin(current_vals)]

            self.history_traj.append(self.X.copy())
            
        return self.gbest_val, self.history_traj

# ==========================================
# 3. RL Component: PPO Agent (Standard MLP)
# ==========================================
class Normalizer:
    def __init__(self, n_inputs):
        self.n = 0
        self.mean = np.zeros(n_inputs)
        self.mean_diff = np.zeros(n_inputs)
        self.var = np.ones(n_inputs)

    def observe(self, x):
        self.n += 1
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        if self.n > 1: self.var = self.mean_diff / (self.n - 1)
    
    def normalize(self, inputs):
        obs_std = np.sqrt(self.var + 1e-8)
        return (inputs - self.mean) / obs_std

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # MLP Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Output [-1, 1]
        )
        
        # MLP Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

    def act(self, state):
        action_mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(action_mean, std)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).sum(dim=-1).detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        state_values = self.critic(state)
        std = self.log_std.exp()
        dist = Normal(action_mean, std)
        return dist.log_prob(action).sum(dim=-1), state_values, dist.entropy().sum(dim=-1)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = []
        self.K_epochs = 10
        self.eps_clip = 0.2
        self.gamma = 0.99

    def select_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action, logprob = self.policy_old.act(state)
        return action.cpu().numpy(), logprob

    def store(self, s, a, lp, r, d):
        self.buffer.append((s, a, lp, r, d))

    def update(self):
        if not self.buffer: return
        
        # Convert buffer to tensors
        s = torch.tensor(np.array([t[0] for t in self.buffer]), dtype=torch.float32)
        a = torch.tensor(np.array([t[1] for t in self.buffer]), dtype=torch.float32)
        lp = torch.tensor(np.array([t[2] for t in self.buffer]), dtype=torch.float32)
        r = [t[3] for t in self.buffer]
        d = [t[4] for t in self.buffer]
        
        rewards = []
        disc_r = 0
        for reward, done in zip(reversed(r), reversed(d)):
            if done: disc_r = 0
            disc_r = reward + (self.gamma * disc_r)
            rewards.insert(0, disc_r)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        if rewards.std() > 1e-5:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        else:
            rewards = rewards - rewards.mean()

        for _ in range(self.K_epochs):
            logprobs, values, dist_entropy = self.policy.evaluate(s, a)
            values = values.squeeze()
            
            ratios = torch.exp(logprobs - lp)
            adv = rewards - values.detach()
            
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * adv
            
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []

# ==========================================
# 4. Novel PPO-PSO (Navigation)
# ==========================================
class NovelPPO_PSO:
    def __init__(self, env, num_particles=30, max_iter=500):
        self.env = env
        self.dim = 2
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = env.bounds

        self.X = np.random.uniform(0, 10, (num_particles, self.dim))
        self.V = np.random.uniform(-1, 1, (num_particles, self.dim))
        self.pbest_pos = self.X.copy()
        self.pbest_val = np.array([self.env.cost_function(x) for x in self.X])
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)
        
        self.history_traj = [self.X.copy()]
        self.prev_gbest = self.gbest_val
        
        self.ppo_agent = PPOAgent(state_dim=3, action_dim=3)
        self.normalizer = Normalizer(3)
        self.update_timestep = 20

    def get_state(self, i):
        # Calculate current snapshot features
        dist_to_goal = np.linalg.norm(self.gbest_pos - self.env.goal) / 100.0
        center = np.mean(self.X, axis=0)
        diversity = np.mean(np.linalg.norm(self.X - center, axis=1)) / 50.0
        progress = i / self.max_iter
        
        return np.array([progress, dist_to_goal, diversity])

    def scale_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        w = 0.7 + 0.3 * action[0]
        c1 = 1.5 + 1.0 * action[1]
        c2 = 1.5 + 1.0 * action[2]
        return w, c1, c2

    def reset_swarm(self):
        """Resets particle positions for a new episode but keeps the RL Agent"""
        self.X = np.random.uniform(0, 10, (self.num_particles, self.dim))
        self.V = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.pbest_pos = self.X.copy()
        self.pbest_val = np.array([self.env.cost_function(x) for x in self.X])
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)
        self.prev_gbest = self.gbest_val
        self.history_traj = [self.X.copy()]

    def pretrain(self, episodes=50):
        print(f"\n>>> Pre-training Agent for {episodes} episodes...")
        original_max_iter = self.max_iter
        self.max_iter = 200 # Shorter episodes for training
        
        for ep in tqdm(range(episodes), desc="Pre-training"):
            self.reset_swarm()
            state = self.get_state(0)
            
            for i in range(self.max_iter):
                self.normalizer.observe(state)
                norm_state = self.normalizer.normalize(state)
                
                action, logprob = self.ppo_agent.select_action(norm_state)
                w, c1, c2 = self.scale_action(action)
                
                r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
                self.V = (w * self.V) + (c1 * r1 * (self.pbest_pos - self.X)) + (c2 * r2 * (self.gbest_pos - self.X))
                self.V = np.clip(self.V, -5, 5)
                self.X = np.clip(self.X + self.V, self.bounds[0], self.bounds[1])

                vals = np.array([self.env.cost_function(x) for x in self.X])
                improved = vals < self.pbest_val
                self.pbest_pos[improved] = self.X[improved]
                self.pbest_val[improved] = vals[improved]

                if np.min(vals) < self.gbest_val:
                    self.gbest_val = np.min(vals)
                    self.gbest_pos = self.X[np.argmin(vals)]

                improvement = max(0, self.prev_gbest - self.gbest_val)
                reward = improvement if improvement > 0 else -0.1
                
                self.ppo_agent.store(norm_state, action, logprob, reward, i==self.max_iter-1)
                if i % self.update_timestep == 0: self.ppo_agent.update()
                
                self.prev_gbest = self.gbest_val
                state = self.get_state(i)
        
        self.max_iter = original_max_iter # Restore original length
        self.reset_swarm() # Reset for the real run
        print(">>> Pre-training Complete.\n")

    def run(self):
        state = self.get_state(0)
        
        for i in tqdm(range(self.max_iter), desc="Novel PPO-PSO"):
            self.normalizer.observe(state)
            norm_state = self.normalizer.normalize(state)
            
            action, logprob = self.ppo_agent.select_action(norm_state)
            w, c1, c2 = self.scale_action(action)
            
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            self.V = (w * self.V) + (c1 * r1 * (self.pbest_pos - self.X)) + (c2 * r2 * (self.gbest_pos - self.X))
            self.V = np.clip(self.V, -5, 5) # Max Speed
            
            self.X = self.X + self.V
            self.X = np.clip(self.X, self.bounds[0], self.bounds[1])

            vals = np.array([self.env.cost_function(x) for x in self.X])
            improved = vals < self.pbest_val
            self.pbest_pos[improved] = self.X[improved]
            self.pbest_val[improved] = vals[improved]

            if np.min(vals) < self.gbest_val:
                self.gbest_val = np.min(vals)
                self.gbest_pos = self.X[np.argmin(vals)]

            # Reward
            improvement = max(0, self.prev_gbest - self.gbest_val)
            reward = improvement if improvement > 0 else -0.1
            
            self.ppo_agent.store(norm_state, action, logprob, reward, i==self.max_iter-1)
            if i % self.update_timestep == 0: self.ppo_agent.update()
            
            self.prev_gbest = self.gbest_val
            state = self.get_state(i)
            self.history_traj.append(self.X.copy())
            
        return self.gbest_val, self.history_traj

# ==========================================
# 5. Visualization
# ==========================================
def run_comparison():
    env = ObstacleEnvironment()
    
    # Run Standard
    std_pso = StandardPSO(env, max_iter=500)
    std_best, std_traj = std_pso.run()
    
    # Run Novel
    nov_pso = NovelPPO_PSO(env, max_iter=500)
    nov_pso.pretrain(episodes=50) # Train first
    nov_best, nov_traj = nov_pso.run()
    
    print(f"\nFinal Cost (Standard): {std_best:.2f}")
    print(f"Final Cost (Novel PPO): {nov_best:.2f}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    titles = ["Standard PSO", "Novel PPO-PSO"]
    trajectories = [std_traj, nov_traj]
    
    for ax, title, traj in zip(axes, titles, trajectories):
        ax.set_title(title)
        ax.set_xlim(env.bounds)
        ax.set_ylim(env.bounds)
        ax.set_aspect('equal')
        
        # Draw Obstacles
        for ox, oy, r in env.obstacles:
            circle = patches.Circle((ox, oy), r, edgecolor='black', facecolor='gray', alpha=0.6)
            ax.add_patch(circle)
            
        # Draw Goal
        goal_circle = patches.Circle(env.goal, 3, color='green', label='Goal')
        ax.add_patch(goal_circle)
        
        # Draw Trajectories
        traj = np.array(traj) # [Iter, Particles, Dim]
        for p in range(traj.shape[1]):
            ax.plot(traj[:, p, 0], traj[:, p, 1], alpha=0.3, color='blue', linewidth=0.5)
            
        # Draw Start/End
        ax.scatter(traj[0, :, 0], traj[0, :, 1], c='red', s=10, label='Start')
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c='blue', s=20, label='End')
        
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('swarm_obstacle_avoidance.png')
    print("Saved simulation plot to 'swarm_obstacle_avoidance_3.png'")

if __name__ == "__main__":
    run_comparison()