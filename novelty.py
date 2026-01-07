import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

# ==========================================
# 1. Benchmark Function (Rastrigin)
# ==========================================
def rastrigin(x):
    """
    Rastrigin function: Global minimum is 0 at x = [0, ...].
    """
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# ==========================================
# 2. Standard PSO Implementation
# ==========================================
class StandardPSO:
    def __init__(self, cost_func, dim=30, num_particles=50, max_iter=1500, bounds=(-5.12, 5.12)):
        self.cost_func = cost_func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds

        self.X = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
        self.V = np.random.uniform(-1, 1, (num_particles, dim))
        self.pbest_pos = self.X.copy()
        self.pbest_val = np.array([self.cost_func(x) for x in self.X])
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)

        # Standard "Golden" Parameters for PSO
        self.w = 0.729
        self.c1 = 1.494
        self.c2 = 1.494

        self.history = []

    def run(self):
        for i in tqdm(range(self.max_iter), desc="Standard PSO"):
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            
            self.V = (self.w * self.V) + (self.c1 * r1 * (self.pbest_pos - self.X)) + (self.c2 * r2 * (self.gbest_pos - self.X))
            self.X = np.clip(self.X + self.V, self.bounds[0], self.bounds[1])

            current_vals = np.array([self.cost_func(x) for x in self.X])
            improved_indices = current_vals < self.pbest_val
            self.pbest_pos[improved_indices] = self.X[improved_indices]
            self.pbest_val[improved_indices] = current_vals[improved_indices]

            if np.min(current_vals) < self.gbest_val:
                self.gbest_val = np.min(current_vals)
                self.gbest_pos = self.X[np.argmin(current_vals)]

            self.history.append(self.gbest_val)
        return self.gbest_val, self.history

# ==========================================
# 3. RL Component: Deep Continuous PPO Agent
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor Network: Outputs Mean of actions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Continuous output between -1 and 1
        )
        
        # Learnable standard deviation (log_std for numerical stability)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0) # Init std ~0.36
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(action_mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(action_mean, std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = []

    def select_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)
        return action.cpu().numpy(), action_logprob

    def store_transition(self, state, action, logprob, reward, done):
        self.buffer.append((state, action, logprob, reward, done))

    def update(self):
        if not self.buffer:
            return

        # FIX: Optimization to avoid slow list-to-tensor conversion warning
        # Convert lists of numpy arrays to a single numpy array first
        states_np = np.array([t[0] for t in self.buffer])
        actions_np = np.array([t[1] for t in self.buffer])
        logprobs_np = np.array([t[2] for t in self.buffer])
        
        states = torch.tensor(states_np, dtype=torch.float32)
        actions = torch.tensor(actions_np, dtype=torch.float32)
        logprobs = torch.tensor(logprobs_np, dtype=torch.float32)
        rewards = [t[3] for t in self.buffer]
        dones = [t[4] for t in self.buffer]
        
        # Monte Carlo Returns
        rewards_norm = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_norm.insert(0, discounted_reward)
            
        rewards_norm = torch.tensor(rewards_norm, dtype=torch.float32)
        if rewards_norm.std() > 0:
            rewards_norm = (rewards_norm - rewards_norm.mean()) / (rewards_norm.std() + 1e-7)

        # Optimize policy
        for _ in range(self.K_epochs):
            logprobs_new, state_values, dist_entropy = self.policy.evaluate(states, actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs_new - logprobs)

            advantages = rewards_norm - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards_norm) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []

# ==========================================
# 4. Novel PSO: PPO-Controlled (Continuous)
# ==========================================
class NovelPPO_PSO:
    def __init__(self, cost_func, dim=30, num_particles=50, max_iter=1500, bounds=(-5.12, 5.12)):
        self.cost_func = cost_func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds

        self.X = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
        self.V = np.random.uniform(-1, 1, (num_particles, dim))
        self.pbest_pos = self.X.copy()
        self.pbest_val = np.array([self.cost_func(x) for x in self.X])
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)
        
        self.history = []
        self.reward_history = []
        self.param_history = []  # Store parameter history
        self.stagnation_counter = 0
        self.prev_gbest = self.gbest_val
        self.prev_avg_cost = np.mean(self.pbest_val)

        self.ppo_agent = PPOAgent(state_dim=3, action_dim=3, lr=0.0003, K_epochs=10)
        self.update_timestep = 50 

    def get_state(self, iteration):
        center = np.mean(self.X, axis=0)
        max_dist = np.linalg.norm(np.array([self.bounds[1]]*self.dim) - np.array([self.bounds[0]]*self.dim))
        diversity = np.mean(np.linalg.norm(self.X - center, axis=1)) / (max_dist + 1e-9)
        progress = iteration / self.max_iter
        stagnation = min(1.0, self.stagnation_counter / 50.0)
        return [progress, diversity, stagnation]

    def scale_action(self, action):
        """
        WARM START STRATEGY:
        Center the action scaling around the 'Golden Parameters' (w=0.729, c=1.494).
        """
        action = np.clip(action, -1.0, 1.0)
        
        # w:  0 -> 0.73. Range [0.4, 1.0]
        w  = 0.73 + 0.3 * action[0]
        # c1: 0 -> 1.5.  Range [0.5, 2.5]
        c1 = 1.50 + 1.0 * action[1]
        # c2: 0 -> 1.5.  Range [0.5, 2.5]
        c2 = 1.50 + 1.0 * action[2]
        
        return w, c1, c2

    def calculate_reward(self, current_gbest, current_avg_cost):
        gbest_improve = max(0, self.prev_gbest - current_gbest)
        avg_improve = max(0, self.prev_avg_cost - current_avg_cost)
        
        reward = 0
        
        if self.prev_gbest > 1e-9:
             reward += (gbest_improve / self.prev_gbest) * 20
        else:
             reward += gbest_improve * 100
             
        if self.prev_avg_cost > 1e-9:
            reward += (avg_improve / self.prev_avg_cost) * 5
            
        if gbest_improve <= 0:
            reward -= 0.05 
            
        return reward

    def run(self):
        state = self.get_state(0)
        
        for i in tqdm(range(1, self.max_iter + 1), desc="Novel Continuous PPO-PSO"):
            action_raw, prob = self.ppo_agent.select_action(state)
            w, c1, c2 = self.scale_action(action_raw)
            
            # Store params for plotting
            self.param_history.append([w, c1, c2])
            
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            self.V = (w * self.V) + (c1 * r1 * (self.pbest_pos - self.X)) + (c2 * r2 * (self.gbest_pos - self.X))
            self.X = np.clip(self.X + self.V, self.bounds[0], self.bounds[1])

            current_vals = np.array([self.cost_func(x) for x in self.X])
            improved_indices = current_vals < self.pbest_val
            self.pbest_pos[improved_indices] = self.X[improved_indices]
            self.pbest_val[improved_indices] = current_vals[improved_indices]

            current_best_val = np.min(current_vals)
            current_avg_val = np.mean(current_vals)
            
            if current_best_val < self.gbest_val:
                self.gbest_val = current_best_val
                self.gbest_pos = self.X[np.argmin(current_vals)]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            reward = self.calculate_reward(self.gbest_val, current_avg_val)
            self.reward_history.append(reward)
            
            done = (i == self.max_iter)
            self.ppo_agent.store_transition(state, action_raw, prob, reward, done)
            
            if i % self.update_timestep == 0:
                self.ppo_agent.update()
            
            self.prev_gbest = self.gbest_val
            self.prev_avg_cost = current_avg_val
            state = self.get_state(i)
            self.history.append(self.gbest_val)
            
        return self.gbest_val, self.history, self.reward_history, np.array(self.param_history)

# ==========================================
# 5. Comparison
# ==========================================
def run_comparison():
    DIM = 30
    PARTICLES = 50 
    ITERATIONS = 2000 
    
    print(f"--- Running Comparison (PyTorch PPO Mode) ---")
    
    start = time.time()
    std_pso = StandardPSO(rastrigin, dim=DIM, num_particles=PARTICLES, max_iter=ITERATIONS)
    std_best, std_hist = std_pso.run()
    std_time = time.time() - start
    
    start = time.time()
    nov_pso = NovelPPO_PSO(rastrigin, dim=DIM, num_particles=PARTICLES, max_iter=ITERATIONS)
    nov_best, nov_hist, rewards, params = nov_pso.run()
    nov_time = time.time() - start
    
    print("\nResults:")
    print(f"{'Metric':<20} | {'Standard PSO':<15} | {'Novel PPO-PSO':<15}")
    print("-" * 56)
    print(f"{'Best Fitness':<20} | {std_best:.6f}        | {nov_best:.6f}")
    print(f"{'Time (s)':<20} | {std_time:.4f}          | {nov_time:.4f}")

    plt.figure(figsize=(18, 5))
    
    # 1. Convergence
    plt.subplot(1, 3, 1)
    plt.plot(std_hist, label='Standard', color='blue', alpha=0.7)
    plt.plot(nov_hist, label='Novel PPO', color='red', linewidth=2)
    plt.title('Convergence (Lower is Better)')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Reward
    plt.subplot(1, 3, 2)
    plt.plot(rewards, label='RL Reward', color='green', alpha=0.6)
    plt.title('Agent Learning Signal')
    plt.xlabel('Iteration')
    plt.grid(True, alpha=0.3)

    # 3. Learned Parameters
    plt.subplot(1, 3, 3)
    plt.plot(params[:, 0], label='Inertia (w)', color='purple', alpha=0.8)
    plt.plot(params[:, 1], label='Cognitive (c1)', color='orange', alpha=0.8, linestyle='--')
    plt.plot(params[:, 2], label='Social (c2)', color='cyan', alpha=0.8, linestyle='-.')
    plt.title('Learned Parameter Strategy')
    plt.xlabel('Iteration')
    plt.ylim(0, 3.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('pso_comparison.png')
    print("\nPlot saved as 'pso_comparison.png'")

if __name__ == "__main__":
    run_comparison()