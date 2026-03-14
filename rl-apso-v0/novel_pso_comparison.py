import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

# ==========================================
# 1. Benchmark Functions
# ==========================================
def sphere(x):
    """Unimodal, separable."""
    return np.sum(x**2)

def rosenbrock(x):
    """Unimodal/Multimodal, non-separable, narrow valley."""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    """Multimodal, many local optima."""
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

def griewank(x):
    """Multimodal, many widespread local minima."""
    sum_sq = np.sum(x**2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_sq / 4000 - prod_cos

def rastrigin(x):
    """Multimodal, classic stress test."""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Dictionary mapping name -> (function, bounds)
BENCHMARKS = {
    "Sphere": (sphere, (-100, 100)),
    "Rosenbrock": (rosenbrock, (-30, 30)),
    "Ackley": (ackley, (-32, 32)),
    "Griewank": (griewank, (-600, 600)),
    "Rastrigin": (rastrigin, (-5.12, 5.12))
}

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

    def run(self, desc="Standard PSO"):
        for i in tqdm(range(self.max_iter), desc=desc, leave=False):
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
class Normalizer:
    """Tracks running mean and variance to normalize inputs (states)."""
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
        if self.n > 1:
            self.var = self.mean_diff / (self.n - 1)
    
    def normalize(self, inputs):
        obs_std = np.sqrt(self.var) + 1e-8 # Avoid div by zero
        return (inputs - self.mean) / obs_std

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() 
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

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
        if not self.buffer: return
        states_np = np.array([t[0] for t in self.buffer])
        actions_np = np.array([t[1] for t in self.buffer])
        logprobs_np = np.array([t[2] for t in self.buffer])
        states = torch.tensor(states_np, dtype=torch.float32)
        actions = torch.tensor(actions_np, dtype=torch.float32)
        logprobs = torch.tensor(logprobs_np, dtype=torch.float32)
        rewards = [t[3] for t in self.buffer]
        dones = [t[4] for t in self.buffer]
        
        rewards_norm = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_norm.insert(0, discounted_reward)
        rewards_norm = torch.tensor(rewards_norm, dtype=torch.float32)
        if rewards_norm.std() > 0:
            rewards_norm = (rewards_norm - rewards_norm.mean()) / (rewards_norm.std() + 1e-7)

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
        self.param_history = []  
        self.stagnation_counter = 0
        self.prev_gbest = self.gbest_val
        self.prev_avg_cost = np.mean(self.pbest_val)

        self.ppo_agent = PPOAgent(state_dim=3, action_dim=3, lr=0.0003, K_epochs=10)
        self.update_timestep = 50
        self.normalizer = Normalizer(n_inputs=3)

    def get_state(self, iteration):
        center = np.mean(self.X, axis=0)
        max_dist = np.linalg.norm(np.array([self.bounds[1]]*self.dim) - np.array([self.bounds[0]]*self.dim))
        diversity = np.mean(np.linalg.norm(self.X - center, axis=1)) / (max_dist + 1e-9)
        progress = iteration / self.max_iter
        stagnation = min(1.0, self.stagnation_counter / 50.0)
        return np.array([progress, diversity, stagnation])

    def scale_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        w  = 0.73 + 0.3 * action[0]
        c1 = 1.50 + 1.0 * action[1]
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

    def run(self, desc="Novel PPO-PSO"):
        state = self.get_state(0)
        for i in tqdm(range(1, self.max_iter + 1), desc=desc, leave=False):
            self.normalizer.observe(state)
            norm_state = self.normalizer.normalize(state)
            
            action_raw, prob = self.ppo_agent.select_action(norm_state)
            w, c1, c2 = self.scale_action(action_raw)
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
            self.ppo_agent.store_transition(norm_state, action_raw, prob, reward, done)
            if i % self.update_timestep == 0:
                self.ppo_agent.update()
            
            self.prev_gbest = self.gbest_val
            self.prev_avg_cost = current_avg_val
            state = self.get_state(i)
            self.history.append(self.gbest_val)
            
        return self.gbest_val, self.history, self.reward_history, np.array(self.param_history)

# ==========================================
# 5. Comparison & Plotting
# ==========================================
def run_comparison():
    DIM = 30
    ITERATIONS = 2000
    
    # Define Sweep Range
    particle_sweep = [20, 40, 60, 80, 100]
    
    print(f"--- Running Benchmark Suite: {list(BENCHMARKS.keys())} ---")
    print(f"--- Particle Sweep: {particle_sweep} ---")
    
    for func_name, (func, bounds) in BENCHMARKS.items():
        print(f"\n>>> Testing Fitness Function: {func_name}")
        
        std_results = []
        nov_results = []
        std_times = []
        nov_times = []
        
        # Data for the LAST run (for detailed plotting)
        last_std_hist = []
        last_nov_hist = []
        last_rewards = []
        last_params = []
        
        for n_particles in particle_sweep:
            # 1. Standard PSO
            start = time.time()
            std_pso = StandardPSO(func, dim=DIM, num_particles=n_particles, max_iter=ITERATIONS, bounds=bounds)
            std_best, std_hist = std_pso.run(desc=f"Std PSO ({n_particles})")
            std_time = time.time() - start
            std_results.append(std_best)
            std_times.append(std_time)
            
            # 2. Novel PPO PSO
            start = time.time()
            nov_pso = NovelPPO_PSO(func, dim=DIM, num_particles=n_particles, max_iter=ITERATIONS, bounds=bounds)
            nov_best, nov_hist, rewards, params = nov_pso.run(desc=f"Novel PSO ({n_particles})")
            nov_time = time.time() - start
            nov_results.append(nov_best)
            nov_times.append(nov_time)
            
            if n_particles == particle_sweep[-1]:
                last_std_hist = std_hist
                last_nov_hist = nov_hist
                last_rewards = rewards
                last_params = params

        # --- Visualization for THIS function ---
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"Benchmark: {func_name} (Bounds: {bounds})", fontsize=16)
        
        # Subplot 1: Sweep Results (Fitness)
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(particle_sweep, std_results, marker='o', linestyle='--', label='Standard PSO', color='blue')
        ax1.plot(particle_sweep, nov_results, marker='s', linestyle='-', label='Novel PPO', color='red')
        ax1.set_title('Impact of Swarm Size on Fitness')
        ax1.set_xlabel('Number of Particles')
        ax1.set_ylabel('Best Fitness (Lower is Better)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Convergence (For the largest swarm size)
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(last_std_hist, label='Standard', color='blue', alpha=0.5)
        ax2.plot(last_nov_hist, label='Novel PPO', color='red', linewidth=1.5)
        ax2.set_title(f'Convergence Curve (N={particle_sweep[-1]})')
        ax2.set_yscale('log')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cost')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Learned Parameters (For the largest swarm size)
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(last_params[:, 0], label='Inertia (w)', color='purple', alpha=0.8)
        ax3.plot(last_params[:, 1], label='Cognitive (c1)', color='orange', alpha=0.8, linestyle='--')
        ax3.plot(last_params[:, 2], label='Social (c2)', color='cyan', alpha=0.8, linestyle='-.')
        ax3.set_title('Learned Parameter Strategy (RL Agent)')
        ax3.set_xlabel('Iteration')
        ax3.set_ylim(0, 3.0)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Execution Time
        ax4 = plt.subplot(2, 2, 4)
        x = np.arange(len(particle_sweep))
        width = 0.35
        ax4.bar(x - width/2, std_times, width, label='Standard', color='blue', alpha=0.6)
        ax4.bar(x + width/2, nov_times, width, label='Novel PPO', color='red', alpha=0.6)
        ax4.set_xticks(x)
        ax4.set_xticklabels(particle_sweep)
        ax4.set_title('Execution Time Comparison')
        ax4.set_xlabel('Number of Particles')
        ax4.set_ylabel('Time (seconds)')
        ax4.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        filename = f"pso_comparison_{func_name.lower()}.png"
        plt.savefig(filename)
        print(f"Results Summary for {func_name}:")
        print(f"{'Particles':<10} | {'Std Fitness':<15} | {'Novel Fitness':<15}")
        print("-" * 45)
        for i, p in enumerate(particle_sweep):
            print(f"{p:<10} | {std_results[i]:<15.4f} | {nov_results[i]:<15.4f}")
        print(f"Plot saved as '{filename}'\n")

if __name__ == "__main__":
    run_comparison()