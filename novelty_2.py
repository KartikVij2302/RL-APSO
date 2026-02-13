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
def set_seed(seed=42):
    """Freezes the random number generator for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    # Ensure PyTorch deterministic operations (slightly slower but necessary)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

def griewank(x):
    sum_sq = np.sum(x**2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_sq / 4000 - prod_cos

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

BENCHMARKS = {
    "Sphere": (sphere, (-100, 100)),
    "Rosenbrock": (rosenbrock, (-30, 30)),
    "Ackley": (ackley, (-32, 32)),
    "Griewank": (griewank, (-600, 600)),
    "Rastrigin": (rastrigin, (-5.12, 5.12))
}

# ==========================================
# Helper: Deterministic Initialization
# ==========================================
def deterministic_init(num_particles, dim, bounds):
    X = np.zeros((num_particles, dim))
    # Create evenly spaced values
    linear_spread = np.linspace(bounds[0], bounds[1], num_particles)
    
    # Latin-Hypercube style shift
    for i in range(num_particles):
        for d in range(dim):
            # Shift index to ensure good coverage in high dimensions
            idx = (i + d) % num_particles
            X[i, d] = linear_spread[idx]
    return X

# ==========================================
# 2. Standard PSO
# ==========================================
class StandardPSO:
    def __init__(self, cost_func, dim=30, num_particles=50, max_iter=1500, bounds=(-5.12, 5.12)):
        self.cost_func = cost_func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds
        
        # Calculate Max Velocity (20% of range)
        self.v_max = 0.2 * (bounds[1] - bounds[0])

        self.X = deterministic_init(num_particles, dim, bounds)
        self.V = np.zeros((num_particles, dim))
        
        self.pbest_pos = self.X.copy()
        self.pbest_val = np.array([self.cost_func(x) for x in self.X])
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)
        
        self.w, self.c1, self.c2 = 0.729, 1.494, 1.494
        self.history = []

    def run(self, desc="Standard PSO"):
        for i in tqdm(range(self.max_iter), desc=desc, leave=False):
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            
            # Update Velocity
            self.V = (self.w * self.V) + (self.c1 * r1 * (self.pbest_pos - self.X)) + (self.c2 * r2 * (self.gbest_pos - self.X))
            
            # --- CRITICAL FIX: Velocity Clamping ---
            self.V = np.clip(self.V, -self.v_max, self.v_max)
            
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
# 3. RL Component
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
        return (inputs - self.mean) / (np.sqrt(self.var) + 1e-8)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, action_dim), nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1) 
        )
    def act(self, state):
        action_mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(action_mean, std)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).sum(dim=-1).detach()
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(action_mean, std)
        return dist.log_prob(action).sum(dim=-1), self.critic(state), dist.entropy().sum(dim=-1)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=50):
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []

    def select_action(self, state):
        with torch.no_grad():
            action, log_prob = self.policy_old.act(torch.FloatTensor(state))
        return action.cpu().numpy(), log_prob

    def store_transition(self, s, a, lp, r, d):
        self.buffer.append((s, a, lp, r, d))

    def update(self):
        if not self.buffer: return
        states = torch.tensor(np.array([t[0] for t in self.buffer]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in self.buffer]), dtype=torch.float32)
        logprobs = torch.tensor(np.array([t[2] for t in self.buffer]), dtype=torch.float32)
        rewards, dones = [t[3] for t in self.buffer], [t[4] for t in self.buffer]
        
        rewards_norm = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_norm.insert(0, discounted_reward)
        rewards_norm = torch.tensor(rewards_norm, dtype=torch.float32)
        if rewards_norm.std() > 0: rewards_norm = (rewards_norm - rewards_norm.mean()) / (rewards_norm.std() + 1e-7)

        for _ in range(self.K_epochs):
            logprobs_new, state_values, dist_entropy = self.policy.evaluate(states, actions)
            state_values = torch.squeeze(state_values) 
            ratios = torch.exp(logprobs_new - logprobs)
            advantages = rewards_norm - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards_norm) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []

# ==========================================
# 4. Novel PPO-PSO
# ==========================================
class NovelPPO_PSO:
    def __init__(self, cost_func, dim=30, num_particles=50, max_iter=1500, bounds=(-5.12, 5.12)):
        self.cost_func = cost_func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds
        
        # Velocity Limit
        self.v_max = 0.2 * (bounds[1] - bounds[0])

        self.X = deterministic_init(num_particles, dim, bounds)
        self.V = np.zeros((num_particles, dim)) 
        self.pbest_pos = self.X.copy()
        self.pbest_val = np.array([self.cost_func(x) for x in self.X])
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)
        
        self.history = []
        self.ppo_agent = PPOAgent(3, 3)
        self.normalizer = Normalizer(3)
        self.prev_gbest = self.gbest_val
        self.stagnation_counter = 0

    def get_state(self, iteration):
        center = np.mean(self.X, axis=0)
        max_dist = np.linalg.norm(np.array([self.bounds[1]]*self.dim) - np.array([self.bounds[0]]*self.dim))
        diversity = np.mean(np.linalg.norm(self.X - center, axis=1)) / (max_dist + 1e-9)
        progress = iteration / self.max_iter
        stagnation = min(1.0, self.stagnation_counter / 50.0)
        return np.array([progress, diversity, stagnation])

    def run(self, desc="Novel PPO-PSO"):
        state = self.get_state(0)
        for i in tqdm(range(1, self.max_iter + 1), desc=desc, leave=False):
            self.normalizer.observe(state)
            norm_state = self.normalizer.normalize(state)
            
            action, prob = self.ppo_agent.select_action(norm_state)
            
            # --- SAFER SCALING ---
            # w: [0.4, 0.9] -> Prevents explosion
            w  = float(0.65 + 0.25 * action[0]) 
            c1 = float(1.50 + 1.0 * action[1])
            c2 = float(1.50 + 1.0 * action[2])

            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            
            self.V = (w * self.V) + (c1 * r1 * (self.pbest_pos - self.X)) + (c2 * r2 * (self.gbest_pos - self.X))
            
            # --- CRITICAL FIX: Velocity Clamping ---
            self.V = np.clip(self.V, -self.v_max, self.v_max)
            
            self.X = np.clip(self.X + self.V, self.bounds[0], self.bounds[1])

            current_vals = np.array([self.cost_func(x) for x in self.X])
            improved_indices = current_vals < self.pbest_val
            self.pbest_pos[improved_indices] = self.X[improved_indices]
            self.pbest_val[improved_indices] = current_vals[improved_indices]

            current_best = np.min(current_vals)
            if current_best < self.gbest_val:
                self.gbest_val = current_best
                self.gbest_pos = self.X[np.argmin(current_vals)]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            reward = (np.log10(self.prev_gbest + 1e-50) - np.log10(self.gbest_val + 1e-50)) * 10.0
            if reward <= 0: reward = -0.1
            
            done = (i == self.max_iter)
            self.ppo_agent.store_transition(norm_state, action, prob, reward, done)
            if i % 50 == 0:
                self.ppo_agent.update()
            
            self.prev_gbest = self.gbest_val
            state = self.get_state(i)
            self.history.append(self.gbest_val)
            
        return self.gbest_val, self.history

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    set_seed(42)
    DIM = 30
    PARTICLES = 30 
    ITERATIONS = 1000 
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    print(f"--- Running Comparison on {len(BENCHMARKS)} Functions ---")
    print(f"--- Init: Deterministic | Feature: Velocity Clamping ---")

    for idx, (name, (func, bounds)) in enumerate(BENCHMARKS.items()):
        print(f"\n>>> Running Benchmark: {name}")
        
        std_pso = StandardPSO(func, dim=DIM, num_particles=PARTICLES, max_iter=ITERATIONS, bounds=bounds)
        std_best, std_hist = std_pso.run(desc=f"Std PSO {name}")
        
        nov_pso = NovelPPO_PSO(func, dim=DIM, num_particles=PARTICLES, max_iter=ITERATIONS, bounds=bounds)
        nov_best, nov_hist = nov_pso.run(desc=f"Novel PSO {name}")
        
        print(f"{name:<12} | Std Best: {std_best:.4e} | Novel Best: {nov_best:.4e}")
        
        ax = axes[idx]
        ax.semilogy(std_hist, label='Standard', linestyle='--', color='blue', alpha=0.7)
        ax.semilogy(nov_hist, label='Novel PPO', color='red', linewidth=1.5)
        ax.set_title(f"{name} (Best: {nov_best:.2e})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log Cost")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if len(BENCHMARKS) < 6:
        fig.delaxes(axes[5])
        
    plt.tight_layout()
    plt.savefig("pso_benchmarks_clamped.png")
    print("\nComparison complete. Plot saved to 'pso_benchmarks_clamped.png'")
    plt.show()