import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

# ==========================================
# 1. Drone Environment (Battery & Physics)
# ==========================================
class DroneEnvironment:
    def __init__(self):
        self.bounds = (-10, 110)
        self.goal = np.array([100.0, 100.0])
        
        # Energy Parameters
        self.initial_battery = 1000.0
        self.alpha_dist = 1.0 
        self.beta_accel = 5.0 

    def calculate_energy_consumption(self, velocity, acceleration):
        speed = np.linalg.norm(velocity)
        accel = np.linalg.norm(acceleration)
        instant_consumption = (self.alpha_dist * speed) + (self.beta_accel * accel)
        return instant_consumption

    def get_fitness(self, position):
        return np.linalg.norm(position - self.goal)

# ==========================================
# 2. Standard PSO (Baseline)
# ==========================================
class StandardDroneSwarm:
    def __init__(self, env, num_drones=30, max_iter=400):
        self.env = env
        self.num_drones = num_drones
        self.max_iter = max_iter
        self.X = np.random.uniform(0, 10, (num_drones, 2))
        self.V = np.zeros((num_drones, 2))
        self.battery = np.full(num_drones, env.initial_battery)
        self.active = np.ones(num_drones, dtype=bool) 
        self.pbest_pos = self.X.copy()
        self.pbest_val = np.array([env.get_fitness(x) for x in self.X])
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)
        self.w = 0.729
        self.c1 = 1.494
        self.c2 = 1.494
        self.history_energy = []
        self.history_cost = []

    def run(self):
        for i in tqdm(range(self.max_iter), desc="Standard Drone Swarm"):
            prev_V = self.V.copy()
            r1 = np.random.rand(self.num_drones, 2)
            r2 = np.random.rand(self.num_drones, 2)
            new_V = (self.w * self.V) + (self.c1 * r1 * (self.pbest_pos - self.X)) + (self.c2 * r2 * (self.gbest_pos - self.X))
            self.V = np.clip(new_V, -8, 8)
            move_mask = self.active[:, np.newaxis]
            self.X = np.where(move_mask, self.X + self.V, self.X)
            
            acceleration = self.V - prev_V
            for d in range(self.num_drones):
                if self.active[d]:
                    consumption = self.env.calculate_energy_consumption(self.V[d], acceleration[d])
                    self.battery[d] -= consumption
                    if self.battery[d] <= 0:
                        self.battery[d] = 0
                        self.active[d] = False 
            
            current_vals = np.array([self.env.get_fitness(x) for x in self.X])
            improved = (current_vals < self.pbest_val) & self.active
            self.pbest_pos[improved] = self.X[improved]
            self.pbest_val[improved] = current_vals[improved]
            
            if np.any(self.active):
                active_indices = np.where(self.active)[0]
                min_idx = active_indices[np.argmin(current_vals[active_indices])]
                if current_vals[min_idx] < self.gbest_val:
                    self.gbest_val = current_vals[min_idx]
                    self.gbest_pos = self.X[min_idx].copy()

            self.history_energy.append(np.mean(self.battery))
            self.history_cost.append(self.gbest_val)

            if self.gbest_val < 1.0:
                remaining = self.max_iter - i - 1
                if remaining > 0:
                    self.history_energy.extend([self.history_energy[-1]] * remaining)
                    self.history_cost.extend([self.history_cost[-1]] * remaining)
                break
            
        return self.history_energy, self.history_cost

# ==========================================
# 3. RL Components (PPO)
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
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim), nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def act(self, state):
        mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        return dist.sample(), dist
    def evaluate(self, state, action):
        mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(-1), self.critic(state), dist.entropy().sum(-1)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = []
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.K_epochs = 10

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state)
            action, dist = self.policy.act(state_t)
        return action.numpy(), dist.log_prob(action).sum(-1)

    def store(self, s, a, lp, r, d):
        self.buffer.append((s, a, lp, r, d))

    def update(self):
        if len(self.buffer) < 2: return
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
            logprobs, values, entropy = self.policy.evaluate(s, a)
            values = values.view(-1)
            if values.ndim == 0: values = values.unsqueeze(0)
            ratios = torch.exp(logprobs - lp)
            adv = rewards - values.detach()
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * adv
            loss = -torch.min(surr1, surr2) + 0.5*nn.MSELoss()(values, rewards) - 0.01*entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        self.buffer = []

# ==========================================
# 4. RL-Drone Swarm (PPO Controlled)
# ==========================================
class RLDroneSwarm:
    def __init__(self, env, num_drones=30, max_iter=400):
        self.env = env
        self.num_drones = num_drones
        self.max_iter = max_iter
        self.X = np.random.uniform(0, 10, (num_drones, 2))
        self.V = np.zeros((num_drones, 2))
        self.battery = np.full(num_drones, env.initial_battery)
        self.active = np.ones(num_drones, dtype=bool)
        self.pbest_pos = self.X.copy()
        self.pbest_val = np.array([env.get_fitness(x) for x in self.X])
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)
        self.prev_gbest_val = self.gbest_val
        self.ppo = PPOAgent(state_dim=3, action_dim=3)
        self.norm = Normalizer(3)
        self.history_energy = []
        self.history_cost = []
        self.history_params = []

    def get_state(self, t):
        progress = t / self.max_iter
        dist_to_goal = self.gbest_val / 150.0 
        avg_battery = np.mean(self.battery) / self.env.initial_battery
        return np.array([progress, dist_to_goal, avg_battery])

    def scale_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        w = 0.7 + 0.3 * action[0] 
        c1 = 1.5 + 1.0 * action[1]
        c2 = 1.5 + 1.0 * action[2]
        return w, c1, c2

    def calculate_reward(self, gbest_new, prev_gbest, avg_battery_loss, dist_to_goal, battery_state):
        # REWARD TUNING
        # 1. Distance Penalty: DRASTICALLY INCREASED to prevent laziness
        # If you are 50% away, you lose -2.5 per step. This hurts!
        dist_penalty = (dist_to_goal / 150.0) * 5.0  
        
        # 2. Improvement Reward: Keep high to encourage progress
        improvement = max(0, prev_gbest - gbest_new)
        
        # 3. Energy Penalty: DECREASED slightly so it doesn't fear moving
        energy_penalty = avg_battery_loss * 0.05 
        
        reward = (improvement * 50.0) - energy_penalty - dist_penalty
        
        # Completion Bonus
        if gbest_new < 1.0:
            reward += 100.0 + (battery_state * 0.1)
            
        return reward

    def reset_swarm(self):
        self.X = np.random.uniform(0, 10, (self.num_drones, 2))
        self.V = np.zeros((self.num_drones, 2))
        self.battery = np.full(self.num_drones, self.env.initial_battery)
        self.active = np.ones(self.num_drones, dtype=bool)
        self.pbest_pos = self.X.copy()
        self.pbest_val = np.array([self.env.get_fitness(x) for x in self.X])
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)
        self.prev_gbest_val = self.gbest_val
        self.history_energy = []
        self.history_cost = []
        self.history_params = []

    def pretrain(self, episodes=50):
        print(f"\n>>> Pre-training Agent for {episodes} episodes...")
        for ep in tqdm(range(episodes), desc="Pre-training"):
            self.reset_swarm()
            state = self.get_state(0)
            for i in range(self.max_iter):
                self.norm.observe(state)
                norm_state = self.norm.normalize(state)
                action, logprob = self.ppo.select_action(norm_state)
                w, c1, c2 = self.scale_action(action)
                
                prev_V = self.V.copy()
                r1 = np.random.rand(self.num_drones, 2)
                r2 = np.random.rand(self.num_drones, 2)
                new_V = (w * self.V) + (c1 * r1 * (self.pbest_pos - self.X)) + (c2 * r2 * (self.gbest_pos - self.X))
                self.V = np.clip(new_V, -8, 8)
                self.X += self.V
                
                acceleration = self.V - prev_V
                avg_battery_loss = 0
                for d in range(self.num_drones):
                    cons = self.env.calculate_energy_consumption(self.V[d], acceleration[d])
                    self.battery[d] -= cons
                    avg_battery_loss += cons
                avg_battery_loss /= self.num_drones

                current_vals = np.array([self.env.get_fitness(x) for x in self.X])
                improved = (current_vals < self.pbest_val)
                self.pbest_pos[improved] = self.X[improved]
                self.pbest_val[improved] = current_vals[improved]
                self.gbest_val = np.min(self.pbest_val)

                current_battery_total = np.sum(self.battery)
                reward = self.calculate_reward(self.gbest_val, self.prev_gbest_val, avg_battery_loss, self.gbest_val, current_battery_total)
                done = (i == self.max_iter - 1) or (self.gbest_val < 1.0)
                
                self.ppo.store(norm_state, action, logprob, reward, done)
                if i % 20 == 0: self.ppo.update()
                self.prev_gbest_val = self.gbest_val
                state = self.get_state(i)
                if done: break
        self.reset_swarm()
        print(">>> Pre-training Complete.\n")

    def run(self):
        state = self.get_state(0)
        for i in tqdm(range(self.max_iter), desc="RL Drone Swarm (Eval)"):
            self.norm.observe(state)
            norm_state = self.norm.normalize(state)
            action, logprob = self.ppo.select_action(norm_state)
            w, c1, c2 = self.scale_action(action)
            self.history_params.append([w, c1, c2])
            
            prev_V = self.V.copy()
            r1 = np.random.rand(self.num_drones, 2)
            r2 = np.random.rand(self.num_drones, 2)
            new_V = (w * self.V) + (c1 * r1 * (self.pbest_pos - self.X)) + (c2 * r2 * (self.gbest_pos - self.X))
            self.V = np.clip(new_V, -8, 8)
            move_mask = self.active[:, np.newaxis]
            self.X = np.where(move_mask, self.X + self.V, self.X)
            
            acceleration = self.V - prev_V
            battery_loss_step = 0
            for d in range(self.num_drones):
                if self.active[d]:
                    consumption = self.env.calculate_energy_consumption(self.V[d], acceleration[d])
                    self.battery[d] -= consumption
                    battery_loss_step += consumption
                    if self.battery[d] <= 0:
                        self.battery[d] = 0
                        self.active[d] = False
            
            if np.any(self.active):
                battery_loss_step /= np.sum(self.active)
            
            current_vals = np.array([self.env.get_fitness(x) for x in self.X])
            improved = (current_vals < self.pbest_val) & self.active
            self.pbest_pos[improved] = self.X[improved]
            self.pbest_val[improved] = current_vals[improved]
            
            if np.any(self.active):
                active_indices = np.where(self.active)[0]
                min_idx = active_indices[np.argmin(current_vals[active_indices])]
                if current_vals[min_idx] < self.gbest_val:
                    self.gbest_val = current_vals[min_idx]
                    self.gbest_pos = self.X[min_idx].copy()

            current_battery_total = np.sum(self.battery)
            reward = self.calculate_reward(self.gbest_val, self.prev_gbest_val, battery_loss_step, self.gbest_val, current_battery_total)
            done = (i == self.max_iter - 1) or (self.gbest_val < 1.0)
            
            self.ppo.store(norm_state, action, logprob, reward, done)
            if i % 20 == 0: self.ppo.update()
            
            self.prev_gbest_val = self.gbest_val
            state = self.get_state(i)
            
            self.history_energy.append(np.mean(self.battery))
            self.history_cost.append(self.gbest_val)
            
            if done and self.gbest_val < 1.0:
                remaining = self.max_iter - i - 1
                if remaining > 0:
                    self.history_energy.extend([self.history_energy[-1]] * remaining)
                    self.history_cost.extend([self.history_cost[-1]] * remaining)
                break
                
        return self.history_energy, self.history_cost, np.array(self.history_params)

def run_experiment():
    env = DroneEnvironment()
    std_swarm = StandardDroneSwarm(env)
    std_energy, std_cost = std_swarm.run()
    rl_swarm = RLDroneSwarm(env)
    rl_swarm.pretrain(episodes=50)
    rl_energy, rl_cost, rl_params = rl_swarm.run()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(std_cost, label='Standard PSO', color='blue')
    axes[0, 0].plot(rl_cost, label='RL-PSO (Energy Aware)', color='red')
    axes[0, 0].set_title('Distance to Goal (Convergence)')
    axes[0, 0].set_ylabel('Distance (m)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(std_energy, label='Standard PSO', color='blue')
    axes[0, 1].plot(rl_energy, label='RL-PSO', color='red')
    axes[0, 1].set_title('Average Battery Remaining')
    axes[0, 1].set_ylabel('Battery (Units)')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(std_energy[-1], std_cost[-1], color='blue', s=100, label='Standard Final')
    axes[1, 0].scatter(rl_energy[-1], rl_cost[-1], color='red', s=100, label='RL Final')
    axes[1, 0].set_title('Final Efficiency')
    axes[1, 0].set_xlabel('Remaining Battery')
    axes[1, 0].set_ylabel('Final Distance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    if len(rl_params) > 0:
        axes[1, 1].plot(rl_params[:, 0], label='Inertia (w)', color='purple')
        axes[1, 1].plot(rl_params[:, 1], label='Cognitive (c1)', color='orange', alpha=0.6)
        axes[1, 1].plot(rl_params[:, 2], label='Social (c2)', color='cyan', alpha=0.6)
        axes[1, 1].set_title('Learned Parameters')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drone_energy_comparison.png')
    print(f"\nStandard Final Battery: {std_energy[-1]:.2f} | Distance: {std_cost[-1]:.2f}")
    print(f"RL-PSO Final Battery:   {rl_energy[-1]:.2f} | Distance: {rl_cost[-1]:.2f}")
    print("Saved plot to 'drone_energy_comparison.png'")

if __name__ == "__main__":
    run_experiment()