import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class Drone:
    def __init__(self, bounds, id):
        # Initialize drone at random position within bounds
        self.position = np.random.uniform(bounds[0], bounds[1], 2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.best_position = self.position.copy()
        self.best_score = float('inf')
        self.id = id

    def update_best(self, score):
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()

class APSO:
    def __init__(self, n_drones=30, bounds=(-5.12, 5.12), w1=0.675, w2=-0.285, c1=1.193, c2=1.193, T=1.0):
        self.n_drones = n_drones
        self.bounds = bounds
        self.w1, self.w2, self.c1, self.c2 = w1, w2, c1, c2
        self.T = T
        # Source (global minimum) is at (0,0) for Rastrigin
        self.source_position = np.array([0.0, 0.0])
        self.drones = [Drone(bounds, i) for i in range(n_drones)]
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.min_distances = []
        self.prev_min_distance = float('inf')
        self.steps_without_improvement = 0

    def evaluate_rastrigin(self, position):
        """
        Rastrigin function: f(x) = An + sum(x_i^2 - A cos(2pi x_i))
        where A = 10, x_i in [-5.12, 5.12]
        Global minimum at x = 0, f(x) = 0
        """
        A = 10
        n = len(position)
        return A * n + np.sum(position**2 - A * np.cos(2 * np.pi * position))

    def update_drone(self, drone):
        r1, r2 = np.random.uniform(0, self.c1), np.random.uniform(0, self.c2)
        
        g_best = self.global_best_position if self.global_best_position is not None else drone.best_position
        
        drone.acceleration = (self.w1 * drone.acceleration +
                            r1 * (drone.best_position - drone.position) +
                            r2 * (g_best - drone.position))
        drone.velocity = self.w2 * drone.velocity + drone.acceleration * self.T
        drone.position = drone.position + drone.velocity * self.T
        drone.position = np.clip(drone.position, self.bounds[0], self.bounds[1])

    def get_swarm_metrics(self):
        positions = np.array([drone.position for drone in self.drones])
        centroid = np.mean(positions, axis=0)
        distances_to_centroid = np.linalg.norm(positions - centroid, axis=1)
        swarm_radius = np.max(distances_to_centroid)
        swarm_density = np.mean(distances_to_centroid)
        return swarm_radius, swarm_density

    def step(self):
        for drone in self.drones:
            score = self.evaluate_rastrigin(drone.position)
            drone.update_best(score)
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = drone.position.copy()

        for drone in self.drones:
            self.update_drone(drone)

        # Distance to global minimum (0,0)
        min_dist = min(np.linalg.norm(drone.position - self.source_position) for drone in self.drones)
        
        # Track improvement
        if min_dist < self.prev_min_distance:
            self.steps_without_improvement = 0
            self.prev_min_distance = min_dist
        else:
            self.steps_without_improvement += 1
            
        self.min_distances.append(min_dist)
        
        # Success if score is very low (close to 0)
        return self.global_best_score < 1e-6

class APSOEnv(gym.Env):
    def __init__(self):
        super(APSOEnv, self).__init__()
        self.action_space = spaces.Box(low=-0.5, high=2, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)
        self.apso = None
        self.episode_steps = 0
        self.max_steps = 200
        self.reset()

    def reset(self):
        self.apso = APSO()
        self.episode_steps = 0
        return self.get_observation()

    def get_observation(self):
        swarm_radius, swarm_density = self.apso.get_swarm_metrics()
        return np.array([
            self.apso.w1, 
            self.apso.w2, 
            self.apso.c1, 
            self.apso.c2,
            np.mean([np.linalg.norm(d.position - self.apso.source_position) for d in self.apso.drones]),
            np.std([np.linalg.norm(d.position - self.apso.source_position) for d in self.apso.drones]),
            swarm_radius,
            swarm_density
        ])

    def calculate_reward(self, done):
        current_min_dist = min(np.linalg.norm(d.position - self.apso.source_position) 
                             for d in self.apso.drones)
        
        # Base reward based on distance improvement
        if self.apso.steps_without_improvement == 0:
            improvement_reward = 10.0 * (self.apso.prev_min_distance - current_min_dist)
        else:
            improvement_reward = 0
        
        # Distance-based reward component
        distance_reward = -0.1 * current_min_dist
        
        # Swarm behavior rewards
        swarm_radius, swarm_density = self.apso.get_swarm_metrics()
        # Adjust thresholds for Rastrigin scale
        exploration_reward = 0.1 * swarm_radius if current_min_dist > 2 else 0
        
        # Early convergence penalty
        early_convergence_penalty = -5.0 if swarm_radius < 0.5 and current_min_dist > 1 else 0
        
        # Stability reward
        stability_reward = 2.0 if self.jury_stability_test(self.apso.w1, self.apso.w2, 
                                                         self.apso.c1, self.apso.c2) else -2.0
        
        # Combine all rewards
        reward = (
            distance_reward +
            improvement_reward +
            exploration_reward +
            early_convergence_penalty +
            stability_reward
        )
        
        # Success bonus
        if done and self.apso.global_best_score < 1e-6:
            reward += 2000
            
        return reward

    def step(self, action):
        self.episode_steps += 1
        self.apso.w1, self.apso.w2, self.apso.c1, self.apso.c2 = np.clip(action, -0.5, 2)
        done = self.apso.step()
        
        # Additional termination conditions
        current_min_dist = min(np.linalg.norm(d.position - self.apso.source_position) 
                             for d in self.apso.drones)
        
        # End episode if:
        # 1. Found the source (done is already True from apso.step())
        # 2. Max steps reached
        # 3. Stuck in local minimum
        done = done or                self.episode_steps >= self.max_steps or                (self.apso.steps_without_improvement > 50 and current_min_dist > 2)
        
        reward = self.calculate_reward(done)
        return self.get_observation(), reward, done, {}

    def jury_stability_test(self, w1, w2, c1, c2):
        return (abs(w1) < 1) and (abs(w2) < 1) and (c1 > 0) and (c2 > 0)

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self):
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        done = self.locals['dones'][0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        return True

def train_rl():
    env = APSOEnv()
    
    # Create callback
    reward_callback = RewardCallback()
    
    # Modified PPO parameters
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=5e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        )
    )
    
    # Reduced timesteps for demonstration purposes
    model.learn(total_timesteps=100000, callback=reward_callback)
    
    # Plot rewards
    episodes = np.arange(len(reward_callback.episode_rewards))
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, reward_callback.episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Episode Reward')
    plt.title('Mean Episode Reward vs Episode')
    plt.grid(True)
    plt.savefig('reward_plot.png')
    plt.close()
    
    return model, reward_callback.episode_rewards

def evaluate_rl(model, n_evaluations=50):
    seeking_times = []
    iteration_counts = []
    all_distances = []
    
    print(f"Starting evaluation with {n_evaluations} runs...")
    
    for eval_num in range(n_evaluations):
        env = APSOEnv()
        obs = env.reset()
        start_time = time.time()
        distances = []
        
        for step in range(200): # Max steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            
            min_dist = min(np.linalg.norm(drone.position - env.apso.source_position) 
                          for drone in env.apso.drones)
            distances.append(min_dist)
            
            if done:
                break
        
        elapsed_time = time.time() - start_time
        seeking_times.append(elapsed_time)
        iteration_counts.append(step + 1)
        all_distances.append(distances)
        
        if (eval_num + 1) % 10 == 0:
            print(f"Completed {eval_num + 1}/{n_evaluations} runs")

    avg_seeking_time = np.mean(seeking_times)
    avg_iterations = np.mean(iteration_counts)
    
    print(f"\nResults over {n_evaluations} runs:")
    print(f"Average Source Seeking Time (mu(Ts)): {avg_seeking_time:.4f} s")
    print(f"Average Number of Iterations (mu(I)): {avg_iterations:.2f}")
    
    # Plotting Metrics
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Source Seeking Time Distribution
    plt.subplot(1, 2, 1)
    plt.hist(seeking_times, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(avg_seeking_time, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_seeking_time:.4f}s')
    plt.xlabel('Source Seeking Time (s)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Source Seeking Time\n(N={n_evaluations})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Number of Iterations Distribution
    plt.subplot(1, 2, 2)
    plt.hist(iteration_counts, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.axvline(avg_iterations, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_iterations:.2f}')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Iterations\n(N={n_evaluations})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_apso_metrics.png')
    plt.close()
    
    # Plot Performance (Convergence)
    plt.figure(figsize=(12, 8))
    for i, distances in enumerate(all_distances):
        plt.plot(distances, alpha=0.3)
    
    # Pad distances to calculate mean
    max_len = max(len(d) for d in all_distances)
    padded_distances = []
    for d in all_distances:
        padded = np.pad(d, (0, max_len - len(d)), 'edge')
        padded_distances.append(padded)
        
    mean_distances = np.mean(padded_distances, axis=0)
    plt.plot(mean_distances, 'r-', linewidth=2, label='Mean Performance')
    
    plt.xlabel("Steps")
    plt.ylabel("Min Distance to Source")
    plt.title("RL Optimized APSO Performance on Rastrigin")
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.savefig('performance_plot.png')
    plt.close()

if __name__ == "__main__":
    model, episode_rewards = train_rl()
    evaluate_rl(model)
