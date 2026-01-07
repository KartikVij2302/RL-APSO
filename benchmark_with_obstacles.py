import numpy as np
import matplotlib.pyplot as plt
import time
import math

# --- 1. Base Benchmark Function (Ackley) ---

def ackley(x):
    """Ackley benchmark function for D dimensions."""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2.0 * np.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20.0 + np.e

# --- 2. New Obstacle and Fitness Functions ---

def create_obstacles(n_dim, n_obs, min_bound, max_bound):
    """Creates a list of high-dimensional obstacle 'traps'."""
    obstacles = []
    # Create N-dimensional obstacles in random locations
    for _ in range(n_obs):
        # Pick a random center for the trap
        center = np.random.uniform(min_bound / 2, max_bound / 2, n_dim)
        # Pick a random radius
        radius = np.random.uniform(2, 5)
        obstacles.append({'center': center, 'radius': radius})
    print(f"  > Created {n_obs} obstacles for this run.")
    return obstacles

def evaluate_fitness(position, obstacles):
    """Calculates fitness as Ackley + obstacle penalties."""
    # 1. Get base fitness from Ackley function
    base_fitness = ackley(position)
    
    # 2. Add penalties for being in or near obstacles
    obstacle_penalty = 0
    for obs in obstacles:
        # Calculate Euclidean distance from position to obstacle center
        distance = np.linalg.norm(position - obs['center'])
        if distance < obs['radius']:
            # Massive penalty for being inside the trap
            obstacle_penalty += 10000 
            
    return base_fitness + obstacle_penalty

def is_in_obstacle(position, obstacles):
    """Checks if a final position is inside any obstacle trap."""
    for obs in obstacles:
        distance = np.linalg.norm(position - obs['center'])
        if distance < obs['radius']:
            return True
    return False

# --- 3. Particle & Algorithm Classes (Modified) ---

class Particle:
    """A single particle in the swarm."""
    def __init__(self, n_dim, min_bound, max_bound, obstacles):
        self.position = np.random.uniform(min_bound, max_bound, n_dim)
        self.velocity = np.random.uniform(-1, 1, n_dim)
        self.pbest_pos = np.copy(self.position)
        # Evaluate fitness using the new function
        self.pbest_val = evaluate_fitness(self.position, obstacles)

class StandardPSO:
    """Standard PSO with fixed parameters."""
    def __init__(self, n_particles, n_dim, max_iter, min_bound, max_bound, obstacles):
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.obstacles = obstacles # Store obstacles

        # --- Fixed PSO Parameters ---
        self.w = 0.729  # Inertia weight
        self.c1 = 1.494 # Cognitive (personal) coefficient
        self.c2 = 1.494 # Social (global) coefficient

        # Initialize swarm
        self.swarm = [Particle(n_dim, min_bound, max_bound, self.obstacles) for _ in range(n_particles)]
        
        # Initialize global best
        self.gbest_val = np.inf
        self.gbest_pos = np.zeros(n_dim)
        self._update_gbest()

    def _update_gbest(self):
        """Find the best particle in the swarm."""
        for p in self.swarm:
            if p.pbest_val < self.gbest_val:
                self.gbest_val = p.pbest_val
                self.gbest_pos = np.copy(p.pbest_pos)

    def optimize(self):
        """Run the PSO optimization loop."""
        history = [] # To store gbest value at each iteration
        
        for _ in range(self.max_iter):
            for p in self.swarm:
                # 1. Update velocity
                r1, r2 = np.random.rand(2)
                cognitive_vel = self.c1 * r1 * (p.pbest_pos - p.position)
                social_vel = self.c2 * r2 * (self.gbest_pos - p.position)
                p.velocity = self.w * p.velocity + cognitive_vel + social_vel

                # 2. Update position
                p.position = p.position + p.velocity
                
                # 3. Handle bounds
                p.position = np.clip(p.position, self.min_bound, self.max_bound)

                # 4. Evaluate fitness (using new function)
                current_val = evaluate_fitness(p.position, self.obstacles)

                # 5. Update pbest
                if current_val < p.pbest_val:
                    p.pbest_val = current_val
                    p.pbest_pos = np.copy(p.position)
            
            # 6. Update gbest
            self._update_gbest()
            history.append(self.gbest_val)
            
        return self.gbest_pos, self.gbest_val, history
    

class RLAPSO:
    """
    PSO with parameters adapted by a Q-Learning agent.
    """
    def __init__(self, n_particles, n_dim, max_iter, min_bound, max_bound, obstacles):
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.obstacles = obstacles # Store obstacles

        # --- 1. Define RL States ---
        self.n_states = 3 
        
        # --- 2. Define RL Actions (Parameter sets) ---
        self.actions = {
            0: (0.9, 2.5, 1.0), # Action 0: High exploration (high w, high c1)
            1: (0.7, 2.0, 2.0), # Action 1: Balanced
            2: (0.4, 1.0, 2.5)  # Action 2: High exploitation (low w, high c2)
        }
        self.n_actions = len(self.actions)

        # --- 3. Initialize Q-Table ---
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # --- 4. RL Hyperparameters ---
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1 # Exploration rate (for epsilon-greedy)

        # Initialize swarm and gbest (same as standard PSO)
        self.swarm = [Particle(n_dim, min_bound, max_bound, self.obstacles) for _ in range(n_particles)]
        self.gbest_val = np.inf
        self.gbest_pos = np.zeros(n_dim)
        self._update_gbest()

    def _update_gbest(self):
        """Find the best particle in the swarm."""
        improved = False
        for p in self.swarm:
            if p.pbest_val < self.gbest_val:
                self.gbest_val = p.pbest_val
                self.gbest_pos = np.copy(p.pbest_pos)
                improved = True
        return improved

    def _get_state(self, iteration):
        """Determine the current state based on the iteration."""
        if iteration < self.max_iter / 3:
            return 0 # Early phase
        elif iteration < 2 * self.max_iter / 3:
            return 1 # Mid phase
        else:
            return 2 # Late phase

    def _choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions) # Explore
        else:
            return np.argmax(self.q_table[state, :]) # Exploit

    def optimize(self):
        """Run the RL-guided PSO optimization loop."""
        history = []
        
        for i in range(self.max_iter):
            # --- RL Agent's Turn ---
            state = self._get_state(i)
            action = self._choose_action(state)
            w, c1, c2 = self.actions[action]
            
            gbest_before_update = self.gbest_val
            
            # --- PSO's Turn ---
            for p in self.swarm:
                r1, r2 = np.random.rand(2)
                cognitive_vel = c1 * r1 * (p.pbest_pos - p.position)
                social_vel = c2 * r2 * (self.gbest_pos - p.position)
                p.velocity = w * p.velocity + cognitive_vel + social_vel

                p.position = p.position + p.velocity
                p.position = np.clip(p.position, self.min_bound, self.max_bound)

                # Evaluate fitness (using new function)
                current_val = evaluate_fitness(p.position, self.obstacles)
                if current_val < p.pbest_val:
                    p.pbest_val = current_val
                    p.pbest_pos = np.copy(p.position)
            
            self._update_gbest()
            
            # --- RL Agent Learns ---
            # Reward is based on fitness improvement.
            # Getting stuck in an obstacle trap (fitness ~10000) will
            # result in a huge NEGATIVE reward, teaching the agent to avoid.
            reward = gbest_before_update - self.gbest_val
            
            next_state = self._get_state(i + 1)
            best_next_action = np.argmax(self.q_table[next_state, :])
            
            td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
            td_error = td_target - self.q_table[state, action]
            self.q_table[state, action] = self.q_table[state, action] + self.alpha * td_error
            
            history.append(self.gbest_val)
            
        return self.gbest_pos, self.gbest_val, history, self.q_table
    
# --- 4. Run the Comparison ---

# --- Parameters ---
N_DIM = 10         # Dimensions
N_OBSTACLES = 3    # Number of obstacle traps
MAX_ITER = 200     # Iterations
N_PARTICLES = 30   # Swarm size
MIN_BOUND = -32.768
MAX_BOUND = 32.768
N_RUNS = 100       # Number of times to run each algo for a fair average

print(f"Comparing Standard PSO vs. RL-APSO on {N_DIM}-D Ackley function with {N_OBSTACLES} traps...")
print(f"Running each algorithm {N_RUNS} times...")

pso_histories = []
rl_apso_histories = []

pso_final_vals = []
rl_apso_final_vals = []

# --- New Stats ---
pso_avoided_count = 0
rl_apso_avoided_count = 0

start_time = time.time()

for i in range(N_RUNS):
    # Create one set of obstacles for this run
    obstacles = create_obstacles(N_DIM, N_OBSTACLES, MIN_BOUND, MAX_BOUND)

    # Run Standard PSO
    pso = StandardPSO(N_PARTICLES, N_DIM, MAX_ITER, MIN_BOUND, MAX_BOUND, obstacles)
    pso_final_pos, final_val, history = pso.optimize()
    pso_histories.append(history)
    pso_final_vals.append(final_val)
    if not is_in_obstacle(pso_final_pos, obstacles):
        pso_avoided_count += 1

    # Run RL-APSO
    rl_apso = RLAPSO(N_PARTICLES, N_DIM, MAX_ITER, MIN_BOUND, MAX_BOUND, obstacles)
    rl_apso_final_pos, final_val, history, q_table = rl_apso.optimize()
    rl_apso_histories.append(history)
    rl_apso_final_vals.append(final_val)
    if not is_in_obstacle(rl_apso_final_pos, obstacles):
        rl_apso_avoided_count += 1

    if (i + 1) % (N_RUNS // 4) == 0:
        print(f"  ... completed run {i+1}/{N_RUNS}")

print(f"Total comparison time: {time.time() - start_time:.2f} seconds\n")

# --- 5. Process and Analyze Results ---

# Calculate average convergence
# We filter out the runs that got stuck in a trap (fitness > 1000)
# to get a more meaningful average fitness plot
def filter_histories(histories):
    filtered = [h for h in histories if h[-1] < 1000]
    if not filtered:
        # If all runs failed, just return the raw data
        return np.mean(histories, axis=0)
    return np.mean(filtered, axis=0)

avg_pso_history = filter_histories(pso_histories)
avg_rl_apso_history = filter_histories(rl_apso_histories)

# Calculate average final fitness
avg_pso_final = np.mean(pso_final_vals)
avg_rl_apso_final = np.mean(rl_apso_final_vals)

print("--- Final Results (Average over " + str(N_RUNS) + " runs) ---")
print(f"Standard PSO   | Avg. Best Fitness: {avg_pso_final:.6f}")
print(f"RL-Adaptive PSO| Avg. Best Fitness: {avg_rl_apso_final:.6f}")
print("\n--- Obstacle Avoidance Stats ---")
print(f"Standard PSO   | Successful Avoidances: {pso_avoided_count} / {N_RUNS} ({pso_avoided_count/N_RUNS*100:.1f}%)")
print(f"RL-Adaptive PSO| Successful Avoidances: {rl_apso_avoided_count} / {N_RUNS} ({rl_apso_avoided_count/N_RUNS*100:.1f}%)")


print("\n--- Learned Q-Table from last RL-APSO run ---")
print("(Rows: State (Early, Mid, Late), Cols: Action (Explore, Balance, Exploit))")
print(q_table)

# --- 6. Plot the Convergence Curves ---
plt.figure(figsize=(12, 7))
# Use semilogy for better visibility of fitness improvement
plt.semilogy(avg_pso_history, label="Standard PSO (Avg. of Successful Runs)", color='blue', linestyle='--')
plt.semilogy(avg_rl_apso_history, label="RL-Adaptive PSO (Avg. of Successful Runs)", color='red', linewidth=2)
plt.title(f'Convergence Comparison on {N_DIM}-D Ackley with {N_OBSTACLES} Traps (Avg. of {N_RUNS} Runs)')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.savefig('pso_vs_rl_apso_with_obstacles.png')
print("\nSaved convergence plot to 'pso_vs_rl_apso_with_obstacles.png'")
plt.show()