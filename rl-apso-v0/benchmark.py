import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. Benchmark Function (Ackley) ---

def ackley(x):
    """Ackley benchmark function for D dimensions."""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2.0 * np.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20.0 + np.e

class Particle:
    """A single particle in the swarm."""
    def __init__(self, n_dim, min_bound, max_bound):
        self.position = np.random.uniform(min_bound, max_bound, n_dim)
        self.velocity = np.random.uniform(-1, 1, n_dim)
        self.pbest_pos = np.copy(self.position)
        self.pbest_val = ackley(self.position)

class StandardPSO:
    """Standard PSO with fixed parameters."""
    def __init__(self, n_particles, n_dim, max_iter, min_bound, max_bound):
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.min_bound = min_bound
        self.max_bound = max_bound

        # --- Fixed PSO Parameters ---
        self.w = 0.729  # Inertia weight
        self.c1 = 1.494 # Cognitive (personal) coefficient
        self.c2 = 1.494 # Social (global) coefficient

        # Initialize swarm
        self.swarm = [Particle(n_dim, min_bound, max_bound) for _ in range(n_particles)]
        
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

                # 4. Evaluate fitness
                current_val = ackley(p.position)

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
    def __init__(self, n_particles, n_dim, max_iter, min_bound, max_bound):
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.min_bound = min_bound
        self.max_bound = max_bound

        # --- 1. Define RL States ---
        # We discretize the search into 3 phases
        self.n_states = 3 
        
        # --- 2. Define RL Actions (Parameter sets) ---
        # (w, c1, c2)
        self.actions = {
            0: (0.9, 2.5, 1.0), # Action 0: High exploration (high w, high c1)
            1: (0.7, 2.0, 2.0), # Action 1: Balanced
            2: (0.4, 1.0, 2.5)  # Action 2: High exploitation (low w, high c2)
        }
        self.n_actions = len(self.actions)

        # --- 3. Initialize Q-Table ---
        # Q(s, a) -> expected future reward for taking action 'a' in state 's'
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # --- 4. RL Hyperparameters ---
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1 # Exploration rate (for epsilon-greedy)

        # Initialize swarm and gbest (same as standard PSO)
        self.swarm = [Particle(n_dim, min_bound, max_bound) for _ in range(n_particles)]
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

            state = self._get_state(i)
            
            # 2. Choose an action (parameter set)
            action = self._choose_action(state)
            w, c1, c2 = self.actions[action]
            
            gbest_before_update = self.gbest_val
            
            for p in self.swarm:
                r1, r2 = np.random.rand(2)
                cognitive_vel = c1 * r1 * (p.pbest_pos - p.position)
                social_vel = c2 * r2 * (self.gbest_pos - p.position)
                p.velocity = w * p.velocity + cognitive_vel + social_vel

                p.position = p.position + p.velocity
                p.position = np.clip(p.position, self.min_bound, self.max_bound)

                current_val = ackley(p.position)
                if current_val < p.pbest_val:
                    p.pbest_val = current_val
                    p.pbest_pos = np.copy(p.position)
            
            self._update_gbest()
            
            # --- RL Agent Learns ---
            # 3. Calculate Reward
            # Reward is the *improvement* in fitness (lower is better)
            reward = gbest_before_update - self.gbest_val
            
            # 4. Get next state and update Q-Table
            next_state = self._get_state(i + 1)
            best_next_action = np.argmax(self.q_table[next_state, :])
            
            # Q-learning update rule
            td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
            td_error = td_target - self.q_table[state, action]
            self.q_table[state, action] = self.q_table[state, action] + self.alpha * td_error
            
            history.append(self.gbest_val)
            
        return self.gbest_pos, self.gbest_val, history, self.q_table
    
# --- 4. Run the Comparison ---

# --- Parameters ---
N_DIM = 10         # Dimensions
MAX_ITER = 500     # Iterations
N_PARTICLES = 30   # Swarm size
MIN_BOUND = -32.768
MAX_BOUND = 32.768
N_RUNS = 100       # Number of times to run each algo for a fair average

print(f"Comparing Standard PSO vs. RL-APSO on {N_DIM}-D Ackley function...")
print(f"Running each algorithm {N_RUNS} times...")

pso_histories = []
rl_apso_histories = []

pso_final_vals = []
rl_apso_final_vals = []

start_time = time.time()

for i in range(N_RUNS):
    # Run Standard PSO
    pso = StandardPSO(N_PARTICLES, N_DIM, MAX_ITER, MIN_BOUND, MAX_BOUND)
    _, final_val, history = pso.optimize()
    pso_histories.append(history)
    pso_final_vals.append(final_val)

    # Run RL-APSO
    rl_apso = RLAPSO(N_PARTICLES, N_DIM, MAX_ITER, MIN_BOUND, MAX_BOUND)
    _, final_val, history, q_table = rl_apso.optimize()
    rl_apso_histories.append(history)
    rl_apso_final_vals.append(final_val)

    if (i + 1) % (N_RUNS // 4) == 0:
        print(f"  ... completed run {i+1}/{N_RUNS}")

print(f"Total comparison time: {time.time() - start_time:.2f} seconds\n")

# --- 5. Process and Analyze Results ---

# Calculate average convergence
avg_pso_history = np.mean(pso_histories, axis=0)
avg_rl_apso_history = np.mean(rl_apso_histories, axis=0)

# Calculate average final fitness
avg_pso_final = np.mean(pso_final_vals)
avg_rl_apso_final = np.mean(rl_apso_final_vals)

print("--- Final Results (Average over " + str(N_RUNS) + " runs) ---")
print(f"Standard PSO   | Avg. Best Fitness: {avg_pso_final:.6f}")
print(f"RL-Adaptive PSO| Avg. Best Fitness: {avg_rl_apso_final:.6f}")

print("\n--- Learned Q-Table from last RL-APSO run ---")
print("(Rows: State (Early, Mid, Late), Cols: Action (Explore, Balance, Exploit))")
print(q_table)

# --- 6. Plot the Convergence Curves ---
plt.figure(figsize=(12, 7))
# Use semilogy for better visibility of fitness improvement
plt.semilogy(avg_pso_history, label="Standard PSO", color='blue', linestyle='--')
plt.semilogy(avg_rl_apso_history, label="RL-Adaptive PSO", color='red', linewidth=2)
plt.title(f'Convergence Comparison on {N_DIM}-D Ackley Function (Avg. of {N_RUNS} Runs)')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.show()
plt.savefig('pso_vs_rl_apso_convergence.png')