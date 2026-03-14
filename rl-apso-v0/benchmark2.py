import pygame
import random
import numpy as np
import math

# --- 1. Simulation Settings ---
WIDTH, HEIGHT = 1000, 700
DRONE_COUNT = 25
OBSTACLE_COUNT = 5
MAX_ITER = 500      # Max lifetime of one "run" before auto-reset
MAX_VEL = 5         # Max pixels per frame
TARGET_TOLERANCE = 0.1 # Distance to target to be considered "found"

# Colors
COLOR_BG = (240, 240, 240)
COLOR_TARGET = (0, 200, 100)
COLOR_STD_PSO = (0, 100, 255)  # Blue
COLOR_RL_PSO = (220, 50, 50)   # Red
COLOR_OBSTACLE = (80, 80, 80)
COLOR_TEXT = (10, 10, 10)
COLOR_GBEST = (255, 200, 0) # Gold
COLOR_WINNER = (50, 200, 50) # Green for winner text

# --- 2. The Obstacle Class ---
class Obstacle:
    def __init__(self, x, y, radius):
        self.position = pygame.Vector2(x, y)
        self.radius = radius
        self.color = COLOR_OBSTACLE

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.radius)

# --- 3. The Drone (Particle) Class ---
class Drone:
    def __init__(self, x, y, color):
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        self.color = color
        
        self.pbest_pos = self.position.copy()
        self.pbest_val = float('inf')

    def evaluate(self, target_pos, obstacles):
        """Calculate fitness based on distance AND obstacles."""
        distance_to_target = self.position.distance_to(target_pos)
        
        obstacle_penalty = 0
        for obs in obstacles:
            dist_to_obs = self.position.distance_to(obs.position)
            if dist_to_obs < obs.radius:
                # High penalty for being inside an obstacle
                obstacle_penalty += 10000 
            elif dist_to_obs < obs.radius + 30: # 30px buffer zone
                # Scaled penalty for being near
                obstacle_penalty += 100 * ( (obs.radius + 30) - dist_to_obs)
                
        fitness = distance_to_target + obstacle_penalty
        
        if fitness < self.pbest_val:
            self.pbest_val = fitness
            self.pbest_pos = self.position.copy()
            
    def update(self, w, c1, c2, gbest_pos, obstacles):
        """Update drone's velocity and position, now with local avoidance."""
        r1, r2 = random.random(), random.random()
        
        # --- PSO Global Pathfinding ---
        cognitive_vel = c1 * r1 * (self.pbest_pos - self.position)
        social_vel = c2 * r2 * (gbest_pos - self.position)
        pso_velocity = (w * self.velocity) + cognitive_vel + social_vel

        # --- Local Obstacle Avoidance (Reflex) ---
        avoidance_velocity = pygame.Vector2(0, 0)
        for obs in obstacles:
            dist_vec = self.position - obs.position
            dist = dist_vec.magnitude()
            safe_dist = obs.radius + 20 # 20px "reflex" buffer
            
            if dist < safe_dist and dist > 0:
                # Add a force pushing away from the obstacle
                repel_strength = (safe_dist - dist) / safe_dist # Stronger when closer
                avoidance_velocity += (dist_vec.normalize() * repel_strength * MAX_VEL * 2)

        # --- Combine Velocities ---
        self.velocity = pso_velocity + avoidance_velocity
        
        # Clamp velocity to max speed
        if self.velocity.magnitude() > MAX_VEL:
            self.velocity.scale_to_length(MAX_VEL)
            
        # Update position
        self.position += self.velocity
        
        # Clamp position to screen bounds
        self.position.x = max(0, min(self.position.x, WIDTH))
        self.position.y = max(0, min(self.position.y, HEIGHT))

    def draw(self, screen):
        # Draw drone
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), 6)
        # Draw outline
        pygame.draw.circle(screen, COLOR_TEXT, (int(self.position.x), int(self.position.y)), 6, 1)

# --- 4. The Q-Learning Agent ---
class RLAgent:
    def __init__(self):
        self.n_states = 3  # Early, Mid, Late
        self.actions = {
            0: (0.9, 2.5, 1.0), # Explore (high w, high c1)
            1: (0.7, 2.0, 2.0), # Balance
            2: (0.4, 1.0, 2.5)  # Exploit (low w, high c2)
        }
        self.n_actions = len(self.actions)
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1 # Exploration rate

    def get_state(self, iteration, max_iter):
        if iteration < max_iter / 3:
            return 0  # Early
        elif iteration < 2 * max_iter / 3:
            return 1  # Mid
        else:
            return 2  # Late

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.actions.keys())) # Explore
        else:
            return np.argmax(self.q_table[state, :]) # Exploit

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

# --- 5. The Swarm Manager ---
class Swarm:
    def __init__(self, n_drones, color, swarm_type="Standard"):
        self.n_drones = n_drones
        self.color = color
        self.swarm_type = swarm_type
        
        self.drones = [] # Initialized in reset
        self.gbest_pos = pygame.Vector2(0, 0)
        self.gbest_val = float('inf')
        
        if self.swarm_type == "RL-PSO":
            self.rl_agent = RLAgent()
            self.current_rl_action = 0 # For display
        else:
            # Fixed parameters for Standard PSO
            self.w, self.c1, self.c2 = 0.729, 1.494, 1.494

        self.reset_swarm_state()

    def reset_swarm_state(self):
        # Reset individual drone positions and pbests
        self.drones = [Drone(random.randint(0, WIDTH), random.randint(0, HEIGHT), self.color) 
                       for _ in range(self.n_drones)]
        self.gbest_val = float('inf')
        # gbest_pos will be updated on the first evaluation
        # No need to reset Q-table, RL agent learns over many episodes

    def update_gbest(self):
        """Find the best pbest among all drones."""
        improved = False
        for p in self.drones:
            if p.pbest_val < self.gbest_val:
                self.gbest_val = p.pbest_val
                self.gbest_pos = p.pbest_pos.copy()
                improved = True
        return improved

    def update(self, target_pos, obstacles, iteration, max_iter):
        gbest_before = self.gbest_val
        
        # --- 1. Get PSO parameters ---
        if self.swarm_type == "Standard":
            w, c1, c2 = self.w, self.c1, self.c2
        else:
            # Ask the RL agent for parameters
            state = self.rl_agent.get_state(iteration, max_iter)
            action_idx = self.rl_agent.choose_action(state)
            w, c1, c2 = self.rl_agent.actions[action_idx]
            self.current_rl_action = action_idx
            
        # --- 2. Evaluate all drones FIRST to find pbests ---
        for drone in self.drones:
            drone.evaluate(target_pos, obstacles)
            
        # --- 3. Find the new gbest based on this evaluation ---
        improved = self.update_gbest()
        
        # --- 4. Now, update all drone positions using the NEW gbest ---
        for drone in self.drones:
            drone.update(w, c1, c2, self.gbest_pos, obstacles)

        # --- 5. (RL-PSO only) Learn from the result ---
        if self.swarm_type == "RL-PSO":
            # Reward is based on *improvement* in fitness
            # (Getting stuck gives 0 reward, forcing agent to try new actions)
            reward = (gbest_before - self.gbest_val) 
            if improved:
                reward += 1 # Bonus for any improvement
                
            next_state = self.rl_agent.get_state(iteration + 1, max_iter)
            self.rl_agent.update_q_table(state, action_idx, reward, next_state)

    def draw(self, screen):
        for drone in self.drones:
            drone.draw(screen)
            
        # Draw the gbest position
        # Only draw gbest if it's not a crazy high (penalized) value
        if self.gbest_val < 10000:
             pygame.draw.circle(screen, COLOR_GBEST, (int(self.gbest_pos.x), int(self.gbest_pos.y)), 8)
             pygame.draw.circle(screen, self.color, (int(self.gbest_pos.x), int(self.gbest_pos.y)), 8, 2)


# --- 6. Main Simulation Class ---
class PSOSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("PSO (Blue) vs. RL-Adaptive PSO (Red) with Obstacles")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        self.target_pos = pygame.Vector2(WIDTH // 2, HEIGHT // 2)
        self.obstacles = []
        self.iteration = 0
        self.running = True
        
        # --- Stats Tracking ---
        self.std_wins = 0
        self.rl_wins = 0
        self.std_stuck = 0
        self.rl_stuck = 0
        self.winner_this_round = None # "Standard", "RL-PSO", or None
        
        self.swarm_std = Swarm(DRONE_COUNT, COLOR_STD_PSO, "Standard")
        self.swarm_rl = Swarm(DRONE_COUNT, COLOR_RL_PSO, "RL-PSO")
        self.reset_round() # Initial reset

    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60) # 60 FPS
        pygame.quit()

    def create_obstacles(self):
        """Create a new set of random obstacles."""
        self.obstacles = []
        target_buffer = 100 # Don't spawn obstacles too close to target
        
        for _ in range(OBSTACLE_COUNT):
            while True:
                radius = random.randint(20, 60)
                x = random.randint(radius, WIDTH - radius)
                y = random.randint(radius, HEIGHT - radius)
                
                # Ensure it's not spawned on top of the target
                if self.target_pos.distance_to(pygame.Vector2(x, y)) > radius + target_buffer:
                    self.obstacles.append(Obstacle(x, y, radius))
                    break

    def reset_round(self):
        self.iteration = 0
        self.create_obstacles() # Create new obstacles
        self.swarm_std.reset_swarm_state()
        self.swarm_rl.reset_swarm_state()
        self.winner_this_round = None

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.target_pos = pygame.Vector2(event.pos)
                self.reset_round() # Reset simulation on click

    def update(self):
        if self.winner_this_round is not None:
            # If a round has ended, auto-reset after a short delay
            pygame.time.wait(1000) # Wait 1 second to show winner
            self.target_pos = pygame.Vector2(random.randint(0, WIDTH), random.randint(0, HEIGHT))
            self.reset_round()
            return 
            
        # Stop updating if max iterations is reached (timeout)
        if self.iteration >= MAX_ITER:
            if self.winner_this_round is None: # Only count as stuck if no one won
                if self.swarm_std.gbest_val >= TARGET_TOLERANCE:
                    self.std_stuck += 1
                if self.swarm_rl.gbest_val >= TARGET_TOLERANCE:
                    self.rl_stuck += 1
            
            # Auto-reset
            self.target_pos = pygame.Vector2(random.randint(0, WIDTH), random.randint(0, HEIGHT))
            self.reset_round()
            return

        # --- If simulation is running ---
        self.swarm_std.update(self.target_pos, self.obstacles, self.iteration, MAX_ITER)
        self.swarm_rl.update(self.target_pos, self.obstacles, self.iteration, MAX_ITER)
        self.iteration += 1

        # Check for wins
        # A "win" now means fitness is low (target is close) AND not penalized (not in an obstacle)
        std_won = self.swarm_std.gbest_val < TARGET_TOLERANCE 
        rl_won = self.swarm_rl.gbest_val < TARGET_TOLERANCE

        if std_won and not rl_won:
            self.std_wins += 1
            self.winner_this_round = "Standard"
        elif rl_won and not std_won:
            self.rl_wins += 1
            self.winner_this_round = "RL-PSO"
        elif std_won and rl_won:
            # Both reached, give it to RL-PSO for simplicity
            self.rl_wins += 1 
            self.winner_this_round = "RL-PSO (Tie)"

    def draw_text(self, text, x, y, color, font):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def draw(self):
        self.screen.fill(COLOR_BG)
        
        # Draw obstacles
        for obs in self.obstacles:
            obs.draw(self.screen)
            
        # Draw target
        pygame.draw.circle(self.screen, COLOR_TARGET, (int(self.target_pos.x), int(self.target_pos.y)), 12)
        
        # Draw swarms
        self.swarm_std.draw(self.screen)
        self.swarm_rl.draw(self.screen)
        
        # --- Draw Info Panel ---
        pygame.draw.rect(self.screen, (255, 255, 255, 200), (5, 5, 250, 265)) # Taller panel

        self.draw_text(f"Iteration: {self.iteration} / {MAX_ITER}", 15, 15, COLOR_TEXT, self.font)
        
        # Standard PSO Info
        std_fitness = self.swarm_std.gbest_val
        self.draw_text(f"Blue (Std) Fitness: {std_fitness:.2f}", 15, 45, COLOR_STD_PSO, self.font)
        self.draw_text(f"   Params: Fixed", 15, 70, COLOR_STD_PSO, self.font_small)

        # RL-PSO Info
        rl_fitness = self.swarm_rl.gbest_val
        action_map = {0: "Explore", 1: "Balance", 2: "Exploit"}
        rl_action_str = action_map.get(self.swarm_rl.current_rl_action, "N/A")
        
        self.draw_text(f"Red (RL) Fitness: {rl_fitness:.2f}", 15, 95, COLOR_RL_PSO, self.font)
        self.draw_text(f"   Action: {rl_action_str}", 15, 120, COLOR_RL_PSO, self.font_small)

        # Winner status for current round
        if self.winner_this_round:
            self.draw_text(f"Winner: {self.winner_this_round}", 15, 145, COLOR_WINNER, self.font)
        else:
            self.draw_text(f"Winner: None yet", 15, 145, COLOR_TEXT, self.font)

        # --- Overall Stats ---
        pygame.draw.line(self.screen, (200, 200, 200), (10, 170), (250, 170), 1)
        self.draw_text(f"Total Std Wins: {self.std_wins}", 15, 180, COLOR_STD_PSO, self.font)
        self.draw_text(f"Total RL Wins: {self.rl_wins}", 15, 205, COLOR_RL_PSO, self.font)
        self.draw_text(f"Std Stuck: {self.std_stuck}", 15, 230, COLOR_STD_PSO, self.font)
        self.draw_text(f"RL Stuck: {self.rl_stuck}", 15, 255, COLOR_RL_PSO, self.font)


        pygame.display.flip()

# --- 7. Start the Simulation ---
if __name__ == "__main__":
    sim = PSOSimulation()
    sim.run()