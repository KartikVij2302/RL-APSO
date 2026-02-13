from apso_rl_agent.benchmark_evaluation import evaluate_trained_rl_apso_on_benchmarks
from apso_rl_agent.rl_apso import RLAPSOEnv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch
# Ensure we can import modules from the same directory
import random
from PPO import PPOAgent

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # for more deterministic CUDA behaviour (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_rl_apso_training():
    set_global_seed(42)
    
    # Configuration
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    source = np.array([50.0, 50.0]) 

    num_particles = 20
    max_iter = 500

    # Init Env and Agent
    env = RLAPSOEnv(source, (lo, hi), num_particles, max_iter)

    state_dim = 8   
    action_dim = 4
    lr = 3e-4
    agent = PPOAgent(state_dim, action_dim, lr=lr)

    num_episodes = 10000
    rewards_history = []

    print(f"Starting RL-APSO Training for {num_episodes} episodes...")

    for ep in range(num_episodes):
        # Use fixed source position [50, 50] for all training episodes
        state = env.reset(source_pos=source)
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

    agent.save(os.path.join(current_dir, "latest_ppo_apso_fixed_source_2.pth"))
    print("Model saved to latest_ppo_apso_fixed_source_2.pth")

    # Save mean reward component statistics for offline analysis
    component_means = env.get_reward_component_means()
    np.savez(os.path.join(current_dir, "reward_component_means.npz"), **component_means)
    print("Saved reward component means to reward_component_means.npz")

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


def run_rl_apso_training_random_particles():
    """Train RL-APSO with a random swarm size per episode.

    - Grid/bounds are fixed: [0, 100] x [0, 100]
    - Source position is fixed at [50, 50]
    - For each episode, the number of particles is an integer
      sampled uniformly from [5, 15].
    """
    set_global_seed(42)

    # Fixed environment geometry and source
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    source = np.array([50.0, 50.0])

    # Max iterations unchanged; base num_particles will be overridden per episode
    base_num_particles = 10
    max_iter = 500

    env = RLAPSOEnv(source, (lo, hi), base_num_particles, max_iter)

    state_dim = 8
    action_dim = 4
    lr = 3e-4
    agent = PPOAgent(state_dim, action_dim, lr=lr)

    num_episodes = 14000
    rewards_history = []

    print(f"Starting RL-APSO Training with random swarm sizes for {num_episodes} episodes...")

    for ep in range(num_episodes):
        # Sample an integer swarm size in [5, 15] for this episode
        ep_num_particles = random.randint(5, 15)

        # Source is fixed; only swarm size varies across episodes
        state = env.reset(source_pos=source, num_particles=ep_num_particles)
        ep_reward = 0.0
        valid_actions = 0

        for t in range(max_iter):
            action, logprob = agent.select_action(state)

            next_state, reward, done, valid = env.step(action)
            if valid:
                valid_actions += 1

            agent.store(state, action, logprob, reward, done)

            state = next_state
            ep_reward += reward

            if done:
                break

        agent.update()
        rewards_history.append(ep_reward)

        if (ep + 1) % 10 == 0:
            avg_rew = np.mean(rewards_history[-10:])
            print(
                f"Episode {ep+1}/{num_episodes} | Avg Reward: {avg_rew:.4f} | "
                f"Valid Actions: {valid_actions}/{t+1} | Particles this episode: {ep_num_particles}"
            )

    # Save model trained with variable swarm sizes
    agent.save(os.path.join(current_dir, "latest_ppo_apso_random_particles.pth"))
    print("Model saved to latest_ppo_apso_random_particles.pth")

    # Save mean reward component statistics for offline analysis
    component_means = env.get_reward_component_means()
    np.savez(os.path.join(current_dir, "reward_component_means.npz"), **component_means)
    print("Saved reward component means to reward_component_means.npz")

    # Plot training curve
    try:
        plt.figure()
        plt.plot(rewards_history)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("RL-APSO Training Performance (Random Swarm Sizes)")
        plt.savefig("rl_apso_training_random_particles.png")
        print("Training plot saved to rl_apso_training_random_particles.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "benchmarks":
            evaluate_trained_rl_apso_on_benchmarks()
        elif arg in ("random_particles", "var_particles", "vary_particles"):
            run_rl_apso_training_random_particles()
        else:
            run_rl_apso_training()
    else:
        run_rl_apso_training()
