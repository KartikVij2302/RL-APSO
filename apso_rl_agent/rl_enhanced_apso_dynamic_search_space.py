import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

# project-specific imports (ensure these modules are on sys.path)
from .rl_apso_dynamic_swarm_size import RLAPSOEnv
from .PPO import PPOAgent


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


# ensure current directory is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def run_rl_apso_training_variable_arena():
    set_global_seed(42)

    # Training hyperparams (kept same as your previous script)
    state_dim = 8
    action_dim = 4
    lr = 1e-4
    agent = PPOAgent(state_dim, action_dim, lr=lr)

    num_episodes = 12000
    max_iter = 400

    rewards_history = []

    # Create models directory if missing
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(
        f"Starting RL-APSO Training (variable arena size) for {num_episodes} episodes..."
    )

    for ep in range(num_episodes):
        # Sample arena size y in [10, 100] meters (float or int; use int for simplicity)
        y = random.randint(10, 100)

        # Define square bounds [0, y] x [0, y]
        lo = np.array([0.0, 0.0])
        hi = np.array([float(y), float(y)])

        # Place source at the center of the square
        source = np.array([y / 2.0, y / 2.0])

        # Sample number of particles for this episode
        ep_num_particles = random.randint(5, 30)

        # Re-create environment for this episode so bounds & swarm size are applied cleanly
        env = RLAPSOEnv(
            source_pos=source,
            bounds=(lo, hi),
            num_particles=ep_num_particles,
            max_iter=max_iter,
        )

        # (If your RLAPSOEnv.reset accepts source_pos/num_particles kwargs, you could reuse env.
        #  Here we recreate env to be robust to different signatures.)

        # Reset environment and get initial state
        state = env.reset()
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

        # PPO update after each episode (same as before)
        agent.update()
        rewards_history.append(ep_reward)

        if (ep + 1) % 10 == 0:
            avg_rew = np.mean(rewards_history[-10:])
            print(
                f"Episode {ep+1}/{num_episodes} | Avg Reward: {avg_rew:.4f} | "
                f"Valid Actions: {valid_actions}/{t+1} | Particles: {ep_num_particles} | Arena: {y}x{y}"
            )

    # Save model
    save_path = os.path.join(models_dir, "latest_ppo_apso_variable_arena.pth")
    agent.save(save_path)
    print(f"Model saved to {save_path}")

    # Plot training curve
    try:
        plt.figure()
        plt.plot(rewards_history)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("RL-APSO Training Performance (Variable Arena Size)")
        plt.grid(True)
        out_plot = os.path.join(current_dir, "rl_apso_training_variable_arena.png")
        plt.savefig(out_plot)
        plt.close()
        print(f"Training plot saved to {out_plot}")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    run_rl_apso_training_variable_arena()
