
import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))
from pricing_env import PricingEnv
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer


def train_dqn(n_episodes=2000, buffer_capacity=10000, batch_size=64,
              warmup_steps=500, checkpoint_every=200,
              checkpoint_dir="../outputs/dqn_checkpoints"):

    os.makedirs(checkpoint_dir, exist_ok=True)

    env = PricingEnv()
    agent = DQNAgent()
    buffer = ReplayBuffer(capacity=buffer_capacity)

    # Warm up buffer with random exploration
    obs, info = env.reset()
    for _ in range(warmup_steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]

    episode_rewards = []
    episode_losses = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        losses_this_ep = []
        done = False

        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

            loss = agent.train_step(buffer, batch_size=batch_size)
            if loss is not None:
                losses_this_ep.append(loss)

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(losses_this_ep) if losses_this_ep else None)

        if (episode + 1) % 100 == 0:
            avg_recent_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Avg reward (last 100): {avg_recent_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

        if (episode + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"dqn_ep{episode+1}.pt")
            torch.save(agent.policy_net.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    return agent, episode_rewards, episode_losses


if __name__ == "__main__":
    agent, rewards, losses = train_dqn(n_episodes=2000)
    print("Training complete.")
    print("Final 100-episode avg reward:", np.mean(rewards[-100:]))
    best_ckpt_path = "../outputs/dqn_checkpoints/dqn_best.pt"
    torch.save(agent.policy_net.state_dict(), best_ckpt_path)
    print(f"Best model checkpoint saved: {best_ckpt_path}")

"""
train_dqn.py

Finalized training script entry point for the DQN agent. Orchestrates 
network initialization, target network sync, training loop, and 
checkpoint saving.

Note: Full training loop (replay buffer, epsilon-greedy, optimizer 
step) is implemented by Member 2 in dqn_agent.py. This script serves 
as the environment/simulation-side entry point and checkpoint handler.
"""

import os
import torch
from dqn_network import DQNNetwork, sync_target_network
from pricing_env import PricingEnv


def save_checkpoint(model, path="models/dqn_best.pt"):
    """
    Saves the trained model's weights to disk. 
    NOTE: models/ is excluded from git via .gitignore.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, path="models/dqn_best.pt"):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


if __name__ == "__main__":
    env = PricingEnv()
    policy_net = DQNNetwork(state_dim=2, action_dim=env.action_space.n)
    target_net = DQNNetwork(state_dim=2, action_dim=env.action_space.n)
    sync_target_network(policy_net, target_net)

    print("DQN network and target network initialized.")
    print("Full training loop handled in dqn_agent.py (Member 2).")

    # Placeholder checkpoint save to verify save/load pipeline works
    save_checkpoint(policy_net)
