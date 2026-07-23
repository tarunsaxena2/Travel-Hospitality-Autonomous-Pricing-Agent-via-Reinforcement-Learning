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