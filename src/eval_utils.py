import numpy as np


def run_episodes(agent, env, n_episodes=500):
    """
    Simulate an agent against an env for n_episodes and return
    revenue statistics.

    Assumes agent has an .act(obs) method and env follows the
    Gymnasium API.
    """
    episode_rewards = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)

    stats = {
        "episodes": n_episodes,
        "mean_revenue": float(np.mean(episode_rewards)),
        "std_revenue": float(np.std(episode_rewards)),
        "min_revenue": float(np.min(episode_rewards)),
        "max_revenue": float(np.max(episode_rewards)),
        "all_rewards": episode_rewards,
    }
    return stats


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    import gymnasium as gym
    from random_agent import RandomAgent

    env = gym.make("CartPole-v1")
    agent = RandomAgent(env.action_space)

    results = run_episodes(agent, env, n_episodes=50)
    print("Mean revenue:", results["mean_revenue"])
    print("Std revenue:", results["std_revenue"])
    env.close()