import numpy as np
import matplotlib.pyplot as plt


def get_episode_trajectory(agent, env, agent_type="tabular"):
    """
    Run one episode and return price/inventory/days trajectories.
    agent_type: 'tabular' (uses .act) or 'dqn' (uses .act_greedy)
    """
    act_fn = agent.act if agent_type == "tabular" else agent.act_greedy

    obs, info = env.reset()
    prices, inventory, days = [], [], []
    done = False

    while not done:
        prices.append(act_fn(obs))
        inventory.append(obs[0])
        days.append(obs[1])
        obs, reward, terminated, truncated, info = env.step(prices[-1])
        done = terminated or truncated

    return {"prices": prices, "inventory": inventory, "days": days}


def plot_multi_season_trajectories(agent, env, n_seasons=5, agent_type="dqn",
                                    title="Agent Price Trajectory Across Seasons",
                                    save_path=None):
    """Plot price trajectories across multiple sample seasons (episodes)."""
    plt.figure(figsize=(10, 6))

    for season in range(n_seasons):
        traj = get_episode_trajectory(agent, env, agent_type=agent_type)
        plt.plot(traj["prices"], marker='o', alpha=0.7, label=f"Season {season+1}")

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Price Level")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from pricing_env import PricingEnv
    from dqn_agent import DQNAgent

    env = PricingEnv()
    agent = DQNAgent()  # untrained, just testing the plotting utility works

    traj = get_episode_trajectory(agent, env, agent_type="dqn")
    print("Trajectory length:", len(traj["prices"]))
    print("Sample prices:", traj["prices"][:5])