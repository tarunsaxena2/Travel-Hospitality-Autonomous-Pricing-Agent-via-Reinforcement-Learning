import random


class RandomAgent:
    """Baseline agent that samples a random valid action at each step."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state=None):
        return self.action_space.sample()

    def __repr__(self):
        return f"RandomAgent(action_space={self.action_space})"


if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env.action_space)
    print(agent)
    print("Sample action:", agent.act())