import sys
sys.path.append('../src')
import gymnasium as gym
import pytest


@pytest.fixture
def env():
    e = gym.make("CartPole-v1")
    yield e
    e.close()


def test_reset_returns_valid_initial_state(env):
    obs, info = env.reset()
    assert obs is not None
    assert env.observation_space.contains(obs)


def test_step_returns_valid_reward(env):
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, (int, float))


def test_terminal_condition_reached(env):
    obs, info = env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert done is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])