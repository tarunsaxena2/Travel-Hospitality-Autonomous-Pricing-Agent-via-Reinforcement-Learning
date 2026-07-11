from pricing_env import PricingEnv
import numpy as np

env = PricingEnv()
num_episodes = 100
crashes = 0

for ep in range(num_episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        except Exception as e:
            crashes += 1
            print(f"Episode {ep} crashed: {e}")
            break

print(f"Completed {num_episodes} episodes with {crashes} crashes.")