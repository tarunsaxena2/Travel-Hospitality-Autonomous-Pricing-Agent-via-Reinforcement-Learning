from baseline_agents import TimeBasedDiscountAgent
from pricing_env import PricingEnv

env = PricingEnv()
agent = TimeBasedDiscountAgent()

obs, info = env.reset()
agent.reset()

for step in range(5):
    action = agent.act(obs)
    print(f"Step {step}: action={action}")
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break