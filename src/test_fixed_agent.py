from baseline_agents import FixedPriceAgent
from pricing_env import PricingEnv

env = PricingEnv()
agent = FixedPriceAgent(price_level_index=5)

obs, info = env.reset()
for _ in range(5):
    action = agent.act(obs)
    assert action == 5, "FixedPriceAgent should always return same action"
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

print("FixedPriceAgent test passed: action stays constant.")