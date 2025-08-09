from src.env_portfolio import PortfolioEnv
import numpy as np

env = PortfolioEnv()

obs = env.reset()
done = False
total_reward = 0

while not done:
    action = np.random.dirichlet(np.ones(env.num_assets))  # random allocation
    obs, reward, done, info = env.step(action)
    total_reward += reward

print("Final portfolio value:", info["portfolio_value"])
print("Total reward:", total_reward)
