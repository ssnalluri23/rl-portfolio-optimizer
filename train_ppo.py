from stable_baselines3 import PPO
from src.env_portfolio import PortfolioEnv

# Instantiate your custom environment
env = PortfolioEnv()

# Create the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_trading_agent")
print("✅ Training complete. Model saved as 'ppo_trading_agent.zip'")

