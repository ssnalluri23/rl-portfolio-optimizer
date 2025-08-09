import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, data_path='data/processed_data.csv', initial_cash=10000):
        super(PortfolioEnv, self).__init__()

        self.data = pd.read_csv(data_path, index_col=0)
        self.num_assets = 5
        self.initial_cash = initial_cash
        self.current_step = 0

        # Define action space: allocation weights per asset (continuous between 0–1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)

        # Observation = state vector (technical indicators, prices)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.portfolio = np.zeros(self.num_assets)
        self.prev_value = self.initial_cash  # If using percent return
        obs = self._get_observation()
        return obs, {}  # ✅ Return (obs, info)


    def _get_observation(self):
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)  # ✅ Clean!
        return obs


    def step(self, action):
        if np.sum(action) == 0:
            action = np.ones_like(action) / len(action)
        else:
            action = action / np.sum(action)

        prices = self.data.iloc[self.current_step, :self.num_assets].values
        portfolio_value = self.cash + np.sum(self.portfolio * prices)

        # Rebalance portfolio
        new_allocation = portfolio_value * action
        self.portfolio = new_allocation / prices
        self.cash = 0  # fully invested

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False  # You can add max_steps logic later if desired

        reward = (portfolio_value - self.prev_value) / self.prev_value
        self.prev_value = portfolio_value
        next_obs = self._get_observation()
        info = {"portfolio_value": portfolio_value}

        return next_obs, reward, terminated, truncated, info

