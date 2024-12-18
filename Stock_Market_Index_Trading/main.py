# Approach -----------------------------------------------------------------
# 1. Collect real-time stock data using the yfinance library.
# 2. Define a custom OpenAI Gym environment for stock trading.
# 3. Train a Dueling Deep Q-Network (DQN) model using Stable Baselines3.
# 4. Make daily trading predictions.
# 5. Send email alerts with the predicted actions.
# --------------------------------------------------------------------------

import numpy as np                  # Used for numerical computations.
import yfinance as yf               # Fetches historical stock data.
from stable_baselines3 import DQN   # A reinforcement learning algorithm to train the agent.
import gym                          # Used to create the custom trading environment.


class StockTradingEnv(gym.Env):
    # Custom Environment Definition
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.action_space = gym.spaces.Discrete(3)  # Action Space - Buy (0), Sell (1), Hold (2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(len(df.columns),), dtype=np.float32)     #  The current state (normalized features like Open, High, Low, etc.).
        self.reset()                # Resets the environment for a new episode.

    # Reset Method
    def reset(self):
        self.current_step = 0       # current_step: Points to the first row of stock data.
        self.total_profit = 0       # total_profit: Tracks cumulative profit.
        self.positions = []         # positions: Tracks buy positions.
        return self._next_observation() # Returns the first observation for the agent.

    # Observation 
    def _next_observation(self):
        return self.df.iloc[self.current_step].values   # Fetches the stock data for the current step as the agent's input.

    # Step Method
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0
        profit = 0
        transaction_cost = 0.001  # Define a transaction cost

        if action == 0:                             # Buy -  Adds the price to positions and deducts a transaction cost.
            self.positions.append(current_price)
            reward -= transaction_cost              # Deduct transaction cost for buying
        elif action == 1 and len(self.positions) > 0:  # Sell
            buy_price = self.positions.pop(0)
            profit = current_price - buy_price      # Calculates profit as current_price - buy_price.
            reward = profit - transaction_cost      # Reward is profit minus transaction cost
            self.total_profit += profit

        # Add a penalty for holding positions too long
        holding_penalty = 0.001 * len(self.positions)
        reward -= holding_penalty

        self.current_step += 1
        done = self.current_step == len(self.df) - 1    # Marks the episode as finished when the last data point is reached.
        obs = self._next_observation()

        return obs, reward, done, {'profit': profit}    # obs: Next observation, reward: Reward for the action, done: Whether the episode is over, info: Metadata (e.g., profit).

    def render(self, mode='human'):
        profit = self.total_profit
        print(f'Step: {self.current_step}, Profit: {profit}')
        
# Fetch historical data for Nifty 50
def fetch_data(ticker='^NSEI', start_date='2015-01-16', end_date='2024-12-18'):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna()
    data = (data - data.mean()) / data.std()  # Normalize the data
    return data

symbol = '^NSEI'
stock_data = fetch_data()
# Initialize the environment with new data
env = StockTradingEnv(stock_data)

# Train the Model from Scratch
model = DQN('MlpPolicy', env, verbose=1)        # Model: A Dueling Deep Q-Network is trained for 10,000 timesteps.
model.learn(total_timesteps=10000)

# Save the model for future use
model.save("dqn_stock_trading_model")           # Save Model: Saves the trained model to a file.

# Making Predictions - Uses the trained model to predict the best action for today's stock market state.    
def make_prediction(env, model):
    obs = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    actions_map = {0: 'Buy', 1: 'Sell', 2: 'Hold'}
    action = int(action)  # Convert the numpy array to an integer
    return actions_map[action]

# Make prediction
action = make_prediction(env, model)
print(f"Today's action for {symbol}: {action}")