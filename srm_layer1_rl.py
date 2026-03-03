import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from stable_baselines3 import DQN

class SRM_Layer1_Env(gym.Env):
    """
    The strict Neuro-Symbolic Training Environment.
    Airtight reward shaping: Rule-breaking is heavily punished, 
    and the agent is forced to learn both 'Take Profit' and 'Stop Loss' reflexes.
    """
    def __init__(self):
        super(SRM_Layer1_Env, self).__init__()
        
        self.action_space = spaces.Discrete(3) # 0: HOLD, 1: BUY, 2: SELL
        
        # [Current Price, Priority Fee, Position Held, Entry Price, Price Delta, Unrealized PnL]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -100.0, -1000.0], dtype=np.float32), 
            high=np.array([1000.0, 100000.0, 1.0, 1000.0, 100.0, 1000.0], dtype=np.float32),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        self.portfolio_cash = 1000.00
        self.position_held = 0.0
        self.entry_price = 0.0
        self.current_tick = 0
        self.time_step = 0 
        self.prev_price = 145.0
        self.current_price = 145.0
        self.price_delta = 0.0
        self.max_ticks = 500 
        
        self._generate_next_tick()
        return self._get_obs(), {}

    def _generate_next_tick(self):
        self.time_step += 1
        self.prev_price = self.current_price
        
        trend = 145.0 + 5.0 * math.sin(self.time_step * 0.1)
        noise = random.uniform(-0.2, 0.2)
        self.current_price = round(trend + noise, 2)
        
        self.price_delta = round(self.current_price - self.prev_price, 2)
        self.current_fee = random.choice([100, 500, 1000]) if random.random() > 0.2 else 80000
        self.fee_usd = self.current_fee / 10000.0 

    def _get_obs(self):
        unrealized_pnl = round(self.current_price - self.entry_price, 2) if self.position_held == 1.0 else 0.0
        return np.array([
            self.current_price, self.current_fee, self.position_held, 
            self.entry_price, self.price_delta, unrealized_pnl
        ], dtype=np.float32)

    def step(self, action):
        self.current_tick += 1
        reward = 0.0
        info_string = "HOLD (Ignored tick)"

        unrealized_pnl = (self.current_price - self.entry_price) if self.position_held == 1.0 else 0.0

        if action == 1: # BUY
            if self.position_held == 0:
                if self.current_fee >= 50000:
                    reward = -2.0 
                    info_string = "REJECTED BUY (High Fee)"
                else:
                    self.position_held = 1.0
                    self.entry_price = self.current_price
                    self.portfolio_cash -= self.fee_usd
                    reward = 0.0 
                    info_string = "EXECUTED BUY"
            else:
                reward = -5.0 # Unforgivable penalty to stop loophole logic
                info_string = "INVALID BUY"

        elif action == 2: # SELL
            if self.position_held == 1.0:
                self.position_held = 0.0
                net_profit = (self.current_price - self.entry_price) - self.fee_usd
                
                reward = net_profit * 10.0 
                self.portfolio_cash += net_profit
                self.entry_price = 0.0
                info_string = f"EXECUTED SELL (PnL: ${net_profit:.2f})"
            else:
                reward = -5.0 # Unforgivable penalty
                info_string = "INVALID SELL"

        elif action == 0: # HOLD
            reward = -0.01 
            
            if self.position_held == 1.0:
                info_string = "HOLDING POSITION"
                
                # 1. Take Profit Reflex (Bag-Holder penalty)
                if unrealized_pnl > 0.5 and self.price_delta < 0:
                    reward = -2.0 # Sell now or suffer!
                
                # 2. Stop Loss Reflex (The missing logic)
                elif unrealized_pnl < -0.5:
                    reward = -2.0 # Cut your losses now or suffer!

        self._generate_next_tick()
        done = self.current_tick >= self.max_ticks
        
        info = {"status": info_string, "portfolio": self.portfolio_cash}
        return self._get_obs(), reward, done, False, info

if __name__ == "__main__":
    print("🧠 Initializing Airtight SRM Layer 1 Gym Environment...")
    env = SRM_Layer1_Env()
    
    # Using a slightly larger buffer and 250k steps to ensure the logic locks in
    model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=100000, exploration_fraction=0.6, verbose=0)
    print("🧬 Training Neuro-Symbolic Agent (DQN) for 250,000 steps...")
    model.learn(total_timesteps=250000) 
    print("✅ Training Complete.")
    print("-" * 75)

    print("⚡ Executing Trained Agent on Live Sensory Stream (Hot Path):")
    obs, _ = env.reset()
    for i in range(40): 
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        
        price = obs[0]
        fee = obs[1]
        delta = obs[4]
        pnl = obs[5]
        
        pnl_str = f"| PnL: ${pnl:+.2f} " if obs[2] == 1.0 else "| PnL: $0.00  "
        print(f"Tick {i+1:02d} | Price: ${price:.2f} (Δ {delta:+.2f}) | Fee: {fee:05.0f} {pnl_str}| Intent: {info['status']:<28} | Portfolio: ${info['portfolio']:.2f}")