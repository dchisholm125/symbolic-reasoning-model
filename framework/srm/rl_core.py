from typing import Dict, Any
try:
    from stable_baselines3 import DQN
    import numpy as np
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    
from .core import LogicEngine

class RL_LogicEngine(LogicEngine):
    """
    Wraps a trained Reinforcement Learning agent (like our Layer 1 PnL Environment) 
    inside the strict LogicEngine boundary.
    
    This fulfills the SRM architecture step where a trained neuro-symbolic agent 
    acts as the core logic processor, cleanly isolated from language processing.
    """
    def __init__(self, model_path: str, environment=None):
        self.model_path = model_path
        self.env = environment
        
        if HAS_SB3:
            try:
                # Loads the pre-trained weights compiled from 250,000 steps of Monte Carlo Tree Search
                self.model = DQN.load(model_path)
                self.is_loaded = True
                print(f"[RL_LogicEngine] Successfully loaded brain weights from {model_path}.")
            except Exception as e:
                print(f"[RL_LogicEngine] Could not load model: {e}. Running in passive mode.")
                self.is_loaded = False
        else:
            print("[RL_LogicEngine] WARNING: stable_baselines3 not installed. Running in passive mode.")
            self.is_loaded = False
            
        self.action_dictionary = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def process(self, intent: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        The Framework automatically hands the pure JSON symbol to the Brain here.
        We map that symbol to the neural network to execute the optimal state transition.
        """
        # If the intent is merely a human asking a question, read the state without acting.
        if intent == "query_status":
            if self.env:
                return {
                    "system_status": "Neural Network Active",
                    "portfolio_value": round(self.env.portfolio_cash, 2),
                    "current_position_held": bool(self.env.position_held == 1.0)
                }
            return {"system_status": "Active", "portfolio_value": "Unknown"}

        # If the intent is an environmental tick (Hot Path), run the Neural Network!
        elif intent == "execute_tick":
            if not self.is_loaded or not self.env:
                return {"error": "Cannot execute tick. Brain or Environment missing."}
                
            # 1. Take a snapshot of reality
            observation = self.env._get_obs()
            
            # 2. Feed it through the trained weights
            action, _states = self.model.predict(observation, deterministic=True)
            
            # 3. Execute the physical state transition
            obs, reward, done, truncated, info = self.env.step(action)
            
            # 4. Return the pure factual result of the logic computation
            return {
                "action_taken": self.action_dictionary.get(int(action), "UNKNOWN"),
                "net_profit_generated": round(reward, 2),
                "portfolio_value": round(info.get("portfolio", 0.0), 2),
                "internal_reasoning": info.get("status", "")
            }
            
        # Fallback for unrecognized intents
        return {"error": f"LogicEngine does not recognize intent symbol: {intent}"}
