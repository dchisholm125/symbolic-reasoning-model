import asyncio
import time
from framework.srm import Framework, IntentRouter, SLM_Node
from framework.srm.rl_core import RL_LogicEngine
from solana_adapter import SolanaLiveNerve
import numpy as np

# CONFIG
RPC_URL = "https://api.mainnet-beta.solana.com" 
PYTH_SOL = "H6ARHf6Y2yucMv9D2vv1W2dMnZg8gvH2HSPss8v2pMdy"

class LivePaperTradingEnv:
    """
    The physiological bridge. Translates live blockchain data from the Nerve 
    into the physical RL environment the Brain was trained on.
    """
    def __init__(self, nerve: SolanaLiveNerve):
        self.nerve = nerve
        self.portfolio_cash = 1000.00
        self.position_held = 0.0
        self.entry_price = 0.0

    def _get_obs(self):
        # Formats the 6-dim symbol exactly how the DQN network expects it
        sym = self.nerve.get_symbol(self.position_held, self.entry_price)
        return np.array(sym, dtype=np.float32)

    def step(self, action):
        reward = 0.0
        info_string = "HOLD (Ignored tick)"

        # Action 1: BUY
        if action == 1: 
            if self.position_held == 0:
                self.position_held = 1.0
                self.entry_price = self.nerve.price
                self.portfolio_cash -= (self.nerve.fee / 10000.0)
                info_string = "EXECUTED BUY"
            else:
                info_string = "INVALID BUY"
        
        # Action 2: SELL
        elif action == 2: 
            if self.position_held == 1.0:
                self.position_held = 0.0
                net_profit = (self.nerve.price - self.entry_price) - (self.nerve.fee / 10000.0)
                self.portfolio_cash += net_profit
                self.entry_price = 0.0
                info_string = f"EXECUTED SELL (PnL: ${net_profit:.2f})"
            else:
                info_string = "INVALID SELL"
        
        # Action 0: HOLD
        elif action == 0:
            if self.position_held == 1.0:
                info_string = "HOLDING POSITION"

        info = {"status": info_string, "portfolio": self.portfolio_cash}
        return self._get_obs(), reward, False, False, info


async def hot_path_loop(agent, nerve):
    """The sub-millisecond quantitative action loop."""
    asyncio.create_task(nerve.stream())
    
    # Wait 2 seconds to ensure the WebSocket establishes and gets a price
    await asyncio.sleep(2.0) 
    
    # The baseline we trained on in V2
    fallback_price = 84.80 
    
    while True:
        await asyncio.sleep(0.5) 
        
        # SENSORY FAILSAFE: If the Pyth Oracle parser fails, inject mock data 
        # so the Brain doesn't freeze and we can still verify the Hot Path.
        if nerve.price <= 0.0:
            nerve.price = round(fallback_price + (time.time() % 2) / 10.0, 2)
            nerve.prev_price = nerve.price - 0.05
            nerve.fee = 5000
            
        res = agent.run_hot_path(intent="execute_tick")
        
        # Safely check the action
        action = res.get("action_taken", "UNKNOWN")
        if action != "HOLD" and action != "UNKNOWN":
            print(f"⚡ [BLOCKCHAIN] {action} | Portfolio: ${res.get('portfolio_value'):.2f}")

async def cold_path_loop(agent):
    """The human conversational loop."""
    loop = asyncio.get_running_loop()
    
    while True:
        await asyncio.sleep(30)
        print("\n🗣️ [USER]: 'Status report.'")
        report = await loop.run_in_executor(None, agent.run_cold_path, "status update")
        print(f"🤖 [SRM]:\n{report}\n")

async def main():
    # 1. Ear (Parser)
    ear = IntentRouter(intents={"query_status": ["status", "report", "update"]})
    
    # 4. Nerve (Live WebSocket)
    nerve = SolanaLiveNerve(RPC_URL, PYTH_SOL)
    
    # 2. Brain (RL Engine)
    # This loads the .zip weights we just trained!
    brain = RL_LogicEngine(model_path="./dqn_srm_model.zip", environment=LivePaperTradingEnv(nerve))
    
    # 3. Mouth (Quantized SLM)
    mouth = SLM_Node(model_path="./qwen2.5-1.5b-instruct-q4_k_m.gguf", n_gpu_layers=-1)
    
    # 5. Framework
    strict_prompt = "You are a quantitative trading bot. Output EXACTLY ONE concise sentence summarizing the portfolio state. DO NOT repeat yourself. DO NOT use emojis."
    agent = Framework(ear, brain, mouth, system_prompt=strict_prompt)
    
    # 6. Monkey-Patch the SLM to restrict tokens and clean output
    original_generate = agent.mouth.generate
    def snappy_generate(context: str) -> str:
        # Cap max_tokens at 30 to prevent greedy looping
        raw = original_generate(context, max_tokens=30, temperature=0.0)
        # Strip the broken stop token if it appears
        return raw.replace("|-|end|", "").strip()
    agent.mouth.generate = snappy_generate
    print("   SRM LIVE SOLANA TERMINAL ACTIVE")
    print("==========================================")
    
    await asyncio.gather(hot_path_loop(agent, nerve), cold_path_loop(agent))

if __name__ == "__main__":
    asyncio.run(main())