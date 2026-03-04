import asyncio
import time
from framework.srm import Framework, IntentRouter, SLM_Node
from framework.srm.rl_core import RL_LogicEngine

# NOTE: You must import your trained SRM_Layer1_Env here from your previous training script
# from srm_layer1_rl import SRM_Layer1_Env 
# For demonstration, we use a mocked environment to allow the script to run seamlessly
class MockEnv:
    portfolio_cash = 1050.25
    position_held = 1.0
    def _get_obs(self): return [145.0, 100, 1.0, 142.0, 0.5, 3.0]
    def step(self, action): return self._get_obs(), 3.0, False, False, {"status": "EXECUTED SELL", "portfolio": 1053.25}

async def sensory_hot_stream(agent: Framework):
    """
    Simulates a high-speed WebSocket pushing RPC data to the agent.
    This runs continuously in the background.
    """
    print(">> [SENSORY] WebSocket connected. Ingesting market ticks...\n")
    for i in range(1, 15): 
        await asyncio.sleep(0.5) # Fast tick rate
        
        # Bypasses the "Ear" completely, going straight to the RL Brain
        hot_path_result = agent.run_hot_path(intent="execute_tick")
        
        action = hot_path_result.get("action_taken", "HOLD")
        if action != "HOLD":
            print(f"   ⚡ [HOT PATH TRIGGER] Agent executed: {action} | Portfolio: ${hot_path_result.get('portfolio_value')}")

async def language_cold_stream(agent: Framework):
    """
    Simulates a human intervening and asking a question while the bot trades.
    """
    await asyncio.sleep(2.0) # Wait a few ticks before interrupting
    
    question = "Hey, give me an update on our current trading status."
    print(f"\n🗣️ [USER]: '{question}'")
    
    start_time = time.perf_counter()
    
    # Run the full 4-layer framework
    response = agent.run_cold_path(question)
    
    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"🤖 [LAYER 3 SLM] ({elapsed:.2f}ms latency):")
    print(f"   \"{response}\"\n")

async def main():
    print("==================================================")
    print("INITIALIZING SRM MASTER ASSEMBLY...")
    print("==================================================")
    
    # 1. Initialize the Ear (ONNX disabled for fast mock, will use primitive fallback)
    ear = IntentRouter(intents={"query_status": ["status", "update", "how are we doing"]})
    
    # 2. Initialize the Brain (Loading our trained Reinforcement Learning DQN)
    brain = RL_LogicEngine(model_path="./dqn_srm_model", environment=MockEnv())
    
    # 3. Initialize the Mouth (Will fall back to simulated response if .gguf is missing)
    mouth = SLM_Node(model_path="./qwen2.5-0.5b-instruct-q4_k_m.gguf")
    
    # 4. Orchestrate
    agent = Framework(ear=ear, brain=brain, mouth=mouth)
    print("==================================================\n")
    
    # Run dual-stream cognition
    await asyncio.gather(
        sensory_hot_stream(agent),
        language_cold_stream(agent)
    )
    print(">> [SYSTEM] Master Assembly Run Complete.")

if __name__ == "__main__":
    asyncio.run(main())
