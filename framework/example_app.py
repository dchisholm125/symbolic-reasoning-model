from srm import Framework, IntentRouter, LogicEngine, SLM_Node

# 1. Define the Ear (Layer 4)
ear = IntentRouter(intents={
    "query_status": ["what's up", "status?", "give me an update"]
})

# 2. Define the Brain (Layer 1)
class MyLogic(LogicEngine):
    def process(self, intent: str, params=None):
        if intent == "query_status":
            return {"status": "systems nominal", "cpu_temp": 45, "SOL_inventory": 34.2}
        return None

# 3. Define the Mouth (Layer 3)
mouth = SLM_Node(model_path="./qwen2.5-0.5b-instruct-q4_k_m.gguf")

# 4. Wire the Framework constraints
agent = Framework(
    ear=ear, 
    brain=MyLogic(), 
    mouth=mouth,
    system_prompt="You are a trading agent. Review the JSON state and explain it clearly."
)

# --- Execution ---
print("Sending English Text: 'Hey bot, what's our status right now?'")
print("...")
response = agent.run("Hey bot, what's our status right now?")
print(f"Agent Output: {response}")
