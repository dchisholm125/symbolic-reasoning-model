import asyncio
import time
import json
import random
from typing import Dict, Any

class SRM_Nervous_System:
    """
    Layer 1 Core wrapped in an Asynchronous Event Loop.
    Proves that the AI can process high-speed environmental data (Sensory/Hot Path)
    while simultaneously holding a conversation (Speech/Cold Path) without blocking.
    """
    def __init__(self):
        # The central, shared memory of the AI. 
        # Continually updated by the environment, occasionally queried by language.
        self.state = {
            "market_params": {
                "SOL_USDC": {"current_price": 145.50, "threshold": 145.00}
            },
            "network_state": {"priority_fee": 10000},
            "temporal_flags": ["accumulate_sol"]
        }
        self.trades_executed = 0

    def process_symbol(self, input_symbol: Dict[str, Any]) -> Dict[str, Any]:
        """The deterministic logic gate processing both Hot and Cold symbols."""
        intent = input_symbol.get("intent_class")
        params = input_symbol.get("parameters", {})
        
        output = {"directive": "NO_ACTION", "state_snapshot": {}}

        # --- SENSORY PROCESSING (High Frequency) ---
        if intent == "update_tick":
            # Update internal worldview
            self.state["market_params"]["SOL_USDC"]["current_price"] = params["price"]
            self.state["network_state"]["priority_fee"] = params["fee"]
            
            # Logic reaction
            if params["price"] <= self.state["market_params"]["SOL_USDC"]["threshold"]:
                if params["fee"] < 50000:
                    self.trades_executed += 1
                    output["directive"] = "EXECUTE_BID"
                else:
                    output["directive"] = "HOLD_CONGESTION"

        # --- LANGUAGE PROCESSING (Low Frequency) ---
        elif intent == "query_state":
            # Taking a snapshot of the constantly moving state for Layer 3 to speak
            output["directive"] = "REPORT_STATUS"
            output["state_snapshot"] = {
                "current_price": self.state["market_params"]["SOL_USDC"]["current_price"],
                "current_fee": self.state["network_state"]["priority_fee"],
                "total_trades": self.trades_executed
            }

        return output

async def sensory_stream(srm_core: SRM_Nervous_System):
    """Simulates a high-speed WebSocket pushing data 10 times a second."""
    print(">> [SENSORY] Optic/Market stream connected. Ingesting ticks...")
    for _ in range(50): # Run for 5 seconds
        await asyncio.sleep(0.1)
        
        # Simulate market volatility
        mock_price = round(random.uniform(144.00, 146.00), 2)
        mock_fee = random.choice([12000, 15000, 80000]) # Occasional congestion spike
        
        tick_symbol = {
            "intent_class": "update_tick",
            "parameters": {"price": mock_price, "fee": mock_fee}
        }
        
        # Core processes this instantly in the background
        result = srm_core.process_symbol(tick_symbol)
        
        if result["directive"] == "EXECUTE_BID":
            print(f"   ⚡ [HOT PATH] Snipe Executed! Price: {mock_price}, Fee: {mock_fee}")

async def language_stream(srm_core: SRM_Nervous_System):
    """Simulates a human asking a question every 2 seconds."""
    await asyncio.sleep(1.5) # Wait a bit before asking the first question
    
    for _ in range(2):
        print("\n🗣️ [USER]: 'Hey, what is our current status?'")
        
        # Simulating Layer 4 sending the query symbol
        query_symbol = {
            "intent_class": "query_state",
            "parameters": {"target": "status"}
        }
        
        # Core takes a snapshot of the chaos
        result = srm_core.process_symbol(query_symbol)
        
        print(f"🧠 [LAYER 1]: Snapshot taken. Passing to Layer 3 (SLM)...")
        print(f"🤖 [LAYER 3]: 'We have executed {result['state_snapshot']['total_trades']} trades so far. "
              f"The current price is {result['state_snapshot']['current_price']} with fees at {result['state_snapshot']['current_fee']}.'\n")
        
        await asyncio.sleep(2.0)

async def main():
    srm = SRM_Nervous_System()
    
    # Run the high-speed sensory stream and the slow language stream concurrently
    await asyncio.gather(
        sensory_stream(srm),
        language_stream(srm)
    )
    print(">> [SYSTEM] Simulation Complete.")

if __name__ == "__main__":
    asyncio.run(main())
