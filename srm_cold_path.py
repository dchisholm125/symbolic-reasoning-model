import json
import time
from typing import Dict, Any
from srm_core_quant import Quant_SRM_Core

# Mocking the local SLM to demonstrate Layer 3 functionality
class MockSLM:
    def __init__(self):
        self.latency_ms = 450.0  # Simulate 450ms token generation delay

    def generate(self, prompt: str) -> str:
        """Simulates an LLM taking a rigid prompt and generating text."""
        time.sleep(self.latency_ms / 1000.0)
        
        # In a real setup, this would be an API call to local llama.cpp or vLLM
        # We manually parse the prompt structure to show how an LLM would complete it based on the new network variables.
        
        if "REPORT_STATUS" in prompt:
            if "current_priority_fee': 80000" in prompt or "current_priority_fee': 85000" in prompt:
                 return "We are pausing accumulation of SOL. Although the price of $144.50 is below our target threshold, priority fees have spiked to 85,000 micro-lamports indicating network congestion. We are holding our position to avoid MEV."
            elif "current_priority_fee': 12000" in prompt:
                 return "We are actively accumulating SOL. The current price of $144.50 has crossed our accumulation threshold, and priority fees are currently stable at 12,000 micro-lamports. We are executing a limit bid."
        
        return "I am unable to interpret the requested directive."

class FormulationBridge:
    def __init__(self):
        self.slm = MockSLM()
        
    def generate_report(self, output_symbol: Dict[str, Any]) -> str:
        """
        Layer 2 (The Bridge): Formats the raw JSON from Layer 1 into a strict prompt.
        Layer 3 (The Mouth): Passes the prompt to the SLM to generate a natural response.
        """
        directive = output_symbol.get("directive_code")
        state_delta = output_symbol.get("state_delta", {})
        
        # 1. Layer 2: Prompt Formulation
        # The SLM context window is strictly this template + the state variables. No chat history.
        system_prompt = "You are a quantitative trading reporting agent for a Solana bot. State the trading operation facts clearly and concisely based ONLY on the provided variables. Explain WHY the current directive is happening using the metrics provided. Do not invent metrics."
        
        instruction = f"Directive: {directive}. Data: {state_delta}. Explain what this means to the user."
        
        full_prompt = f"System: {system_prompt}\nUser: {instruction}\nAgent: "
        
        print("\n--- Layer 2: SLM Context Window (Ultra-Small) ---")
        print(full_prompt)
        
        # 2. Layer 3: Execute Speech
        print("\n--- Layer 3: Generating Articulation ---")
        start_time = time.perf_counter()
        
        response = self.slm.generate(full_prompt)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        print(f"SLM Generation Time: {round(elapsed_ms, 2)}ms")
        return response

if __name__ == "__main__":
    # Initialize the complete Cold Path pipeline
    core = Quant_SRM_Core()
    bridge = FormulationBridge()
    
    # 1. Simulate a fast price tick under normal network conditions (Hot Path)
    tick_symbol_normal = {"intent_class": "update_tick", "parameters": {"pair": "SOL_USDC", "price": 144.50, "priority_fee": 12000}}
    print(f"Normal Conditions Check: {core.process_symbol(tick_symbol_normal)['directive_code']}")

    # 2. Query the state to see what it says
    query_symbol = {
        "intent_class": "query_state",
        "parameters": {"target": "SOL_accumulation_status"}
    }
    
    print("\n--- Scenario 1: Normal Priority Fees ---")
    output_symbol = core.process_symbol(query_symbol)
    print(bridge.generate_report(output_symbol))

    # 3. Simulate a sudden spike in congestion (MEV activity/high priority fees)
    tick_symbol_spiked = {"intent_class": "update_tick", "parameters": {"pair": "SOL_USDC", "price": 144.50, "priority_fee": 80000}}
    print(f"\nCongestion Spike Check: {core.process_symbol(tick_symbol_spiked)['directive_code']}")

    print("\n--- Scenario 2: High Priority Fees (Congestion) ---")
    output_symbol_spiked = core.process_symbol(query_symbol)
    print(bridge.generate_report(output_symbol_spiked))
