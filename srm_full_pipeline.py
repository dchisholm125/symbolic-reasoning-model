import time
import json
from srm_layer4_parser import SRM_Input_Parser
from srm_core_quant import Quant_SRM_Core
from srm_cold_path import FormulationBridge

def run_cognitive_cycle(user_text: str, core: Quant_SRM_Core):
    print("\n=======================================================")
    print(f"STARTING COGNITIVE CYCLE: '{user_text}'")
    print("=======================================================\n")
    
    # Initialize Layers
    parser = SRM_Input_Parser()     # Layer 4
    bridge = FormulationBridge()    # Layers 2 & 3
    
    total_start = time.perf_counter()

    # --- 1. The Ear (Layer 4) ---
    print("[1] Layer 4: Parsing Natural Language...", end=" ")
    input_symbol = parser.parse_natural_language(user_text)
    print(f"({input_symbol.pop('execution_time_ms', 0)}ms)")
    print(f"    -> Extracted Intent: {input_symbol['intent_class']}")
    
    # --- 2. The Brain (Layer 1) ---
    print("\n[2] Layer 1: Core Logic Execution...", end=" ")
    output_symbol = core.process_symbol(input_symbol)
    print(f"({output_symbol['execution_time_ms']}ms)")
    print(f"    -> Deterministic Directive: {output_symbol['directive_code']}")
    
    # --- 3. The Bridge & The Mouth (Layers 2 & 3) ---
    print("\n[3] Layers 2 & 3: Bridging and SLM Generation...")
    # Passing the output symbol to the bridge handles formatting and execution of the SLM
    final_response = bridge.generate_report(output_symbol)
    
    total_elapsed = (time.perf_counter() - total_start) * 1000
    print("\n=======================================================")
    print(f"FINAL BOT RESPONSE (Total Latency: {total_elapsed:.2f}ms):")
    print(f"\"{final_response}\"")
    print("=======================================================")

if __name__ == "__main__":
    # Load the Core System (Layer 1)
    quant_core = Quant_SRM_Core()
    
    # Simulate the Network state before the user asks a question
    # Let's say we had a network spike right before they asked:
    print("--- Simulating Hot Path Tick (High Network Fees) ---")
    quant_core.process_symbol(
        {"intent_class": "update_tick", "parameters": {"pair": "SOL_USDC", "price": 144.50, "priority_fee": 85000}}
    )
    
    # Now run the full end-to-end cognitive cycle triggered by a messy string
    messy_user_query = "Yo, why did we stop buying?"
    run_cognitive_cycle(messy_user_query, quant_core)
