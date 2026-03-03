import time
import json
from typing import Dict, Any

class SRM_Input_Parser:
    """
    Layer 4: The 'Ear'. 
    Translates messy natural language into strict Phase 1 JSON Symbols.
    In a production 6GB VRAM environment, this uses a tiny Encoder-only model 
    (like MiniLM or BERT) or Grammar-Constrained generation. 
    """
    def __init__(self):
        # Mocking a fast embedding space / classifier for demonstration.
        # This represents the "trained" boundaries of what the system understands.
        self.intent_map = {
            "query_state": [
                "why did we stop buying", 
                "what is the status", 
                "give me an update",
                "why are we holding"
            ],
            "execute_action": [
                "buy now", 
                "force execution", 
                "override and buy"
            ]
        }

    def parse_natural_language(self, user_text: str) -> Dict[str, Any]:
        """
        Takes raw English, maps it to the closest intent in microseconds,
        and strictly enforces the Phase 1 Output Schema.
        """
        start_time = time.perf_counter()
        
        user_text = user_text.lower()
        mapped_intent = "invalid_input"
        
        # In production, this loop is replaced by a fast vector similarity search 
        # (e.g., Cosine similarity using a 30MB ONNX embedding model).
        for intent, phrases in self.intent_map.items():
            if any(phrase in user_text for phrase in phrases):
                mapped_intent = intent
                break
                
        # Construct the rigid Phase 1 Input Symbol
        input_symbol = {
            "intent_class": mapped_intent,
            "parameters": {}
        }
        
        # Entity extraction (e.g., finding the 'target')
        if mapped_intent == "query_state":
            input_symbol["parameters"]["target"] = "SOL_accumulation_status"

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        input_symbol["execution_time_ms"] = round(elapsed_ms, 4)
        
        return input_symbol

# --- Quick Test Execution ---
if __name__ == "__main__":
    parser = SRM_Input_Parser()
    
    messy_user_input = "Yo, why did we stop buying?"
    print(f"User Said: '{messy_user_input}'")
    
    # Layer 4 compresses language into a symbol
    symbol = parser.parse_natural_language(messy_user_input)
    
    print(f"\nLayer 4 parsed text in {symbol.get('execution_time_ms')}ms")
    print("Generated Phase 1 Input Symbol:")
    print(json.dumps(symbol, indent=2))
