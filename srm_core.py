import time
import json
from typing import Dict, Any

class SRM_Core:
    def __init__(self):
        # Initialize SystemState (The persistent Layer 1 memory)
        self.state = {
            "entities": {
                "player": {
                    "id": "player",
                    "type": "agent",
                    "attributes": {
                        "inventory": ["rusted_key"]
                    }
                },
                "iron_gate": {
                    "id": "iron_gate",
                    "type": "environment",
                    "attributes": {
                        "is_locked": True,
                        "required_key": "rusted_key"
                    }
                }
            },
            "temporal_flags": []
        }

    def process_input(self, input_symbol: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes an InputSymbol, evaluates the logic against SystemState, 
        updates the state, and returns the OutputSymbol in microseconds.
        """
        start_time = time.perf_counter()
        
        intent = input_symbol.get("intent_class")
        target = input_symbol.get("target_entity")
        params = input_symbol.get("parameters", {})
        
        output_symbol = {
            "logic_status": "rejected",
            "directive_code": "UNKNOWN_ERROR",
            "state_delta": {}
        }

        # Handle Action: Open Door
        if intent == "execute_action" and params.get("action") == "open":
            entity = self.state["entities"].get(target)
            
            if not entity or entity["type"] != "environment":
                output_symbol["directive_code"] = "INVALID_TARGET"
            elif not entity["attributes"].get("is_locked"):
                output_symbol["directive_code"] = "ALREADY_UNLOCKED"
            else:
                # Check player inventory
                player_inv = self.state["entities"]["player"]["attributes"]["inventory"]
                required_key = entity["attributes"].get("required_key")
                
                if required_key in player_inv:
                    # Logic approved, update state
                    entity["attributes"]["is_locked"] = False
                    output_symbol["logic_status"] = "state_updated"
                    output_symbol["directive_code"] = "CONFIRM_UNLOCKED"
                    output_symbol["state_delta"] = {
                        target: {"is_locked": False}
                    }
                else:
                    output_symbol["directive_code"] = "DENY_ACCESS_MISSING_KEY"

        # Handle Action: Query State
        elif intent == "query_state":
            entity = self.state["entities"].get(target)
            if entity:
                output_symbol["logic_status"] = "approved"
                output_symbol["directive_code"] = "REPORT_STATE_SUCCESS"
                output_symbol["state_delta"] = {"queried_target": target}
            else:
                output_symbol["directive_code"] = "TARGET_NOT_FOUND"

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        output_symbol["execution_time_ms"] = round(elapsed_ms, 4)
        
        return output_symbol

if __name__ == "__main__":
    core = SRM_Core()
    
    # Simulate an InputSymbol coming from Layer 4
    # User said: "I want to use my key to open the iron gate"
    # Layer 4 mapped this to:
    test_input = {
        "intent_class": "execute_action",
        "target_entity": "iron_gate",
        "parameters": {
            "action": "open"
        }
    }
    
    print("--- Initial System State ---")
    print(json.dumps(core.state, indent=2))
    
    print("\n--- Processing Input Symbol ---")
    print(json.dumps(test_input, indent=2))
    
    output = core.process_input(test_input)
    
    print("\n--- Output Symbol (Layer 1 Response) ---")
    print(json.dumps(output, indent=2))
    
    print("\n--- Updated System State ---")
    print(json.dumps(core.state, indent=2))
