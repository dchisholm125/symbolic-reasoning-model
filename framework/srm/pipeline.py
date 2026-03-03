from typing import Dict, Any
from .core import LogicEngine
from .parsers import IntentRouter
from .generators import SLM_Node, ContextBridge

class Framework:
    """
    The Master Orchestrator. Extends the 4-layer Symbolic Reasoning Model architecture.
    Handles Dual-Stream Cognition:
    1. Cold Path: Human Language -> Ear -> Brain -> Bridge -> Mouth
    2. Hot Path: Raw Sensory Data -> Brain -> (Action)
    """
    def __init__(self, ear: IntentRouter, brain: LogicEngine, mouth: SLM_Node, system_prompt: str = ""):
        self.ear = ear
        self.brain = brain
        self.mouth = mouth
        self.system_prompt = system_prompt or "You are a quantitative trading reporting agent. Explain the current state concisely."

    def run_cold_path(self, user_text: str) -> str:
        """
        The communicative loop. Takes messy English, outputs fluid English.
        """
        # Layer 4 (Ear) -> Vectorize and match language
        intent = self.ear.parse(user_text)
        
        if intent == "invalid_intent":
            return "Layer 4 rejected the input: Language outside predefined structural bounds."
            
        # Layer 1 (Brain) -> Determine logic gate
        logic_output = self.brain.process(intent=intent)
        
        if not logic_output:
            return "Layer 1 Error: State mutation failed."

        # Layer 2 (Context Bridge) -> Reformat pure JSON math into a prompt
        context = ContextBridge.construct_prompt(logic_output, self.system_prompt)
        
        # Layer 3 (Mouth) -> Output fluidity
        response = self.mouth.generate(context)
        
        return response

    def run_hot_path(self, intent: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        The sensory loop. Bypasses Layers 2, 3, and 4 entirely.
        Takes pure data (e.g. from a Solana WebSocket), executes logic in Layer 1, 
        and returns an instant directive. Latency: < 1ms.
        """
        return self.brain.process(intent=intent, params=params)
        
    def run(self, user_text: str) -> str:
        """Legacy alias for run_cold_path to maintain backward compatibility."""
        return self.run_cold_path(user_text)
