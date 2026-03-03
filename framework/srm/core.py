from typing import Dict, Any
from abc import ABC, abstractmethod

class SymbolicState(ABC):
    """
    A foundational container for strict state representation.
    Users should subclass this to define their exact JSON schema (Knowledge Graph).
    """
    def __init__(self, initial_state: Dict[str, Any] = None):
        self.state = initial_state or {}

class LogicEngine(ABC):
    """
    The Brain (Layer 1).
    Takes a clean symbol (intent) and state, outputs a clean symbol (directive/snapshot).
    """
    
    @abstractmethod
    def process(self, intent: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Processes an incoming pure symbol and returns a resulting pure symbol.
        No natural language allowed here.
        """
        pass
