import json
from typing import Dict, Any

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

class ContextBridge:
    """
    Layer 2: The Formulator.
    Secures the border between pure mathematical State and loose Language models.
    """
    @staticmethod
    def construct_prompt(data: Dict[str, Any], system_prompt: str) -> str:
        stringified_data = json.dumps(data, indent=2)
        # Refined to force a 'Quant' persona and prevent general-purpose yapping.
        prompt = (
            f"<|system|>\n{system_prompt}. "
            f"Be brief, professional, and focus strictly on the numbers. "
            f"Do not explain basic concepts.<|end|>\n"
            f"<|user|>\nData State:\n{stringified_data}\n\n"
            f"Provide a concise technical summary of this state.<|end|>\n"
            f"<|assistant|>\n"
        )
        return prompt

class SLM_Node:
    """
    Layer 3: 'The Mouth'.
    Updated for 6GB VRAM Safety. Implements partial offloading to prevent system hangs.
    """
    def __init__(self, model_path: str, n_ctx: int = 512, n_gpu_layers: int = 20):
        self.model_path = model_path
        self.is_loaded = False
        
        if HAS_LLAMA:
            # SAFETY CHECK: 
            # On 6GB VRAM, 'n_gpu_layers=-1' is dangerous for 3B+ models.
            # We default to 20 layers. If you use a 1.1B model, you can set this back to -1.
            print(f"[SLM_Node] Loading model: {model_path}")
            print(f"[SLM_Node] Offloading {n_gpu_layers} layers to GPU (Safe Mode)...")
            
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=256,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
                self.is_loaded = True
                print("[SLM_Node] Model loaded successfully.")
            except Exception as e:
                print(f"[SLM_Node] ERROR: Failed to load model. Likely VRAM OOM: {e}")
        else:
            print("[SLM_Node] WARNING: llama-cpp-python not installed. Running in mock mode.")
        
    def generate(self, context: str, max_tokens: int = 80, temperature: float = 0.0) -> str:
        """
        Executes the prompt. 
        Temperature is dropped to 0.0 to ensure the bot stays strictly technical.
        Max tokens lowered to 80 to keep it concise.
        """
        if not self.is_loaded:
            return f"[Simulated Response] -> Synthesizing speech from state data."
            
        output = self.llm(
            context,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|end|>", "User:", "\n\n", "Data State:"]
        )
        
        return output['choices'][0]['text'].strip()