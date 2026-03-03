# SRM: The Symbolic Reasoning Model Framework

A local-first, neuro-symbolic AI framework designed to run on 6GB of VRAM.

Current state-of-the-art Large Language Models (LLMs) are structurally flawed for pure reasoning. They entangle the complex act of thinking (logic, math, state-tracking) with the mechanical act of speaking (token probability). This results in massive VRAM requirements, severe latency, and inevitable hallucinations.

The SRM Framework inverts the paradigm. It completely decouples language from logic.

## The "Inside-Out" Architecture

SRM acts as a biological central nervous system, built on four distinct layers:

- **Layer 4 (The Ear)**: A fast ONNX semantic router. It compresses messy human English into strict JSON symbols in < 10ms using pure CPU compute.
- **Layer 1 (The Brain)**: The core LogicEngine. A pure mathematical state machine or Reinforcement Learning (RL) environment. It executes logic, updates state, and makes decisions with zero natural language context.
- **Layer 2 (The Bridge)**: The ContextBridge. It takes the pure JSON output from the Brain and formats it into a rigid, factual prompt.
- **Layer 3 (The Mouth)**: A heavily quantized Small Language Model (SLM) running via llama.cpp. It acts purely as a text-to-speech engine, translating the factual state into fluid English.

## Why SRM?

- **Zero Hallucinations**: Layer 3 (The SLM) has no memory, no chat history, and no ability to reason. It cannot hallucinate facts because it is only allowed to read the JSON state provided by Layer 1.
- **Context-Window Collapse**: By storing memory natively in Layer 1 Python objects, the SLM's context window is permanently reduced to ~200 tokens.
- **Hardware Efficiency**: Runs entirely locally. The Ear runs on the CPU, the Brain is pure code/math, and the Mouth requires < 3GB of VRAM.

## Quickstart

```bash
pip install -r requirements.txt
```

```python
from srm import Framework, IntentRouter, LogicEngine, SLM_Node

# 1. The Ear (Semantic routing via ONNX)
ear = IntentRouter(intents={"query_status": ["what's up", "give me an update"]})

# 2. The Brain (Pure Logic/Math)
class TradingLogic(LogicEngine):
    def process(self, intent: str, params=None):
        if intent == "query_status":
            return {"status": "systems nominal", "portfolio_value": 1050.20}

# 3. The Mouth (Local generation)
mouth = SLM_Node(model_path="./models/tiny-llama-1.5B-4bit.gguf")

# 4. Orchestration
agent = Framework(ear, TradingLogic(), mouth)

response = agent.run("Give me an update on our operation.")
print(response) 
# "Systems are nominal and the current portfolio value is $1,050.20."
```

## Roadmap

- [x] Basic Framework Orchestration
- [x] Layer 4 ONNX Integration
- [x] Layer 3 Llama.cpp Integration
- [ ] Layer 1 Reinforcement Learning (RL) Gym Wrappers
- [ ] Live WebSocket (Hot Path) ingestion standard
