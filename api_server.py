from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import json
import asyncio

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[System] Booting TINY LLM into System RAM (CPU Mode)...")
if HAS_LLAMA:
    llm = Llama(
        # Pointing to the new 0.5B model!
        model_path="./qwen2.5-0.5b-instruct-q4_k_m.gguf", 
        n_ctx=2048, 
        n_gpu_layers=0, # 0 forces it to strictly use your Intel i5 CPU!
        n_threads=4,    # Maps perfectly to your quad-core i5
        n_batch=512,
        verbose=False
    )
    llm("<|im_start|>system\nHi<|im_end|>\n", max_tokens=1)
    print("[System] CPU Engine Hot and Ready.")
else:
    llm = None
    print("[System] Running in MOCK mode.")

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    message = request.message
    
    # --- TRADITIONAL LLM PIPELINE ---
    history = [{"tick": i, "price": 84.00 + (i * 0.1), "fee": 5000} for i in range(20)]
    traditional_prompt = (
        "<|im_start|>system\nYou are an autonomous trading AI. Analyze the market history and answer the user.<|im_end|>\n"
        f"<|im_start|>user\nHere is the recent market history:\n{json.dumps(history)}\n\n"
        f"The current price is $86.00 and the fee is 5000. User asks: '{message}'. What should we do?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    t_start = time.perf_counter()
    if llm:
        t_out = llm(traditional_prompt, max_tokens=100, temperature=0.1, stop=["<|im_end|>"])
        traditional_text = t_out['choices'][0]['text'].strip()
    else:
        traditional_text = "Mocked Traditional Response"
    t_time = time.perf_counter() - t_start

   # --- SRM FRAMEWORK PIPELINE ---
    srm_state = {"action": "SELL", "pnl": "+$4.50", "portfolio": 1004.50, "reason": "Momentum Negative"}
    
    # Notice we DELETED the user's message from this prompt. 
    # The LLM only translates the math. It does not talk to the user.
    srm_prompt = (
        "<|im_start|>system\nYou are a strict terminal. Summarize state in 1 short sentence.<|im_end|>\n"
        f"<|im_start|>user\nSTATE:\n{json.dumps(srm_state)}\n\nREPORT:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    s_start = time.perf_counter()
    if llm:
        s_out = llm(srm_prompt, max_tokens=30, temperature=0.0, stop=["<|im_end|>"])
        srm_text = s_out['choices'][0]['text'].replace("|-|end|", "").replace("<|im_end|>", "").strip()
    else:
        srm_text = "Mocked SRM Response"
    s_time = time.perf_counter() - s_start

    return {
        "traditional": {"text": traditional_text, "latency": round(t_time, 2)},
        "srm": {"text": srm_text, "latency": round(s_time, 2)}
    }
