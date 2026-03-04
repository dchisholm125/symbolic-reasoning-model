"""
srm_mouth.py

Layer 3 - "The Mouth" of the Symbolic Reasoning Model.

Renders the dense JSON payload produced by SRMBridge into concise,
professional prose using a quantized local LLM (Qwen 2.5 0.5B-Instruct Q4).

Architecture role
-----------------
  SRMBridge (JSON) --> SRMMouth.speak(payload) --> streamed text tokens
                                                         |
                                             console / WebSocket / log

Design principles
------------------
  * The LLM is deliberately constrained to a rendering role only.
    All reasoning is pre-computed by the MCTS Brain and Bridge.
    The system prompt explicitly forbids hallucination of new facts.

  * The prompt is tiny: system (~90 tokens) + JSON (~200 tokens) = ~300 tokens.
    Qwen 0.5B produces the 1-2 sentence summary in ~1-2 seconds on CPU.

  * speak() is a synchronous generator - yields decoded token strings
    one-by-one so the caller can stream to a console or WebSocket as they
    arrive, rather than waiting for the full response.

  * speak_async() is an async generator for FastAPI / asyncio contexts.

  * The model is loaded once at construction time and reused for all calls.
    Re-loading a 469 MB GGUF on every decision would add 3-5 seconds latency.

Prompt format  - Qwen 2.5 ChatML
---------------------------------
  <|im_start|>system ... <|im_end|>
  <|im_start|>user   ... <|im_end|>
  <|im_start|>assistant
  (model generates from here)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Generator, Optional

# -------------------------------------------------------------------------------
# Default model path — the GGUF sitting in the repo root
# -------------------------------------------------------------------------------

DEFAULT_MODEL_PATH = str(
    Path(__file__).parent / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

# -------------------------------------------------------------------------------
# System prompt
# -------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the articulation layer of a quantitative trading AI. "
    "You will receive a JSON payload containing a market symbol, entropy level, "
    "MCTS simulation results, a trading decision, Q-values, and a reasoning chain. "
    "Your sole task: summarize this JSON into ONE or TWO concise, professional sentences "
    "that state the decision and its primary justification. "
    "Output plain English only - no JSON, no bullet points, no markdown. "
    "Never add information not present in the JSON. Never hallucinate prices or facts."
)

# Qwen 2.5 ChatML stop tokens
_IM_END  = "<|im_end|>"
_IM_START = "<|im_start|>"
STOP_TOKENS = [_IM_END, "\n\n\n"]

# -------------------------------------------------------------------------------
# SRMMouth
# -------------------------------------------------------------------------------

class SRMMouth:
    """
    LLM rendering layer for SRM decision payloads.

    Loads a Qwen 2.5 0.5B-Instruct GGUF model via llama-cpp-python and exposes:
      * speak(payload)        -> synchronous token generator (print as they arrive)
      * speak_full(payload)   -> blocking call, returns complete string
      * speak_async(payload)  -> async generator for FastAPI / websockets

    Parameters
    ----------
    model_path : str | Path
        Path to the .gguf model file.
    n_ctx : int
        Context window size.  1024 tokens is sufficient for our slim prompts
        (~200 tokens) with comfortable headroom.  Increase only if payloads grow.
    n_threads : int
        CPU threads for inference.  Defaults to 4 (matches quad-core i5).
    n_gpu_layers : int
        GPU offload layers.  0 = pure CPU (avoids CUDA transfer overhead for
        tiny matrices).  Set to -1 to auto-offload all layers to GPU if
        latency at higher n_ctx becomes an issue.
    max_tokens : int
        Hard cap on generated tokens.  1-2 sentences = ~40-60 tokens.
        We set 80 to give breathing room without risk of runaway generation.
    temperature : float
        Sampling temperature.  0.0 = greedy/deterministic (reproducible).
        Use 0.1-0.2 for slight variation in phrasing.
    verbose : bool
        Whether llama.cpp prints load/inference logs.  Default: False.
    """

    def __init__(
        self,
        model_path:    str | Path = DEFAULT_MODEL_PATH,
        n_ctx:         int   = 1024,
        n_threads:     int   = 4,
        n_gpu_layers:  int   = 0,
        max_tokens:    int   = 80,
        temperature:   float = 0.0,
        verbose:       bool  = False,
    ) -> None:
        self.model_path  = Path(model_path)
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._llm        = None
        self._load_time_s: float = 0.0
        self.is_loaded   = False

        self._load_model(n_ctx, n_threads, n_gpu_layers, verbose)

    # ---------------------------------------------------------------------------
    # Private
    # ---------------------------------------------------------------------------

    def _load_model(
        self,
        n_ctx:        int,
        n_threads:    int,
        n_gpu_layers: int,
        verbose:      bool,
    ) -> None:
        """Load the GGUF model. Raises FileNotFoundError with download hint if missing."""
        if not self.model_path.exists():
            model_name = self.model_path.name
            raise FileNotFoundError(
                f"\n[SRMMouth] Model file not found: {self.model_path}\n\n"
                f"  Download it with:\n"
                f"    pip install huggingface_hub\n"
                f"    python -c \"\n"
                f"      from huggingface_hub import hf_hub_download\n"
                f"      hf_hub_download(\n"
                f"        repo_id='Qwen/Qwen2.5-0.5B-Instruct-GGUF',\n"
                f"        filename='{model_name}',\n"
                f"        local_dir='.'\n"
                f"      )\n"
                f"    \"\n\n"
                f"  Or for 1.5B (better quality, ~900 MB):\n"
                f"    repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF'\n"
                f"    filename='qwen2.5-1.5b-instruct-q4_k_m.gguf'"
            )

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "[SRMMouth] llama-cpp-python is not installed.\n"
                "  Install it with:  pip install llama-cpp-python"
            )

        print(f"[SRMMouth] Loading {self.model_path.name}  "
              f"(ctx={n_ctx}, threads={n_threads}, gpu_layers={n_gpu_layers}) ...",
              end="", flush=True)

        t0 = time.perf_counter()
        self._llm = Llama(
            model_path   = str(self.model_path),
            n_ctx        = n_ctx,
            n_threads    = n_threads,
            n_gpu_layers = n_gpu_layers,
            n_batch      = 512,
            verbose      = verbose,
        )
        self._load_time_s = time.perf_counter() - t0

        # Warm-up pass: runs the tokenizer + 1 forward step so the first real
        # call doesn't suffer the JIT initialisation penalty.
        self._llm(f"{_IM_START}system\nok{_IM_END}\n", max_tokens=1, echo=False)

        self.is_loaded = True
        print(f"  ready in {self._load_time_s:.1f}s")

    def _build_prompt(self, payload: str) -> str:
        """
        Format the JSON payload into a Qwen 2.5 ChatML prompt string.

        We pass the raw JSON as the user turn so the model sees it exactly as
        produced by SRMBridge — no additional wrapping or summarisation.
        """
        return (
            f"{_IM_START}system\n{SYSTEM_PROMPT}{_IM_END}\n"
            f"{_IM_START}user\n{payload}{_IM_END}\n"
            f"{_IM_START}assistant\n"
        )

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def speak(self, payload: str) -> Generator[str, None, None]:
        """
        Stream the LLM's prose narration of a SRMBridge JSON payload.

        Yields individual token strings as they are generated.  The caller
        is responsible for printing/forwarding them.

        Parameters
        ----------
        payload : str
            JSON string from SRMBridge.generate_payload().

        Yields
        ------
        str : one decoded token at a time.

        Example
        -------
        >>> for token in mouth.speak(json_payload):
        ...     print(token, end="", flush=True)
        """
        if not self.is_loaded or self._llm is None:
            raise RuntimeError("[SRMMouth] Model not loaded. Check __init__ errors.")

        prompt = self._build_prompt(payload)

        stream = self._llm(
            prompt,
            max_tokens  = self.max_tokens,
            temperature = self.temperature,
            stop        = STOP_TOKENS,
            stream      = True,
            echo        = False,
        )

        for chunk in stream:
            token_text = chunk["choices"][0]["text"]
            if token_text:
                yield token_text

    def speak_full(self, payload: str) -> tuple[str, float]:
        """
        Blocking variant — waits for the full response and returns it.

        Returns
        -------
        (text, elapsed_seconds) : tuple[str, float]
        """
        t0     = time.perf_counter()
        tokens = []
        for token in self.speak(payload):
            tokens.append(token)
        elapsed = time.perf_counter() - t0
        return "".join(tokens).strip(), elapsed

    async def speak_async(self, payload: str):
        """
        Async generator variant for use with FastAPI StreamingResponse or
        any asyncio context.  Runs the synchronous generator in a thread
        executor to avoid blocking the event loop.

        Usage (FastAPI):
        ----------------
        from fastapi.responses import StreamingResponse

        @app.post("/speak")
        async def speak_endpoint(body: PayloadRequest):
            return StreamingResponse(
                mouth.speak_async(body.payload),
                media_type="text/plain",
            )
        """
        import asyncio

        loop = asyncio.get_event_loop()

        # Run the blocking generator in a thread pool so the event loop stays free
        # We collect into a queue and yield from there.
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def _run_in_thread():
            try:
                for token in self.speak(payload):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        thread = asyncio.get_event_loop().run_in_executor(None, _run_in_thread)

        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            yield item

        await thread

    def token_count(self, text: str) -> int:
        """Estimate token count for a string (uses llama.cpp tokenizer)."""
        if self._llm is None:
            return len(text.split())
        return len(self._llm.tokenize(text.encode()))

    def __repr__(self) -> str:
        status = f"loaded in {self._load_time_s:.1f}s" if self.is_loaded else "not loaded"
        return (
            f"SRMMouth("
            f"model='{self.model_path.name}', "
            f"max_tokens={self.max_tokens}, "
            f"temp={self.temperature}, "
            f"{status})"
        )


# -------------------------------------------------------------------------------
# Execution block
# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # ---- 1. Construct the test payload (exact values from prior MCTS run) ----
    from srm_bridge import SRMBridge

    bridge  = SRMBridge(include_timestamp=False, indent=2)

    # Symbol #0199: high-entropy tick event, HOLD decision
    payload_199 = bridge.generate_payload(
        symbol      = 199,
        entropy     = 7.559,
        action      = 0,           # HOLD
        q_values    = {0: +0.1417, 1: -0.0214, 2: +0.0869},
        simulations = 694,
        top_transitions = [
            (337, 0.01816, 17),
            (294, 0.01816, 17),
            (68,  0.01816, 17),
        ],
        icm_reward  = 1.843,
    )

    # Symbol #0243: low-entropy stasis, SELL decision
    payload_243 = bridge.generate_payload(
        symbol      = 243,
        entropy     = 0.326,
        action      = 2,           # SELL
        q_values    = {0: -0.3963, 1: -0.3967, 2: -0.0108},
        simulations = 77,
        top_transitions = [(243, 0.9699, 7266), (166, 0.00320, 24)],
        icm_reward  = 0.214,
    )

    # ---- 2. Load the model ---------------------------------------------------
    print("=" * 64)
    print("  SRMMouth — Layer 3 'The Mouth'")
    print("=" * 64 + "\n")

    try:
        mouth = SRMMouth(
            model_path  = DEFAULT_MODEL_PATH,
            n_ctx       = 1024,
            n_threads   = 4,
            n_gpu_layers= 0,       # strict CPU — matches existing api_server config
            max_tokens  = 80,
            temperature = 0.0,     # deterministic / reproducible
            verbose     = False,
        )
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"\n{mouth}\n")

    # ---- 3. Token count diagnostics -----------------------------------------
    # Slim the payloads before sending to the LLM - drops 'reasoning' and
    # 'action_rankings' (saves ~65% tokens).  LLM derives narration from facts.
    slim_199 = bridge.slim_payload(payload_199)
    slim_243 = bridge.slim_payload(payload_243)

    for label, pl in [("Symbol #0199 slim payload", slim_199),
                       ("Symbol #0243 slim payload", slim_243)]:
        prompt = mouth._build_prompt(pl)
        toks   = mouth.token_count(prompt)
        print(f"  {label}: ~{toks} tokens in prompt")
    print()

    # ---- 4. Streamed narration — Symbol #0199 (HIGH entropy, HOLD) ----------
    print("=" * 64)
    print("  Narration 1 — Symbol #0199  |  HIGH entropy  |  HOLD")
    print("=" * 64)
    print("\n  [SRM Brain] ", end="", flush=True)

    t0 = time.perf_counter()
    for token in mouth.speak(slim_199):
        print(token, end="", flush=True)
    elapsed_1 = time.perf_counter() - t0
    print(f"\n\n  Latency: {elapsed_1:.2f}s\n")

    # ---- 5. Streamed narration — Symbol #0243 (LOW entropy, SELL) -----------
    print("=" * 64)
    print("  Narration 2 — Symbol #0243  |  LOW entropy   |  SELL")
    print("=" * 64)
    print("\n  [SRM Brain] ", end="", flush=True)

    t0 = time.perf_counter()
    for token in mouth.speak(slim_243):
        print(token, end="", flush=True)
    elapsed_2 = time.perf_counter() - t0
    print(f"\n\n  Latency: {elapsed_2:.2f}s\n")

    # ---- 6. speak_full() demo -----------------------------------------------
    print("=" * 64)
    print("  speak_full() — blocking call, returns (text, elapsed)")
    print("=" * 64)

    text, elapsed_f = mouth.speak_full(slim_199)
    print(f"\n  Result  : {text}")
    print(f"  Elapsed : {elapsed_f:.2f}s")

    # ---- 7. Pipeline throughput estimate ------------------------------------
    avg_ms = ((elapsed_1 + elapsed_2) / 2) * 1000
    print(f"\n{'=' * 64}")
    print(f"  Pipeline summary")
    print(f"{'=' * 64}")
    print(f"  Model load time   : {mouth._load_time_s:.1f}s  (one-time cost)")
    print(f"  Avg narration     : {avg_ms:.0f}ms  ({1000/avg_ms:.1f} decisions/s)")
    print(f"  Full pipeline est : MCTS (~300ms) + Bridge (~0ms) + Mouth (~{avg_ms:.0f}ms)")
    print(f"\n  ✅  SRMMouth operational.")
    print(f"  Downstream usage:")
    print(f"    for token in mouth.speak(bridge.generate_payload(...)): print(token, end='')")
