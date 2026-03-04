"""
LLM Backend Abstraction Layer
=============================
Provides a unified interface for LLM inference that can target:
  1. Remote OpenAI-compatible APIs (OpenRouter, Groq, Together, Mistral, etc.)
  2. Local Ollama server (localhost:11434)
  3. Direct llama_cpp (legacy, for the desktop GPU branch)

Selection is driven by the LLM_BACKEND environment variable.
"""

import os
import json
import time
import requests
from typing import Optional, List, Dict, Any, Generator

# Auto-load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars


# ---------------------------------------------------------------------------
# Configuration (all from env vars, with sane defaults)
# ---------------------------------------------------------------------------
# LLM_BACKEND values: "openai_compat" | "ollama" | "llama_cpp"
LLM_BACKEND   = os.environ.get("LLM_BACKEND", "openai_compat")
LLM_API_URL   = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY   = os.environ.get("LLM_API_KEY", "")
LLM_MODEL     = os.environ.get("LLM_MODEL", "deepseek/deepseek-r1-distill-llama-70b:free")
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

# For the legacy local branch (llama_cpp)
GGUF_MODEL_PATH = os.environ.get("GGUF_MODEL_PATH", "./qwen2.5-0.5b-instruct-q4_k_m.gguf")


class LLMBackend:
    """
    Unified LLM interface.  Every consumer calls .complete() or .stream()
    and the backend routes to the configured provider.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        gguf_path: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_threads: int = 4,
        n_ctx: int = 2048,
    ):
        self.backend      = backend or LLM_BACKEND
        self.api_url      = api_url or LLM_API_URL
        self.api_key      = api_key or LLM_API_KEY
        self.model        = model or LLM_MODEL
        self.ollama_url   = ollama_url or OLLAMA_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.gguf_path    = gguf_path or GGUF_MODEL_PATH
        self.is_loaded    = False
        self._llama       = None  # Only used for llama_cpp backend

        # llama_cpp specific params (legacy GPU branch)
        self._n_gpu_layers = n_gpu_layers
        self._n_threads    = n_threads
        self._n_ctx        = n_ctx

        self._boot()

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------
    def _boot(self):
        if self.backend == "llama_cpp":
            self._boot_llama_cpp()
        elif self.backend == "ollama":
            self._verify_ollama()
        elif self.backend == "openai_compat":
            self._verify_openai_compat()
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {self.backend}")

    def _boot_llama_cpp(self):
        """Load the GGUF model directly into VRAM (desktop branch)."""
        try:
            from llama_cpp import Llama
            print(f"[LLMBackend] Loading GGUF model: {self.gguf_path}")
            self._llama = Llama(
                model_path=self.gguf_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                n_threads=self._n_threads,
                n_batch=512,
                use_mmap=False,
                verbose=False,
            )
            # Warmup
            self._llama(
                "<|im_start|>system\nHi<|im_end|>\n",
                max_tokens=1,
            )
            self.is_loaded = True
            print("[LLMBackend] GGUF model loaded and warm.")
        except Exception as e:
            print(f"[LLMBackend] ERROR loading llama_cpp: {e}")
            self.is_loaded = False

    def _verify_ollama(self):
        """Quick health-check against the Ollama server."""
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"[LLMBackend] Ollama online. Available models: {models}")
            if self.ollama_model not in models and not any(self.ollama_model in m for m in models):
                print(f"[LLMBackend] WARNING: Model '{self.ollama_model}' not found. Pull it with: ollama pull {self.ollama_model}")
            self.is_loaded = True
        except Exception as e:
            print(f"[LLMBackend] WARNING: Ollama not reachable at {self.ollama_url}: {e}")
            self.is_loaded = False

    def _verify_openai_compat(self):
        """Verify we have a key and can reach the endpoint."""
        if not self.api_key:
            print("[LLMBackend] WARNING: LLM_API_KEY not set. Set it in your .env file.")
            print(f"[LLMBackend] Will attempt calls to {self.api_url} with model {self.model}")
        else:
            print(f"[LLMBackend] OpenAI-compatible API configured.")
            print(f"  URL:   {self.api_url}")
            print(f"  Model: {self.model}")
        self.is_loaded = True

    # ------------------------------------------------------------------
    # Chat Completion (non-streaming)
    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.1,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Single-shot text completion.  Returns the generated text.
        Accepts either a raw prompt string (for legacy llama_cpp style)
        or builds a chat messages array for chat APIs.
        """
        if self.backend == "llama_cpp":
            return self._complete_llama_cpp(prompt, max_tokens, temperature, stop)
        elif self.backend == "ollama":
            return self._complete_ollama(prompt, max_tokens, temperature, stop, system_prompt)
        elif self.backend == "openai_compat":
            return self._complete_openai(prompt, max_tokens, temperature, stop, system_prompt)
        else:
            return "[LLMBackend] No backend configured."

    def _complete_llama_cpp(self, prompt, max_tokens, temperature, stop):
        if not self._llama:
            return "[Mock] LLM not loaded."
        output = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        return output["choices"][0]["text"].strip()

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Convert a raw chatml-style prompt into an OpenAI messages array.
        If the prompt already contains <|im_start|> markers we parse them;
        otherwise we treat the whole thing as a user message.
        """
        messages = []

        if "<|im_start|>" in prompt:
            # Parse chatml format
            parts = prompt.split("<|im_start|>")
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # Remove trailing <|im_end|> and anything after
                part = part.split("<|im_end|>")[0].strip()
                if part.startswith("system\n") or part.startswith("system\\n"):
                    content = part.replace("system\n", "", 1).replace("system\\n", "", 1).strip()
                    messages.append({"role": "system", "content": content})
                elif part.startswith("user\n") or part.startswith("user\\n"):
                    content = part.replace("user\n", "", 1).replace("user\\n", "", 1).strip()
                    messages.append({"role": "user", "content": content})
                elif part.startswith("assistant\n") or part.startswith("assistant\\n"):
                    content = part.replace("assistant\n", "", 1).replace("assistant\\n", "", 1).strip()
                    if content:
                        messages.append({"role": "assistant", "content": content})
        elif "<|system|>" in prompt:
            # Parse the phi-style format used in generators.py
            parts = prompt.split("<|")
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.startswith("system|>"):
                    content = part.replace("system|>", "").replace("<|end|>", "").strip()
                    # Also strip any trailing tags 
                    content = content.split("<|end|>")[0].strip()
                    messages.append({"role": "system", "content": content})
                elif part.startswith("user|>"):
                    content = part.replace("user|>", "").replace("<|end|>", "").strip()
                    content = content.split("<|end|>")[0].strip()
                    messages.append({"role": "user", "content": content})
                elif part.startswith("assistant|>"):
                    content = part.replace("assistant|>", "").replace("<|end|>", "").strip()
                    content = content.split("<|end|>")[0].strip()
                    if content:
                        messages.append({"role": "assistant", "content": content})
        else:
            # Plain text prompt
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        # Ensure we have at least a user message
        if not messages:
            messages.append({"role": "user", "content": prompt})

        return messages

    def _complete_ollama(self, prompt, max_tokens, temperature, stop, system_prompt):
        messages = self._build_messages(prompt, system_prompt)
        try:
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stop": stop or [],
                    },
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            return f"[Ollama Error] {e}"

    def _complete_openai(self, prompt, max_tokens, temperature, stop, system_prompt):
        messages = self._build_messages(prompt, system_prompt)
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter specific headers (harmless for other providers)
        headers["HTTP-Referer"] = "https://github.com/symbolic-reasoning-model"
        headers["X-Title"] = "SRM Framework"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        try:
            r = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            # Safety check for message content
            message = data.get("choices", [{}])[0].get("message", {})
            content = message.get("content")
            if not content:
                # Support reasoning models (DeepSeek R1 etc.)
                content = message.get("reasoning_content")
            
            return content.strip() if content else "[API Error] Empty response from model."
        except Exception as e:
            return f"[API Error] {e}"

    # ------------------------------------------------------------------
    # Streaming Completion (yields token strings)
    # ------------------------------------------------------------------
    def stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.1,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming completion — yields individual token strings as they arrive.
        """
        if self.backend == "llama_cpp":
            yield from self._stream_llama_cpp(prompt, max_tokens, temperature, stop)
        elif self.backend == "ollama":
            yield from self._stream_ollama(prompt, max_tokens, temperature, stop, system_prompt)
        elif self.backend == "openai_compat":
            yield from self._stream_openai(prompt, max_tokens, temperature, stop, system_prompt)
        else:
            yield "[LLMBackend] No backend configured."

    def _stream_llama_cpp(self, prompt, max_tokens, temperature, stop):
        if not self._llama:
            yield "[Mock] LLM not loaded."
            return
        stream = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            stream=True,
        )
        for chunk in stream:
            yield chunk["choices"][0]["text"]

    def _stream_ollama(self, prompt, max_tokens, temperature, stop, system_prompt):
        messages = self._build_messages(prompt, system_prompt)
        try:
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stop": stop or [],
                    },
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        break
        except Exception as e:
            yield f"[Ollama Error] {e}"

    def _stream_openai(self, prompt, max_tokens, temperature, stop, system_prompt):
        messages = self._build_messages(prompt, system_prompt)
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["HTTP-Referer"] = "https://github.com/symbolic-reasoning-model"
        headers["X-Title"] = "SRM Framework"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        try:
            r = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=120,
            )
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"[API Error] {e}"

    # ------------------------------------------------------------------
    # Convenience: raw dict output (legacy compat)
    # ------------------------------------------------------------------
    def __call__(self, prompt, **kwargs):
        """
        Legacy compatibility: mimic the llama_cpp Llama() callable interface.
        Returns a dict shaped like llama_cpp output for backward compat.
        """
        max_tokens = kwargs.get("max_tokens", 100)
        temperature = kwargs.get("temperature", 0.1)
        stop = kwargs.get("stop", [])
        stream = kwargs.get("stream", False)

        if stream:
            return self._legacy_stream(prompt, max_tokens, temperature, stop)

        text = self.complete(prompt, max_tokens, temperature, stop)
        return {
            "choices": [{"text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _legacy_stream(self, prompt, max_tokens, temperature, stop):
        """Generator that yields llama_cpp-style dicts for streaming."""
        for token in self.stream(prompt, max_tokens, temperature, stop):
            yield {
                "choices": [{"text": token, "finish_reason": None}],
            }


# ---------------------------------------------------------------------------
# Module-level singleton for easy import
# ---------------------------------------------------------------------------
_default_backend: Optional[LLMBackend] = None

def get_default_backend(**kwargs) -> LLMBackend:
    """Get or create the module-level default backend singleton."""
    global _default_backend
    if _default_backend is None:
        _default_backend = LLMBackend(**kwargs)
    return _default_backend
