"""
srm_nerve.py

Layer 4 – "The Nerve" of the Symbolic Reasoning Model.

Ingests LIVE Solana DLMM market ticks via WebSocket, converts each raw
message into the exact 8-dimensional state vector that SRMEnvironment
uses, fires the MCTS planner on the hot path, and queues the resulting
decision payload for the LLM narration layer (SRMMouth) on a cold path.

Architecture
────────────
                    ┌──────────────────────────────┐
  WebSocket         │         SRMLiveNerve          │
  (live / mock) ──► │                              │
                    │  [HOT PATH]  ~300–500 ms      │
                    │  raw_tick                     │
                    │    → _extract_scalars()       │
                    │    → _compute_state()  (NumPy)│
                    │    → quantizer.encode()       │
                    │    → mcts.search()            │
                    │    → bridge.generate_payload()│
                    │    → asyncio.Queue.put()      │
                    │                              │
                    │  [COLD PATH]  ~30–45 s        │
                    │  Queue.get()                  │
                    │    → mouth.speak() (streamed) │
                    └──────────────────────────────┘

Threading model
───────────────
  • The WebSocket listener and MCTS search run in the asyncio event loop.
  • SRMMouth.speak() is synchronous and CPU-bound (~40 s on CPU).
    It is offloaded to a ThreadPoolExecutor so the event loop never blocks.
  • ICM updates also run in the thread pool (gradient step ~20 ms).
  • The Queue decouples the two loops: new decisions accumulate while the
    LLM is narrating a previous one.  No tick data is ever dropped.

WebSocket data format (Solana DLMM collector schema)
────────────────────────────────────────────────────
  {
    "timestamp":     "2026-02-25T07:40:08.123Z",
    "activeBinId":   5432,
    "activePrice":   84.80,
    "avgBinDepthUsd": 1450.0,
    "obiScore":      -0.34,
    "slot":          295000000,
    "poolAddress":   "HJPj...",
    "bidDepthUsd":   8200.0,
    "askDepthSol":   97.2,
    "microPrice":    84.79,
    "binStep":       25
  }

State vector (8D) ← identical to SRMEnvironment
────────────────────────────────────────────────
  [0] Δ activePrice     %
  [1] Δ microPrice      %
  [2] Δ avgBinDepthUsd  %
  [3] Δ bidDepthUsd     %
  [4] Δ askDepthSol     %
  [5]   obiScore        (raw)
  [6] Δ obiScore
  [7] Δ activeBinId     (raw integer drift)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# State computation helpers  (mirrors SRMEnvironment exactly)
# ──────────────────────────────────────────────────────────────────────────────

_DELTA_CLIP = 100.0
_EPS        = 1e-10


def _pct_delta(current: float, previous: float) -> float:
    if abs(previous) < _EPS:
        return 0.0
    return float(np.clip(((current - previous) / abs(previous)) * 100.0,
                         -_DELTA_CLIP, _DELTA_CLIP))


def _extract_scalars(tick: dict[str, Any]) -> dict[str, float]:
    """Pull the 7 raw scalars we need from any tick dict (live or tape)."""
    return {
        "activePrice":    float(tick.get("activePrice",    0.0)),
        "microPrice":     float(tick.get("microPrice",     0.0)),
        "avgBinDepthUsd": float(tick.get("avgBinDepthUsd", 0.0)),
        "bidDepthUsd":    float(tick.get("bidDepthUsd",    0.0)),
        "askDepthSol":    float(tick.get("askDepthSol",    0.0)),
        "obiScore":       float(tick.get("obiScore",       0.0)),
        "activeBinId":    float(tick.get("activeBinId",    0.0)),
    }


def _compute_state(current: dict[str, float],
                   previous: dict[str, float]) -> np.ndarray:
    """Produce the 8-dimensional relative-delta state vector."""
    return np.array([
        _pct_delta(current["activePrice"],    previous["activePrice"]),    # [0]
        _pct_delta(current["microPrice"],     previous["microPrice"]),     # [1]
        _pct_delta(current["avgBinDepthUsd"], previous["avgBinDepthUsd"]), # [2]
        _pct_delta(current["bidDepthUsd"],    previous["bidDepthUsd"]),    # [3]
        _pct_delta(current["askDepthSol"],    previous["askDepthSol"]),    # [4]
        current["obiScore"],                                                # [5]
        float(np.clip(current["obiScore"] - previous["obiScore"],
                      -_DELTA_CLIP, _DELTA_CLIP)),                         # [6]
        current["activeBinId"] - previous["activeBinId"],                  # [7]
    ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Mock WebSocket server  (replays a tape at configurable speed)
# ──────────────────────────────────────────────────────────────────────────────

async def _mock_ws_server(
    tape_path:   Path,
    host:        str  = "127.0.0.1",
    port:        int  = 8765,
    tick_delay:  float = 0.5,    # seconds between ticks
) -> None:
    """
    Serve a historical tape over a local WebSocket so the Nerve can connect
    to it exactly like a live feed.  Used for integration testing without a
    live Solana node.

    Each JSON line from the tape is broadcast verbatim, preserving the
    exact field schema that the real collector emits.
    """
    try:
        import websockets.server as _wss
    except ImportError:
        raise ImportError("pip install websockets>=13")

    ticks: list[str] = []
    with tape_path.open() as f:
        for i, line in enumerate(f):
            if i < 150:  # skip first 150 ticks to reach volatility for demo
                continue
            line = line.strip()
            if line:
                ticks.append(line)

    print(f"[MockWS] Loaded {len(ticks)} ticks from {tape_path.name}")
    print(f"[MockWS] Serving ws://{host}:{port}  ({tick_delay}s/tick)")

    connections: set = set()

    async def handler(ws):
        connections.add(ws)
        try:
            await ws.wait_closed()
        finally:
            connections.discard(ws)

    async def broadcast():
        await asyncio.sleep(0.3)   # give clients time to connect
        for raw in ticks:
            if connections:
                await asyncio.gather(
                    *[ws.send(raw) for ws in list(connections)],
                    return_exceptions=True,
                )
            await asyncio.sleep(tick_delay)
        # Signal end-of-stream
        eos = json.dumps({"__eos__": True})
        if connections:
            await asyncio.gather(
                *[ws.send(eos) for ws in list(connections)],
                return_exceptions=True,
            )
        print("[MockWS] Tape exhausted. End-of-stream sent.")

    async with _wss.serve(handler, host, port):
        await broadcast()


# ──────────────────────────────────────────────────────────────────────────────
# SRMLiveNerve
# ──────────────────────────────────────────────────────────────────────────────

class SRMLiveNerve:
    """
    Async live-data ingestion + reasoning pipeline for the SRM system.

    Parameters
    ----------
    ws_uri : str
        WebSocket URI to connect to.
        • Real feed  : "wss://your-solana-dlmm-collector/ws"
        • Mock local : "ws://127.0.0.1:8765"
    quantizer : SymbolicQuantizer
        Pre-fitted quantizer for converting 8D state → symbol ID.
    graph : MarketTransitionGraph
        Pre-built transition graph for MCTS rollouts.
    bridge : SRMBridge
        Payload formatter.
    mcts_base_simulations : int
        Passed to MCTSPlanner.  Default 150 (fast but reasonable).
    icm : IntrinsicCuriosityModule | None
        Optional ICM for real curiosity rewards.  If None, MCTS uses its
        built-in placeholder reward.
    decision_queue_maxsize : int
        Max decisions that can pile up in the queue before the oldest is
        dropped.  Prevents unbounded memory growth if narration is slow.
    reconnect_delay : float
        Seconds to wait before reconnecting after a dropped connection.
    verbose : bool
        Print every raw tick to console.
    """

    def __init__(
        self,
        ws_uri:                 str,
        quantizer,              # SymbolicQuantizer
        graph,                  # MarketTransitionGraph
        bridge,                 # SRMBridge
        mcts_base_simulations:  int   = 150,
        icm                         = None,
        decision_queue_maxsize: int   = 32,
        reconnect_delay:        float = 3.0,
        verbose:                bool  = True,
    ) -> None:
        self.ws_uri               = ws_uri
        self.quantizer            = quantizer
        self.graph                = graph
        self.bridge               = bridge
        self.icm                  = icm
        self.reconnect_delay      = reconnect_delay
        self.verbose              = verbose

        # Build MCTS planner
        from srm_mcts import MCTSPlanner
        self.planner = MCTSPlanner(
            quantizer        = quantizer,
            graph            = graph,
            base_simulations = mcts_base_simulations,
            random_state     = 42,
        )

        # Async queue: HOT path puts decisions, COLD path (LLM) gets them
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=decision_queue_maxsize
        )

        # Running stats
        self.ticks_received:   int   = 0
        self.decisions_made:   int   = 0
        self.narrations_done:  int   = 0
        self._running:         bool  = False

        # ── Shadow Evaluator State ───────────────────────────────────────────
        self.eval_horizon: int = 5
        self.pending_evals: dict[int, dict[str, Any]] = {}  # tick -> record
        self.total_wins:    int = 0
        self.total_losses:  int = 0

        # Previous tick snapshot (maintained across reconnects)
        self._prev_snapshot: dict[str, float] | None = None
        self._prev_state:    np.ndarray | None       = None

        # Thread pool for blocking calls (MCTS search + ICM update + LLM speak)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="srm-worker"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # HOT PATH  — WebSocket listener + MCTS + Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def _record_prediction(self, tick_index: int, decision: int, current_price: float) -> None:
        """Store a decision for later grading by the ShadowEvaluator."""
        self.pending_evals[tick_index] = {
            "decision": decision,
            "price":    current_price,
        }

    def _evaluate_tick(self, current_tick: int, current_price: float) -> None:
        """
        Grade a past decision if it has matured (current_tick - horizon).
        Prints a log statement with the result and rolling accuracy.
        """
        target_tick = current_tick - self.eval_horizon
        if target_tick not in self.pending_evals:
            return

        record = self.pending_evals.pop(target_tick)
        decision = record["decision"]
        old_price = record["price"]
        
        price_delta = current_price - old_price
        
        # Grading logic
        result = "IGNORED"  # HOLD is ignored for directional binary accuracy
        if decision == 1:   # BUY
            if price_delta > 0:
                self.total_wins += 1
                result = "WIN"
            else:
                self.total_losses += 1
                result = "LOSS"
        elif decision == 2: # SELL
            if price_delta < 0:
                self.total_wins += 1
                result = "WIN"
            else:
                self.total_losses += 1
                result = "LOSS"
                
        total_graded = self.total_wins + self.total_losses
        if total_graded > 0:
            accuracy = (self.total_wins / total_graded) * 100
        else:
            accuracy = 0.0
            
        action_str = {0: "HOLD", 1: "BUY", 2: "SELL"}[decision]
        if result != "IGNORED":
            print(f"[Evaluator] Tick {target_tick} resolved. {action_str} was a {result}. "
                  f"Δ=${price_delta:+.4f}. Rolling Accuracy: {accuracy:.1f}% "
                  f"({self.total_wins}W / {self.total_losses}L)")

    async def _process_tick(self, raw_msg: str) -> None:
        """
        Parse one raw WebSocket message and run the full hot-path pipeline:
          raw_json → extract scalars → compute 8D state → encode symbol →
          MCTS search → format payload → enqueue

        Runs in the event loop.  MCTS search is offloaded to the thread pool
        so this coroutine yields control while the search runs.
        """
        try:
            tick = json.loads(raw_msg)
        except json.JSONDecodeError:
            return

        # End-of-stream sentinel from mock server
        if tick.get("__eos__"):
            print("\n[Nerve] End-of-stream received. Stopping.")
            self._running = False
            return

        self.ticks_received += 1
        current_snapshot = _extract_scalars(tick)

        # Need at least one previous tick to compute deltas
        if self._prev_snapshot is None:
            self._prev_snapshot = current_snapshot
            if self.verbose:
                print(f"[Nerve] Tick 0 seeded (prev_snapshot initialised)  "
                      f"price=${current_snapshot['activePrice']:.4f}")
            return

        # ── Compute 8D state ─────────────────────────────────────────────────
        state    = _compute_state(current_snapshot, self._prev_snapshot)
        prev_state = self._prev_state   # may be None for tick 2

        # ── ICM curiosity reward (non-blocking, best-effort) ─────────────────
        icm_reward: float | None = None
        if self.icm is not None and prev_state is not None:
            # get_reward() is ~130µs — fast enough to call inline
            # (no gradient update here; that's in the cold path)
            icm_reward = self.icm.get_reward(
                prev_state,
                0,       # action unknown at this point — use HOLD as proxy
                state,
            )

        # ── Symbol encoding ──────────────────────────────────────────────────
        symbol   = int(self.quantizer.encode(state))
        entropy  = self.graph.symbol_entropy(symbol)
        budget   = self.planner._compute_sim_budget(symbol)

        t_start  = time.perf_counter()

        # ── MCTS search (offloaded to thread pool) ───────────────────────────
        loop   = asyncio.get_event_loop()
        action, root = await loop.run_in_executor(
            self._executor,
            lambda: self.planner.search(symbol),
        )

        elapsed_mcts = (time.perf_counter() - t_start) * 1000

        # ── Build payload ─────────────────────────────────────────────────────
        q_values        = {a: node.Q for a, node in root.children.items()}
        top_transitions = self.graph.get_probabilities(symbol, top_k=3)
        actions_label   = {0: "HOLD", 1: "BUY", 2: "SELL"}

        full_payload = self.bridge.generate_payload(
            symbol          = symbol,
            entropy         = entropy,
            action          = action,
            q_values        = q_values,
            simulations     = budget,
            top_transitions = top_transitions,
            icm_reward      = icm_reward,
            extra           = {
                "tick_index":  self.ticks_received,
                "raw_price":   f"${current_snapshot['activePrice']:.4f}",
                "mcts_ms":     f"{elapsed_mcts:.0f}",
                "timestamp":   tick.get("timestamp", ""),
                "pool":        tick.get("poolAddress", ""),
            },
        )
        slim_payload = self.bridge.slim_payload(full_payload)

        decision_record = {
            "tick":         self.ticks_received,
            "symbol":       symbol,
            "entropy":      entropy,
            "action":       action,
            "action_label": actions_label[action],
            "q_values":     q_values,
            "mcts_ms":      elapsed_mcts,
            "full_payload": full_payload,
            "slim_payload": slim_payload,
            "price":        current_snapshot["activePrice"],
            "icm_reward":   icm_reward,
        }

        # ── Enqueue (non-blocking: drop oldest if full) ───────────────────────
        try:
            self._queue.put_nowait(decision_record)
            self.decisions_made += 1
        except asyncio.QueueFull:
            # Drop the oldest item to make room for the newest
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(decision_record)
            self.decisions_made += 1
            if self.verbose:
                print("[Nerve] ⚠️  Queue full — dropped oldest decision")

        # ── ICM update (cold — fire-and-forget into thread pool) ─────────────
        if self.icm is not None and prev_state is not None:
            loop.run_in_executor(
                self._executor,
                lambda s=prev_state, a=action, ns=state: self.icm.push_and_update(
                    s, a, ns, update_every=8
                ),
            )

        # Advance prev pointers
        self._prev_snapshot = current_snapshot
        self._prev_state    = state

        # ── Shadow Evaluator Hook ────────────────────────────────────────────
        self._record_prediction(
            tick_index=self.ticks_received,
            decision=action,
            current_price=current_snapshot['activePrice']
        )
        self._evaluate_tick(
            current_tick=self.ticks_received,
            current_price=current_snapshot['activePrice']
        )

        if self.verbose:
            print(
                f"[Nerve] tick={self.ticks_received:>4}  "
                f"#{symbol:04d}  "
                f"entropy={entropy:.2f}  "
                f"budget={budget:>4}  "
                f"→ {actions_label[action]:<4}  "
                f"({elapsed_mcts:.0f}ms)  "
                f"price=${current_snapshot['activePrice']:.4f}"
            )

    async def _ws_listener(self) -> None:
        """
        Persistent WebSocket listener with exponential-backoff reconnection.

        Runs until self._running is set to False (by EOS sentinel or by
        external stop signal).
        """
        import websockets

        backoff = self.reconnect_delay
        max_backoff = 60.0

        while self._running:
            try:
                print(f"[Nerve] Connecting → {self.ws_uri}")
                async with websockets.connect(
                    self.ws_uri,
                    ping_interval   = 20,
                    ping_timeout    = 10,
                    close_timeout   = 5,
                    max_size        = 2**20,   # 1 MB max message
                ) as ws:
                    print(f"[Nerve] ✅ Connected.")
                    backoff = self.reconnect_delay   # reset on successful connect

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        await self._process_tick(raw_msg)

            except Exception as exc:
                if not self._running:
                    break
                print(f"[Nerve] ⚠️  WebSocket error: {exc!r}  "
                      f"Reconnecting in {backoff:.0f}s …")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, max_backoff)

    # ──────────────────────────────────────────────────────────────────────────
    # COLD PATH  — LLM narration consumer
    # ──────────────────────────────────────────────────────────────────────────

    async def _narration_loop(self, mouth=None) -> None:
        """
        Continuously drain the decision queue and narrate each decision via
        SRMMouth (if provided).

        The LLM call is offloaded to the thread pool so the event loop
        stays responsive to new ticks while narration is generating.
        If no mouth is provided, decisions are printed as raw JSON.
        """
        actions_label = {0: "HOLD", 1: "BUY", 2: "SELL"}
        loop = asyncio.get_event_loop()

        while self._running or not self._queue.empty():
            try:
                record = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            print("\n" + "═" * 64)
            print(f"  Decision #{self.narrations_done + 1}  |  "
                  f"Tick {record['tick']}  |  "
                  f"Symbol #{record['symbol']:04d}  |  "
                  f"{record['action_label']}  |  "
                  f"price={record['price']:.4f}")
            print("═" * 64)

            if mouth is not None:
                # Stream LLM narration in thread pool — non-blocking
                print("\n  [SRM Brain] ", end="", flush=True)

                def _speak_blocking(payload=record["slim_payload"]):
                    result = []
                    for token in mouth.speak(payload):
                        # Print inline as tokens arrive
                        print(token, end="", flush=True)
                        result.append(token)
                    return "".join(result)

                narration = await loop.run_in_executor(self._executor, _speak_blocking)
                print(f"\n")
            else:
                # No mouth — print slim payload directly
                print(record["slim_payload"])

            self.narrations_done += 1
            self._queue.task_done()

    # ──────────────────────────────────────────────────────────────────────────
    # Public control
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self, mouth=None, max_ticks: int | None = None) -> None:
        """
        Start the Nerve.  Runs until the WebSocket closes, max_ticks is
        reached, or the task is cancelled.

        Parameters
        ----------
        mouth : SRMMouth | None
            If provided, uses the LLM to narrate each decision.
            If None, decisions are printed as slim JSON.
        max_ticks : int | None
            Stop after this many ticks.  None = run forever.
        """
        self._running = True
        print(f"[Nerve] Starting.  Queue maxsize={self._queue.maxsize}")

        if max_ticks is not None:
            # Wrap listener to stop after N ticks
            original_process = self._process_tick

            async def _limited_process(raw_msg: str) -> None:
                await original_process(raw_msg)
                if self.ticks_received >= max_ticks:
                    self._running = False

            self._process_tick = _limited_process

        try:
            await asyncio.gather(
                self._ws_listener(),
                self._narration_loop(mouth),
            )
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            print(f"\n[Nerve] Stopped.  "
                  f"ticks={self.ticks_received}  "
                  f"decisions={self.decisions_made}  "
                  f"narrations={self.narrations_done}")

    def stop(self) -> None:
        """Signal the nerve to stop after the current tick."""
        self._running = False

    def stats(self) -> dict[str, Any]:
        return {
            "ticks_received":  self.ticks_received,
            "decisions_made":  self.decisions_made,
            "narrations_done": self.narrations_done,
            "evalualator_wins": self.total_wins,
            "evaluator_losses": self.total_losses,
            "evaluator_acc":   f"{(self.total_wins / max(1, self.total_wins + self.total_losses))*100:.1f}%",
            "queue_depth":     self._queue.qsize(),
            "ws_uri":          self.ws_uri,
        }

    def __repr__(self) -> str:
        return (
            f"SRMLiveNerve("
            f"uri='{self.ws_uri}', "
            f"ticks={self.ticks_received}, "
            f"queue={self._queue.qsize()}/{self._queue.maxsize})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Execution block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).parent

    # ── Configuration ─────────────────────────────────────────────────────────
    MOCK_HOST  = "127.0.0.1"
    MOCK_PORT  = 8765
    WS_URI     = f"ws://{MOCK_HOST}:{MOCK_PORT}"
    TICK_DELAY = 0.02    # extremely fast so we can run 100 ticks quickly
    MAX_TICKS  = 100     # let 95 evaluations resolve
    USE_MOUTH  = False   # Set True to enable LLM narration (~40s/narration on CPU)
                         # Set False to stream slim JSON instead (instant)

    print("=" * 64)
    print("  SRMLiveNerve — Integration Demo")
    print("=" * 64)

    # ── Load components ───────────────────────────────────────────────────────
    print("\n[Boot] Loading quantizer + graph …")
    from srm_quantizer import SymbolicQuantizer
    from srm_transition_graph import MarketTransitionGraph
    from srm_bridge import SRMBridge

    quantizer = SymbolicQuantizer.load(ROOT / "srm_quantizer_512.pkl")
    graph     = MarketTransitionGraph.load(ROOT / "srm_graph_512.pkl", quantizer)
    bridge    = SRMBridge(include_timestamp=True, indent=2)

    # Optional: load ICM (comment out if not needed)
    icm = None
    # from srm_icm import IntrinsicCuriosityModule
    # icm = IntrinsicCuriosityModule()

    # Optional: load mouth (comment out for JSON-only mode)
    mouth = None
    if USE_MOUTH:
        print("[Boot] Loading SRMMouth (Qwen 0.5B) …")
        from srm_mouth import SRMMouth, DEFAULT_MODEL_PATH
        try:
            mouth = SRMMouth(model_path=DEFAULT_MODEL_PATH, n_ctx=1024, verbose=False)
        except FileNotFoundError as e:
            print(f"[Boot] SRMMouth unavailable: {e}\n  Running in JSON-only mode.")
            mouth = None

    # ── Pick the most volatile tape for the demo ──────────────────────────────
    tape_path = ROOT / "solana-tapes" / \
                "market_tape_volatile-bear-dump-stable_2026-02-23_16-54-01.jsonl"
    if not tape_path.exists():
        # Fall back to any tape
        tapes = sorted((ROOT / "solana-tapes").glob("*.jsonl"))
        tape_path = next((t for t in tapes if t.stat().st_size > 50_000), tapes[0])

    print(f"[Boot] Tape: {tape_path.name}")
    print(f"[Boot] WS  : {WS_URI}")
    print(f"[Boot] Mode: {'LLM narration' if USE_MOUTH and mouth else 'JSON-only (instant)'}")
    print(f"[Boot] Ticks to process: {MAX_TICKS}\n")

    # ── Build the Nerve ───────────────────────────────────────────────────────
    nerve = SRMLiveNerve(
        ws_uri                = WS_URI,
        quantizer             = quantizer,
        graph                 = graph,
        bridge                = bridge,
        icm                   = icm,
        mcts_base_simulations = 150,
        decision_queue_maxsize= 32,
        reconnect_delay       = 3.0,
        verbose               = True,
    )

    # ── Run everything ────────────────────────────────────────────────────────
    async def _main():
        # Start the mock WebSocket server as a background task
        server_task = asyncio.create_task(
            _mock_ws_server(
                tape_path  = tape_path,
                host       = MOCK_HOST,
                port       = MOCK_PORT,
                tick_delay = TICK_DELAY,
            )
        )

        # Give the server a moment to bind before connecting
        await asyncio.sleep(0.4)

        try:
            await nerve.run(mouth=mouth, max_ticks=MAX_TICKS)
        finally:
            server_task.cancel()
            try:
                await server_task
            except (asyncio.CancelledError, Exception):
                pass

        # Summary
        print("\n" + "=" * 64)
        print("  Session Summary")
        print("=" * 64)
        s = nerve.stats()
        for k, v in s.items():
            print(f"  {k:<20}: {v}")

    asyncio.run(_main())
