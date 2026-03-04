"""
srm_environment.py

Tape-driven Reinforcement Learning environment for the Symbolic Reasoning Model (SRM).
Simulates a live Solana DLMM market by replaying historical `.jsonl` tapes.

Key design decisions
─────────────────────
• Price-agnostic state: every raw value is converted to a relative % delta from
  the previous tick, so the agent never sees a raw dollar price.
• Flat NumPy array output: ultra-fast, ready for vectorised inference.
• Minimal dependencies: only stdlib + numpy.
• The step() signature follows the modern Gymnasium convention:
      next_state, reward, terminated, truncated, info

State vector layout (8 dimensions)
────────────────────────────────────
  [0] Δ activePrice        (%)   price micro-movement
  [1] Δ microPrice         (%)   micro-price movement (bid/ask mid)
  [2] Δ avgBinDepthUsd     (%)   total liquidity depth change
  [3] Δ bidDepthUsd        (%)   bid-side depth change
  [4] Δ askDepthSol        (%)   ask-side depth change (denominated in SOL)
  [5]   obiScore           (raw) order-book imbalance [-1, 1]; already relative
  [6] Δ obiScore           (%)   change in OBI
  [7] Δ activeBinId        (raw) integer bin drift (not a % – bins are discrete)

All Δ values are clipped to [-100, +100] % to prevent exploding gradients when
a denominator is near zero.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

STATE_DIM = 8          # Length of the flat state vector
DELTA_CLIP = 100.0     # Maximum absolute % change (avoids NaN / Inf states)
EPS = 1e-10            # Protect against division-by-zero


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _pct_delta(current: float, previous: float) -> float:
    """Return the percentage change from `previous` to `current`, clipped."""
    if abs(previous) < EPS:
        return 0.0
    return float(np.clip(((current - previous) / abs(previous)) * 100.0,
                         -DELTA_CLIP, DELTA_CLIP))


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class SRMEnvironment:
    """
    Step-by-step Solana market simulator powered by historical .jsonl tapes.

    Parameters
    ----------
    tape_path : str | Path
        Absolute or relative path to a `.jsonl` tape file.
        Each line must be a self-contained JSON object as produced by the
        live market data collector.
    start_index : int, optional
        Tick index to begin playback from (default 0). Useful for splitting
        a tape into train / eval segments.
    end_index : int | None, optional
        Inclusive last tick index (default: last tick in the tape).
    preload : bool, optional
        If True (default), the entire tape is loaded into RAM on __init__.
        Set to False for very large tapes to enable lazy line-by-line reading
        (not yet implemented – reserved for future work).

    Attributes
    ----------
    observation_space_shape : tuple[int]
        Shape of the state array returned by reset() / step().  Always (8,).
    action_space_n : int
        Placeholder count of discrete actions.  Set to 3 (HOLD/BUY/SELL).
    """

    observation_space_shape: tuple[int, ...] = (STATE_DIM,)
    action_space_n: int = 3  # 0: HOLD, 1: BUY, 2: SELL

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        tape_path: str | Path,
        start_index: int = 0,
        end_index: int | None = None,
        preload: bool = True,
    ) -> None:
        self.tape_path = Path(tape_path)
        if not self.tape_path.exists():
            raise FileNotFoundError(f"Tape not found: {self.tape_path}")

        # Load tape ────────────────────────────────────────────────────────────
        print(f"[SRMEnvironment] Loading tape: {self.tape_path.name}  …", end="", flush=True)
        self._ticks: list[dict[str, Any]] = self._load_tape(self.tape_path)
        print(f"  {len(self._ticks):,} ticks loaded.")

        # Slice the tape if requested
        self._start_index: int = max(0, start_index)
        self._end_index: int = (
            min(end_index, len(self._ticks) - 1)
            if end_index is not None
            else len(self._ticks) - 1
        )
        if self._start_index >= self._end_index:
            raise ValueError(
                f"start_index ({self._start_index}) must be < end_index ({self._end_index})"
            )

        self._playback_ticks: list[dict[str, Any]] = self._ticks[
            self._start_index : self._end_index + 1
        ]
        self._total_ticks: int = len(self._playback_ticks)

        # Internal cursor & previous-tick snapshot initialised by reset()
        self._cursor: int = 0
        self._prev_snapshot: dict[str, float] | None = None
        self._reset_called: bool = False

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_tape(path: Path) -> list[dict[str, Any]]:
        """Read every non-empty line in the .jsonl file and parse it."""
        ticks: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ticks.append(json.loads(raw))
                except json.JSONDecodeError:
                    # Skip corrupted lines silently (tape may be truncated)
                    continue
        return ticks

    @staticmethod
    def _extract_scalars(tick: dict[str, Any]) -> dict[str, float]:
        """Pull the scalar market metrics we care about out of a raw tick dict."""
        return {
            "activePrice":   float(tick.get("activePrice",   0.0)),
            "microPrice":    float(tick.get("microPrice",    0.0)),
            "avgBinDepthUsd": float(tick.get("avgBinDepthUsd", 0.0)),
            "bidDepthUsd":   float(tick.get("bidDepthUsd",   0.0)),
            "askDepthSol":   float(tick.get("askDepthSol",   0.0)),
            "obiScore":      float(tick.get("obiScore",      0.0)),
            "activeBinId":   float(tick.get("activeBinId",   0.0)),
        }

    def _compute_state(
        self,
        current: dict[str, float],
        previous: dict[str, float],
    ) -> np.ndarray:
        """
        Convert two consecutive raw-scalar snapshots into the 8-dim state vector.

        All values are percentage deltas from the *previous* tick except for:
          • obiScore  → kept as-is (already a normalised imbalance ratio)
          • Δ obiScore → absolute change in OBI (not a % – would be meaningless)
          • Δ activeBinId → raw integer drift (bins are discrete ordinal steps)
        """
        state = np.array([
            _pct_delta(current["activePrice"],    previous["activePrice"]),    # [0]
            _pct_delta(current["microPrice"],      previous["microPrice"]),    # [1]
            _pct_delta(current["avgBinDepthUsd"],  previous["avgBinDepthUsd"]),# [2]
            _pct_delta(current["bidDepthUsd"],     previous["bidDepthUsd"]),   # [3]
            _pct_delta(current["askDepthSol"],     previous["askDepthSol"]),   # [4]
            current["obiScore"],                                               # [5]
            float(np.clip(                                                     # [6]
                current["obiScore"] - previous["obiScore"],
                -DELTA_CLIP, DELTA_CLIP
            )),
            current["activeBinId"] - previous["activeBinId"],                 # [7]
        ], dtype=np.float32)
        return state

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Rewind the tape to the beginning and return the initial state.

        The first state is computed as the delta between tick[0] and tick[1],
        so the agent always receives a *relative* snapshot even on reset.

        Returns
        -------
        initial_state : np.ndarray, shape (8,)
            The first relative state vector.
        info : dict
            Raw tick data for logging / debugging.
        """
        self._cursor = 1  # We need tick[0] as the 'previous', tick[1] as 'current'
        self._reset_called = True

        prev_tick    = self._playback_ticks[0]
        current_tick = self._playback_ticks[1]

        self._prev_snapshot = self._extract_scalars(prev_tick)
        current_snapshot    = self._extract_scalars(current_tick)

        initial_state = self._compute_state(current_snapshot, self._prev_snapshot)
        self._prev_snapshot = current_snapshot

        info: dict[str, Any] = {
            "tick_index":    self._cursor,
            "timestamp":     current_tick.get("timestamp"),
            "slot":          current_tick.get("slot"),
            "activePrice":   current_tick.get("activePrice"),
            "microPrice":    current_tick.get("microPrice"),
            "bidDepthUsd":   current_tick.get("bidDepthUsd"),
            "askDepthSol":   current_tick.get("askDepthSol"),
            "obiScore":      current_tick.get("obiScore"),
            "activeBinId":   current_tick.get("activeBinId"),
            "poolAddress":   current_tick.get("poolAddress"),
            "ticks_total":   self._total_ticks,
        }
        return initial_state, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Advance the tape by one tick.

        Parameters
        ----------
        action : int
            The agent's discrete action (0=HOLD, 1=BUY, 2=SELL).
            Currently unused in state/reward logic — the environment is a pure
            market observer.  Action handling will be wired in the reward layer.

        Returns
        -------
        next_state : np.ndarray, shape (8,)
            Relative % deltas from the previous tick.
        reward : float
            Placeholder — always 0.0.  The Intrinsic Curiosity Module (ICM)
            reward will be computed externally and injected here later.
        terminated : bool
            True when the end of the tape is reached.
        truncated : bool
            Always False.  Reserved for time-limit truncation.
        info : dict
            Raw tick data for logging, debugging, and reward computation.
        """
        if not self._reset_called:
            raise RuntimeError("Call reset() before step().")

        # Advance cursor
        self._cursor += 1
        terminated = self._cursor >= self._total_ticks

        if terminated:
            # Return the last valid state again; tape is exhausted
            dummy_state = np.zeros(STATE_DIM, dtype=np.float32)
            info: dict[str, Any] = {
                "tick_index":  self._cursor,
                "terminated":  True,
                "ticks_total": self._total_ticks,
            }
            return dummy_state, 0.0, True, False, info

        current_tick     = self._playback_ticks[self._cursor]
        current_snapshot = self._extract_scalars(current_tick)

        # Compute state relative to previous tick
        next_state = self._compute_state(current_snapshot, self._prev_snapshot)  # type: ignore[arg-type]

        # Advance the previous-snapshot pointer
        self._prev_snapshot = current_snapshot

        # ── Reward: placeholder ─────────────────────────────────────────────
        reward: float = 0.0
        # TODO: Wire Intrinsic Curiosity Module (ICM) here.
        # The ICM will measure the agent's prediction error on next_state
        # and return an intrinsic reward proportional to 'surprise'.

        info = {
            "tick_index":    self._cursor,
            "timestamp":     current_tick.get("timestamp"),
            "slot":          current_tick.get("slot"),
            "activePrice":   current_tick.get("activePrice"),
            "microPrice":    current_tick.get("microPrice"),
            "bidDepthUsd":   current_tick.get("bidDepthUsd"),
            "askDepthSol":   current_tick.get("askDepthSol"),
            "avgBinDepthUsd": current_tick.get("avgBinDepthUsd"),
            "obiScore":      current_tick.get("obiScore"),
            "activeBinId":   current_tick.get("activeBinId"),
            "poolAddress":   current_tick.get("poolAddress"),
            "action":        action,
            "ticks_total":   self._total_ticks,
            "ticks_remaining": self._total_ticks - self._cursor - 1,
        }
        return next_state, reward, False, False, info

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def total_ticks(self) -> int:
        """Total number of playable ticks in this episode."""
        return self._total_ticks

    @property
    def current_tick_index(self) -> int:
        """Zero-based index of the *last returned* tick."""
        return self._cursor

    def __repr__(self) -> str:
        return (
            f"SRMEnvironment("
            f"tape='{self.tape_path.name}', "
            f"ticks={self._total_ticks:,}, "
            f"state_dim={STATE_DIM})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  (run directly: python srm_environment.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    TAPE_DIR = Path(__file__).parent / "solana-tapes"
    tapes = sorted(glob.glob(str(TAPE_DIR / "*.jsonl")))

    if not tapes:
        raise SystemExit("No .jsonl tapes found in solana-tapes/")

    # Pick the smallest tape that isn't trivially tiny (>= 50 ticks)
    # We do a quick pre-check by size proxy (each tick is ~2 KB on average).
    # Tapes below ~100 KB are unlikely to have 50+ ticks.
    sized = [(os.path.getsize(t), t) for t in tapes if os.path.getsize(t) >= 50_000]
    if not sized:
        raise SystemExit("No suitable tapes found (need >= 50 KB).")
    test_tape = min(sized)[1]
    print(f"\n{'='*60}")
    print(f"  SRMEnvironment Smoke Test")
    print(f"  Tape: {Path(test_tape).name}")
    print(f"{'='*60}\n")

    env = SRMEnvironment(test_tape)
    print(env)
    print(f"State vector dim : {env.observation_space_shape}")
    print(f"Action count     : {env.action_space_n}")
    print()

    # Reset ────────────────────────────────────────────────────────────────────
    state, info = env.reset()
    print(f"Initial state (tick {info['tick_index']}):")
    labels = [
        "Δ activePrice %",
        "Δ microPrice  %",
        "Δ avgDepth    %",
        "Δ bidDepth    %",
        "Δ askDepthSol %",
        "  obiScore    ",
        "Δ obiScore    ",
        "Δ activeBinId ",
    ]
    for label, val in zip(labels, state):
        print(f"    {label}: {val:+.6f}")
    print(f"  Raw price snapshot : ${info['activePrice']:.4f}")
    print()

    # Step through a few ticks ─────────────────────────────────────────────────
    N_STEPS = 10
    print(f"Stepping through {N_STEPS} ticks with HOLD (action=0):")
    print(f"  {'Tick':>5}  {'Δ Price %':>12}  {'OBI':>8}  {'Δ OBI':>8}  {'ΔBin':>6}  {'Raw Price':>12}")
    print(f"  {'-'*60}")

    for i in range(N_STEPS):
        next_state, reward, terminated, truncated, info = env.step(action=0)
        raw_price = info.get('activePrice')
        price_str = f"${raw_price:>10.4f}" if raw_price is not None else "[end-of-tape]"
        print(
            f"  {info['tick_index']:>5}  "
            f"{next_state[0]:>+12.4f}  "
            f"{next_state[5]:>+8.4f}  "
            f"{next_state[6]:>+8.4f}  "
            f"{next_state[7]:>+6.0f}  "
            f"{price_str}"
        )
        if terminated:
            print("  [TAPE EXHAUSTED]")
            break

    print()
    print("✅ Smoke test passed — environment is operational.")
