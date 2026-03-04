"""
srm_quantizer.py

Converts the continuous 8-dimensional market state vectors produced by
SRMEnvironment into discrete integer Symbol IDs — the vocabulary of the
Neuro-Symbolic Reasoning Graph.

Architecture role
─────────────────
  SRMEnvironment  →  8D float32 state  →  SymbolicQuantizer  →  int symbol [0, K)
  
  Each integer is a node in the reasoning graph.  A sequence of integers
  produced by stepping through a tape becomes a "symbol sentence" that the
  MCTS planner can reason over.

Algorithm choice: MiniBatchKMeans
──────────────────────────────────
  • O(n) memory instead of O(n²) — safe for 500k+ sample fits.
  • Iterative partial_fit allows streaming data without loading tapes twice.
  • Converges to near-identical cluster geometry as full KMeans on large datasets.

Default vocabulary size: 512 symbols
  Powers of 2 are convenient for future embedding lookups (embedding tables,
  2D bit-packing, etc.).  Configurable via n_clusters at construction time.
"""

from __future__ import annotations

import os
import glob
import time
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# Local
from srm_environment import SRMEnvironment

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_N_CLUSTERS  = 512          # Size of the symbol vocabulary
DEFAULT_BATCH_SIZE  = 4_096        # MiniBatchKMeans mini-batch size
DEFAULT_MAX_SAMPLES = 200_000      # Hard cap on samples collected during fit()
STATE_DIM           = 8            # Must match SRMEnvironment.observation_space_shape


# ──────────────────────────────────────────────────────────────────────────────
# SymbolicQuantizer
# ──────────────────────────────────────────────────────────────────────────────

class SymbolicQuantizer:
    """
    Fits a K-Means vocabulary over Solana market state vectors and encodes
    individual states as discrete integer Symbol IDs.

    Parameters
    ----------
    n_clusters : int
        Number of cluster centroids = size of the symbol vocabulary.
        Default: 512.
    batch_size : int
        Mini-batch size for MiniBatchKMeans.  Larger = more accurate per
        iteration; smaller = faster per pass.  Default: 4096.
    random_state : int
        Seed for reproducibility.  Default: 42.

    Attributes
    ----------
    is_fitted : bool
        True after fit() has been called successfully.
    n_samples_seen : int
        Total number of state vectors used during the last fit.
    inertia_ : float
        Final KMeans inertia (sum of squared distances to nearest centroid).
        Lower is better — useful to track across re-fits.
    """

    def __init__(
        self,
        n_clusters:   int = DEFAULT_N_CLUSTERS,
        batch_size:   int = DEFAULT_BATCH_SIZE,
        random_state: int = 42,
    ) -> None:
        self.n_clusters   = n_clusters
        self.batch_size   = batch_size
        self.random_state = random_state

        self._kmeans: MiniBatchKMeans | None = None
        self._scaler: StandardScaler | None  = None

        self.is_fitted:      bool  = False
        self.n_samples_seen: int   = 0
        self.inertia_:       float = float("inf")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _collect_states(
        self,
        tape_paths:  list[str | Path],
        max_samples: int,
    ) -> np.ndarray:
        """
        Walk through every tape in tape_paths, step the SRMEnvironment, and
        accumulate raw 8D state vectors into a single NumPy array.

        Sampling strategy
        ─────────────────
        States are collected sequentially from each tape.  Once max_samples
        is reached the loop exits early.  Tapes are shuffled so that no single
        regime dominates the first max_samples vectors.
        """
        tape_paths = list(tape_paths)
        # Shuffle so we sample across different market regimes even if we hit
        # the cap early
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(tape_paths)

        buffer: list[np.ndarray] = []
        total = 0

        print(f"\n[Quantizer] Collecting states from {len(tape_paths)} tape(s). "
              f"Cap: {max_samples:,}")

        for tape_idx, tape_path in enumerate(tape_paths):
            if total >= max_samples:
                break

            tape_path = Path(tape_path)
            if tape_path.stat().st_size == 0:
                continue  # skip empty tapes

            try:
                env = SRMEnvironment(tape_path)
                if env.total_ticks < 3:
                    continue  # not enough data
            except Exception as exc:
                print(f"  [skip] {tape_path.name}: {exc}")
                continue

            state, _ = env.reset()
            tape_states: list[np.ndarray] = [state]

            done = False
            while not done:
                state, _, done, _, _ = env.step(action=0)
                if not done:
                    tape_states.append(state)
                if total + len(tape_states) >= max_samples:
                    break

            chunk = np.stack(tape_states, axis=0)  # (T, 8)
            buffer.append(chunk)
            total += len(chunk)

            pct = (tape_idx + 1) / len(tape_paths) * 100
            print(f"  [{tape_idx+1:>3}/{len(tape_paths)}] {tape_path.name:<55} "
                  f"+{len(chunk):>6,} ticks  | total={total:>8,}  ({pct:.0f}%)")

        if not buffer:
            raise RuntimeError("No valid states collected — check your tape paths.")

        all_states = np.concatenate(buffer, axis=0)

        # Trim to exact cap
        if len(all_states) > max_samples:
            all_states = all_states[:max_samples]

        return all_states  # shape (N, 8)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        tape_paths:  Sequence[str | Path],
        max_samples: int = DEFAULT_MAX_SAMPLES,
    ) -> "SymbolicQuantizer":
        """
        Collect state vectors from the provided tapes, fit a StandardScaler
        (so all 8 features are zero-mean / unit-variance), then fit
        MiniBatchKMeans to produce n_clusters centroids.

        Parameters
        ----------
        tape_paths : sequence of str | Path
            Paths to .jsonl tape files.
        max_samples : int
            Maximum number of state vectors to use for fitting.
            More samples → better cluster geometry; 50k–200k is a good range.

        Returns
        -------
        self
        """
        t0 = time.perf_counter()

        # 1. Collect raw states ────────────────────────────────────────────────
        X = self._collect_states(list(tape_paths), max_samples)
        self.n_samples_seen = len(X)
        print(f"\n[Quantizer] Collected {self.n_samples_seen:,} state vectors "
              f"({X.nbytes / 1_048_576:.1f} MB in RAM)")

        # 2. Scale features ────────────────────────────────────────────────────
        # MiniBatchKMeans is distance-based; we must normalize so that a large-
        # variance feature (e.g. Δ depth%) doesn't dominate Δ price%.
        print(f"[Quantizer] Fitting StandardScaler …")
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # 3. Fit KMeans ────────────────────────────────────────────────────────
        print(f"[Quantizer] Fitting MiniBatchKMeans  "
              f"(n_clusters={self.n_clusters}, batch={self.batch_size}) …")
        self._kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            n_init=5,          # number of random initialisations (best kept)
            max_iter=300,
            reassignment_ratio=0.01,  # re-seed stale clusters
            random_state=self.random_state,
            verbose=0,
            compute_labels=True,
        )
        self._kmeans.fit(X_scaled)
        self.inertia_ = float(self._kmeans.inertia_)
        self.is_fitted = True

        elapsed = time.perf_counter() - t0
        print(f"[Quantizer] ✅ Fit complete in {elapsed:.1f}s  |  "
              f"inertia={self.inertia_:,.1f}  |  "
              f"vocab={self.n_clusters} symbols")

        return self

    def encode(self, state: np.ndarray) -> int:
        """
        Map a single 8D state vector to its nearest centroid's integer ID.

        This is the hot-path call during inference — kept as lean as possible.
        The scaler transform + KMeans predict together run in ~5–15 µs.

        Parameters
        ----------
        state : np.ndarray, shape (8,) or (1, 8)
            Raw state vector as returned by SRMEnvironment.step() or .reset().

        Returns
        -------
        int
            Symbol ID in [0, n_clusters).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() or load() before encode().")

        # Ensure shape (1, 8)
        vec = np.asarray(state, dtype=np.float32).reshape(1, STATE_DIM)
        scaled = self._scaler.transform(vec)              # type: ignore[union-attr]
        symbol_id: int = int(self._kmeans.predict(scaled)[0])  # type: ignore[union-attr]
        return symbol_id

    def encode_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Vectorised batch encoding — encode many states at once (e.g., a full
        tape episode).  Significantly faster than calling encode() in a loop.

        Parameters
        ----------
        states : np.ndarray, shape (N, 8)

        Returns
        -------
        np.ndarray of int, shape (N,)
            Symbol IDs.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() or load() before encode_batch().")

        vec = np.asarray(states, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, STATE_DIM)
        scaled = self._scaler.transform(vec)              # type: ignore[union-attr]
        return self._kmeans.predict(scaled)               # type: ignore[union-attr]

    def centroid(self, symbol_id: int) -> np.ndarray:
        """
        Decode a symbol ID back to its centroid vector in *original* (unscaled)
        space.  Useful for interpretability and sanity checks.

        Returns
        -------
        np.ndarray, shape (8,)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() or load() before centroid().")
        c_scaled = self._kmeans.cluster_centers_[symbol_id].reshape(1, -1)  # type: ignore[union-attr]
        return self._scaler.inverse_transform(c_scaled)[0]                  # type: ignore[union-attr]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Persist the fitted quantizer (KMeans model + scaler + metadata) to
        disk using joblib's compressed pickle format.

        Parameters
        ----------
        path : str | Path
            Output file path.  Convention: ``srm_quantizer_512.pkl``.
        """
        if not self.is_fitted:
            raise RuntimeError("Nothing to save — call fit() first.")

        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "n_clusters":   self.n_clusters,
            "batch_size":   self.batch_size,
            "random_state": self.random_state,
            "n_samples_seen": self.n_samples_seen,
            "inertia_":     self.inertia_,
            "kmeans":       self._kmeans,
            "scaler":       self._scaler,
        }
        joblib.dump(payload, path, compress=3)
        size_kb = path.stat().st_size / 1024
        print(f"[Quantizer] 💾 Saved to {path}  ({size_kb:.1f} KB)")

    @classmethod
    def load(cls, path: str | Path) -> "SymbolicQuantizer":
        """
        Load a previously saved quantizer from disk and return a ready-to-use
        instance (is_fitted=True).

        Parameters
        ----------
        path : str | Path
            Path to the .pkl file written by save().

        Returns
        -------
        SymbolicQuantizer
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Quantizer file not found: {path}")

        payload = joblib.load(path)

        instance = cls(
            n_clusters=payload["n_clusters"],
            batch_size=payload["batch_size"],
            random_state=payload["random_state"],
        )
        instance._kmeans        = payload["kmeans"]
        instance._scaler        = payload["scaler"]
        instance.n_samples_seen = payload["n_samples_seen"]
        instance.inertia_       = payload["inertia_"]
        instance.is_fitted      = True

        print(f"[Quantizer] 📂 Loaded vocab={instance.n_clusters} symbols  |  "
              f"trained on {instance.n_samples_seen:,} samples  |  "
              f"inertia={instance.inertia_:,.1f}")
        return instance

    def __repr__(self) -> str:
        status = (f"fitted, {self.n_samples_seen:,} samples, "
                  f"inertia={self.inertia_:,.0f}")  if self.is_fitted else "unfitted"
        return f"SymbolicQuantizer(n_clusters={self.n_clusters}, {status})"


# ──────────────────────────────────────────────────────────────────────────────
# Execution block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    TAPE_DIR   = Path(__file__).parent / "solana-tapes"
    SAVE_PATH  = Path(__file__).parent / "srm_quantizer_512.pkl"

    FEATURE_LABELS = [
        "Δ activePrice %",
        "Δ microPrice  %",
        "Δ avgDepthUsd %",
        "Δ bidDepthUsd %",
        "Δ askDepthSol %",
        "  obiScore    ",
        "Δ obiScore    ",
        "Δ activeBinId ",
    ]

    # ── 1. Gather all non-empty tapes ─────────────────────────────────────────
    all_tapes = sorted(
        [p for p in TAPE_DIR.glob("*.jsonl") if p.stat().st_size >= 50_000],
        key=lambda p: p.stat().st_size,
    )
    print(f"Found {len(all_tapes)} usable tapes (≥ 50 KB) in {TAPE_DIR}/")

    # ── 2. Fit (or reload if already saved) ───────────────────────────────────
    if SAVE_PATH.exists():
        print(f"\n[Main] Pre-trained quantizer found at {SAVE_PATH}.")
        print("[Main] Delete it to force a re-fit.  Loading …\n")
        q = SymbolicQuantizer.load(SAVE_PATH)
    else:
        # Use every tape available; fit() will cap at max_samples internally
        q = SymbolicQuantizer(n_clusters=512)
        q.fit(all_tapes, max_samples=200_000)
        q.save(SAVE_PATH)

    print(f"\n{q}\n")
    print("=" * 68)

    # ── 3. Demo: pick a few tapes, step through them, print symbol stream ─────
    demo_tapes = random.sample(all_tapes, min(3, len(all_tapes)))

    for tape_path in demo_tapes:
        print(f"\n{'─'*68}")
        print(f"  Tape: {tape_path.name}")
        print(f"{'─'*68}")

        env = SRMEnvironment(tape_path)
        state, info = env.reset()

        print(f"  {'Tick':>5}  {'Symbol':>7}  │  State vector (8D)")
        print(f"  {'─'*5}  {'─'*7}  │  {'─'*44}")

        for step_i in range(12):
            symbol = q.encode(state)

            # Pretty-print the state vector inline
            vec_str = "  ".join(f"{v:>+7.3f}" for v in state)
            print(f"  {info['tick_index']:>5}  #{symbol:>06d}  │  [{vec_str}]")

            # Also print the centroid to show what the symbol "means"
            if step_i == 0:
                c = q.centroid(symbol)
                print(f"  {'':>5}  {'centroid':>7}  │  [{' '.join(f'{v:>+7.3f}' for v in c)}]")
                print(f"  {'':>5}  {'':>7}  │")

            state, _, done, _, info = env.step(action=0)
            if done:
                break

    # ── 4. Batch encode a full episode and show symbol histogram ──────────────
    print(f"\n\n{'='*68}")
    print("  Batch encode: full episode symbol distribution")
    print(f"{'='*68}")

    ep_tape = all_tapes[0]
    ep_env  = SRMEnvironment(ep_tape)
    ep_state, _ = ep_env.reset()

    ep_states: list[np.ndarray] = [ep_state]
    done = False
    while not done:
        ep_state, _, done, _, _ = ep_env.step(0)
        if not done:
            ep_states.append(ep_state)

    ep_arr     = np.stack(ep_states, axis=0)        # (T, 8)
    ep_symbols = q.encode_batch(ep_arr)             # (T,)

    unique, counts = np.unique(ep_symbols, return_counts=True)
    top10_idx = np.argsort(counts)[::-1][:10]

    print(f"\n  Tape : {ep_tape.name}")
    print(f"  Ticks: {len(ep_symbols):,}")
    print(f"  Unique symbols used : {len(unique)} / {q.n_clusters}")
    print(f"\n  Top-10 most frequent symbols:")
    print(f"  {'Rank':>4}  {'Symbol':>7}  {'Count':>7}  {'%':>6}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*6}")
    for rank, i in enumerate(top10_idx):
        sid   = unique[i]
        cnt   = counts[i]
        frac  = cnt / len(ep_symbols) * 100
        print(f"  {rank+1:>4}  #{sid:>06d}  {cnt:>7,}  {frac:>5.1f}%")

    print(f"\n✅ SymbolicQuantizer ready — vocab={q.n_clusters} symbols.")
    print(f"   Encode speed: ~O(K) nearest centroid lookup per tick.")
    print(f"   Load with:  q = SymbolicQuantizer.load('{SAVE_PATH}')")
