"""
srm_transition_graph.py

Builds an empirical first-order Markov transition graph over the 512-symbol
vocabulary produced by SymbolicQuantizer.

Architecture role
─────────────────
  SRMEnvironment → 8D state → SymbolicQuantizer → Symbol ID
       └─────────────────────────────────────────────────────┐
                                                             ▼
                              MarketTransitionGraph  (512 × 512 count matrix)
                                                             │
                        get_probabilities(S) → [(S', p%), …] ◄── MCTS planner

What this gives the MCTS planner
─────────────────────────────────
  At any node S in the search tree, the planner needs to know:
    "If the market is in state S right now, which states are most likely next?"
  This class answers that question in O(K) time (one row lookup + argsort).

The matrix is stored in two forms:
  _counts  : np.uint32 (512, 512)  — raw integer transition counts
             Stored as uint32 to keep RAM at 1 MB (vs. 2 MB for float64)
             and to allow incremental updates without precision loss.
  _probs   : np.float32 (512, 512) — row-normalised probability matrix,
             computed lazily on first call to get_probabilities() or
             explicitly via compute_probabilities().

Memory budget:
  512 × 512 × 4 bytes (uint32) ≈ 1.0 MB counts
  512 × 512 × 4 bytes (float32) ≈ 1.0 MB probs
  Total ≈ 2 MB — trivially fits in RAM.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import joblib

# Local
from srm_environment import SRMEnvironment
from srm_quantizer import SymbolicQuantizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 512   # Must match SymbolicQuantizer.n_clusters


# ──────────────────────────────────────────────────────────────────────────────
# MarketTransitionGraph
# ──────────────────────────────────────────────────────────────────────────────

class MarketTransitionGraph:
    """
    Empirical first-order Markov transition matrix over Solana market symbols.

    Each entry ``_counts[i, j]`` holds the number of observed times symbol ``i``
    was immediately followed by symbol ``j`` in any tape.  Dividing a row by its
    sum yields the empirical probability distribution of next states.

    Parameters
    ----------
    quantizer : SymbolicQuantizer
        A fitted quantizer used to encode raw states into symbol IDs during
        graph construction.  Must have n_clusters == VOCAB_SIZE.
    vocab_size : int, optional
        Vocabulary size.  Default: 512 (matches SymbolicQuantizer default).
    """

    def __init__(
        self,
        quantizer: SymbolicQuantizer,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        if not quantizer.is_fitted:
            raise ValueError("SymbolicQuantizer must be fitted before use.")
        if quantizer.n_clusters != vocab_size:
            raise ValueError(
                f"Quantizer vocab size ({quantizer.n_clusters}) != "
                f"expected vocab_size ({vocab_size})."
            )

        self.quantizer  = quantizer
        self.vocab_size = vocab_size

        # Core count matrix: uint32 saves half the RAM vs int64 and can hold
        # counts up to 4,294,967,295 — far more than we'll ever see.
        self._counts: np.ndarray = np.zeros(
            (vocab_size, vocab_size), dtype=np.uint32
        )

        # Lazy-computed row-normalised probability matrix (float32)
        self._probs: np.ndarray | None = None
        self._probs_dirty: bool = True   # True = probabilities need recompute

        # Metadata
        self.is_built:        bool  = False
        self.total_transitions: int = 0
        self.tapes_processed:  int  = 0

    # ── Private ───────────────────────────────────────────────────────────────

    def _process_tape(self, tape_path: Path) -> int:
        """
        Step through one tape, encode each tick, and increment the count matrix.

        Returns the number of transitions observed in this tape.
        """
        try:
            env = SRMEnvironment(tape_path)
        except Exception as exc:
            print(f"  [skip] {tape_path.name}: {exc}")
            return 0

        if env.total_ticks < 3:
            return 0   # Need at least 2 steps for one transition

        state, _ = env.reset()
        prev_sym: int = self.quantizer.encode(state)

        n_transitions = 0
        done = False

        while not done:
            state, _, done, _, _ = env.step(action=0)
            if not done:
                curr_sym: int = self.quantizer.encode(state)
                self._counts[prev_sym, curr_sym] += 1
                prev_sym = curr_sym
                n_transitions += 1

        return n_transitions

    def _process_tape_batch(self, tape_path: Path) -> int:
        """
        Vectorised variant of _process_tape — encodes the entire episode at once
        with encode_batch(), then uses np.add.at() for a single-pass count update.

        2–5× faster than the tick-by-tick loop for large tapes because it
        avoids Python-level iteration over individual states.
        """
        try:
            env = SRMEnvironment(tape_path)
        except Exception as exc:
            print(f"  [skip] {tape_path.name}: {exc}")
            return 0

        if env.total_ticks < 3:
            return 0

        # Collect all states for this tape in one pass
        state, _ = env.reset()
        episode_states: list[np.ndarray] = [state]

        done = False
        while not done:
            state, _, done, _, _ = env.step(action=0)
            if not done:
                episode_states.append(state)

        if len(episode_states) < 2:
            return 0

        # Batch encode → (T,) int array of symbol IDs
        ep_arr  = np.stack(episode_states, axis=0)   # (T, 8)
        symbols = self.quantizer.encode_batch(ep_arr) # (T,)  dtype=int

        # Pairs: (s_t, s_{t+1})
        src = symbols[:-1]   # (T-1,)
        dst = symbols[1:]    # (T-1,)

        # np.add.at is an unbuffered accumulation — safe even with repeats
        np.add.at(self._counts, (src, dst), 1)

        return len(src)

    # ── Public API ────────────────────────────────────────────────────────────

    def build_graph(
        self,
        tape_paths: Sequence[str | Path],
        use_batch:  bool = True,
    ) -> "MarketTransitionGraph":
        """
        Iterate over every tape, extract symbol-to-symbol transitions, and
        accumulate them into the count matrix.

        Parameters
        ----------
        tape_paths : sequence of str | Path
            .jsonl tape files to process.
        use_batch : bool
            If True (default), use the vectorised batch encoder — significantly
            faster for large tapes.  Set False to use the tick-by-tick loop
            (lower peak RAM, useful for debugging).

        Returns
        -------
        self
        """
        tape_paths = [Path(p) for p in tape_paths]
        # Reset before building so calling build_graph() twice is safe
        self._counts[:]    = 0
        self._probs        = None
        self._probs_dirty  = True
        self.total_transitions = 0
        self.tapes_processed   = 0

        t0 = time.perf_counter()
        n  = len(tape_paths)

        print(f"\n[Graph] Building transition matrix over {n} tape(s) …")
        print(f"        Matrix shape: ({self.vocab_size} × {self.vocab_size}), "
              f"dtype=uint32, "
              f"RAM={self._counts.nbytes / 1_048_576:.1f} MB")

        for idx, tape_path in enumerate(tape_paths):
            tape_path = Path(tape_path)

            if tape_path.stat().st_size == 0:
                continue  # skip empty files

            process_fn = self._process_tape_batch if use_batch else self._process_tape
            n_trans    = process_fn(tape_path)

            self.total_transitions += n_trans
            self.tapes_processed   += 1

            pct = (idx + 1) / n * 100
            print(
                f"  [{idx+1:>3}/{n}] {tape_path.name:<58} "
                f"+{n_trans:>7,} edges  | "
                f"total={self.total_transitions:>10,}  ({pct:.0f}%)"
            )

        self.is_built     = True
        self._probs_dirty = True   # counts changed, probs need recompute

        elapsed = time.perf_counter() - t0

        # Stats
        active_nodes = int(np.any(self._counts > 0, axis=1).sum())
        active_edges = int(np.count_nonzero(self._counts))
        density      = active_edges / (self.vocab_size ** 2) * 100

        print(
            f"\n[Graph] ✅ Done in {elapsed:.1f}s\n"
            f"         Total transitions  : {self.total_transitions:>12,}\n"
            f"         Active nodes       : {active_nodes} / {self.vocab_size}\n"
            f"         Non-zero edges     : {active_edges:,}\n"
            f"         Graph density      : {density:.2f}%\n"
            f"         Max single edge    : {int(self._counts.max()):,}"
        )
        return self

    def compute_probabilities(self) -> None:
        """
        Row-normalise the count matrix into a float32 probability matrix.
        Rows with zero total counts (unseen symbols) are left as all-zeros.
        Called automatically by get_probabilities() when needed.
        """
        row_sums = self._counts.sum(axis=1, keepdims=True).astype(np.float32)
        # Avoid divide-by-zero: set denominator to 1 where row is empty
        safe_sums = np.where(row_sums == 0, 1.0, row_sums)
        self._probs       = self._counts.astype(np.float32) / safe_sums
        self._probs_dirty = False

    def get_probabilities(
        self,
        symbol_id: int,
        top_k:     int | None = None,
        min_prob:  float      = 0.0,
    ) -> list[tuple[int, float, int]]:
        """
        Return the empirical next-state probability distribution for a given
        symbol.

        Parameters
        ----------
        symbol_id : int
            The source symbol (node) whose outgoing edges to query.
        top_k : int | None
            If set, return only the top-K most probable next states.
            Default: return all non-zero entries.
        min_prob : float
            Minimum probability threshold to include an entry (default: 0.0).

        Returns
        -------
        list of (next_symbol_id, probability, raw_count)
            Sorted descending by probability.  Each tuple contains:
              - next_symbol: int in [0, vocab_size)
              - probability: float in [0, 1]
              - raw_count:   int — how many times this transition was observed

        Raises
        ------
        ValueError  if symbol_id is out of range or the graph has not been built.
        """
        if not self.is_built:
            raise RuntimeError("Call build_graph() or load() before querying.")
        if not (0 <= symbol_id < self.vocab_size):
            raise ValueError(
                f"symbol_id {symbol_id} out of range [0, {self.vocab_size})"
            )

        # Recompute prob matrix if stale
        if self._probs_dirty or self._probs is None:
            self.compute_probabilities()

        probs_row  = self._probs[symbol_id]   # type: ignore[index]
        counts_row = self._counts[symbol_id]

        # Filter and sort
        nonzero_ids = np.where(probs_row > min_prob)[0]
        if len(nonzero_ids) == 0:
            return []

        order  = np.argsort(probs_row[nonzero_ids])[::-1]
        sorted_ids = nonzero_ids[order]

        if top_k is not None:
            sorted_ids = sorted_ids[:top_k]

        return [
            (int(sid), float(probs_row[sid]), int(counts_row[sid]))
            for sid in sorted_ids
        ]

    def get_count(self, from_sym: int, to_sym: int) -> int:
        """Raw transition count from symbol ``from_sym`` to ``to_sym``."""
        return int(self._counts[from_sym, to_sym])

    def outbound_total(self, symbol_id: int) -> int:
        """Total number of outbound transitions observed from ``symbol_id``."""
        return int(self._counts[symbol_id].sum())

    def most_active_symbols(self, top_k: int = 20) -> list[tuple[int, int]]:
        """
        Return the symbols with the highest total outbound transitions.

        Returns
        -------
        list of (symbol_id, total_outbound_count), descending by count.
        """
        row_totals = self._counts.sum(axis=1)  # (512,)
        top_idx    = np.argsort(row_totals)[::-1][:top_k]
        return [(int(i), int(row_totals[i])) for i in top_idx if row_totals[i] > 0]

    def symbol_entropy(self, symbol_id: int) -> float:
        """
        Shannon entropy of the next-state distribution for ``symbol_id``.

        High entropy → many roughly-equal next states (unpredictable / noisy).
        Low entropy  → concentrated on a few successors (predictable / stable).
        """
        if self._probs_dirty or self._probs is None:
            self.compute_probabilities()

        p = self._probs[symbol_id]          # type: ignore[index]
        p = p[p > 0]
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Persist the graph to disk (counts matrix + metadata).
        The quantizer is *not* saved here — load it separately via
        SymbolicQuantizer.load().

        Parameters
        ----------
        path : str | Path
            Output file path.  Convention: ``srm_graph_512.pkl``.
        """
        if not self.is_built:
            raise RuntimeError("Nothing to save — call build_graph() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "vocab_size":         self.vocab_size,
            "counts":             self._counts,
            "total_transitions":  self.total_transitions,
            "tapes_processed":    self.tapes_processed,
        }
        joblib.dump(payload, path, compress=3)
        size_kb = path.stat().st_size / 1024
        print(f"[Graph] 💾 Saved to {path}  ({size_kb:.1f} KB)")

    @classmethod
    def load(
        cls,
        path:      str | Path,
        quantizer: SymbolicQuantizer,
    ) -> "MarketTransitionGraph":
        """
        Load a pre-built graph from disk.

        Parameters
        ----------
        path      : str | Path — path to the .pkl file from save()
        quantizer : SymbolicQuantizer — the *same* fitted quantizer used during
                    build_graph() (needed for future encode calls).

        Returns
        -------
        MarketTransitionGraph  (is_built=True, probabilities can be queried)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        payload = joblib.load(path)

        instance = cls(quantizer, vocab_size=payload["vocab_size"])
        instance._counts             = payload["counts"]
        instance.total_transitions   = payload["total_transitions"]
        instance.tapes_processed     = payload["tapes_processed"]
        instance.is_built            = True
        instance._probs_dirty        = True   # always recompute on load

        active_nodes = int(np.any(instance._counts > 0, axis=1).sum())
        print(
            f"[Graph] 📂 Loaded {path.name}  │  "
            f"vocab={instance.vocab_size}  │  "
            f"transitions={instance.total_transitions:,}  │  "
            f"active nodes={active_nodes}"
        )
        return instance

    def __repr__(self) -> str:
        if self.is_built:
            active = int(np.any(self._counts > 0, axis=1).sum())
            return (
                f"MarketTransitionGraph("
                f"vocab={self.vocab_size}, "
                f"transitions={self.total_transitions:,}, "
                f"active_nodes={active}, "
                f"tapes={self.tapes_processed})"
            )
        return f"MarketTransitionGraph(vocab={self.vocab_size}, not yet built)"


# ──────────────────────────────────────────────────────────────────────────────
# Execution block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    TAPE_DIR    = Path(__file__).parent / "solana-tapes"
    QUANT_PATH  = Path(__file__).parent / "srm_quantizer_512.pkl"
    GRAPH_PATH  = Path(__file__).parent / "srm_graph_512.pkl"

    # ── 1. Load the pre-trained quantizer ─────────────────────────────────────
    print("=" * 70)
    print("  Step 1 — Load SymbolicQuantizer")
    print("=" * 70)
    q = SymbolicQuantizer.load(QUANT_PATH)
    print(q)

    # ── 2. Gather all usable tapes ────────────────────────────────────────────
    all_tapes = sorted(
        [p for p in TAPE_DIR.glob("*.jsonl") if p.stat().st_size > 0],
        key=lambda p: p.name,
    )
    print(f"\nFound {len(all_tapes)} non-empty tapes in {TAPE_DIR.name}/")

    # ── 3. Build (or reload) the transition graph ─────────────────────────────
    print("\n" + "=" * 70)
    print("  Step 2 — Build / Load MarketTransitionGraph")
    print("=" * 70)

    if GRAPH_PATH.exists():
        print(f"[Main] Pre-built graph found at {GRAPH_PATH}.")
        print("[Main] Delete it to force a rebuild.  Loading …\n")
        graph = MarketTransitionGraph.load(GRAPH_PATH, q)
    else:
        graph = MarketTransitionGraph(q)
        graph.build_graph(all_tapes)
        graph.save(GRAPH_PATH)

    print(f"\n{graph}")

    # ── 4. Graph statistics ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Step 3 — Graph Statistics")
    print("=" * 70)

    active_nodes = int(np.any(graph._counts > 0, axis=1).sum())
    active_edges = int(np.count_nonzero(graph._counts))
    density      = active_edges / (graph.vocab_size ** 2) * 100

    print(f"  Vocabulary size    : {graph.vocab_size}")
    print(f"  Active nodes       : {active_nodes} / {graph.vocab_size}  "
          f"({active_nodes/graph.vocab_size*100:.1f}%)")
    print(f"  Non-zero edges     : {active_edges:,}")
    print(f"  Graph density      : {density:.3f}%")
    print(f"  Total transitions  : {graph.total_transitions:,}")
    print(f"  Avg out-degree     : {active_edges/max(active_nodes,1):.1f} edges/node")

    # ── 5. Most active symbols ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Step 4 — Most Active Symbols (highest outbound traffic)")
    print("=" * 70)
    print(f"\n  {'Rank':>4}  {'Symbol':>8}  {'OutEdges':>10}  {'Entropy':>8}  "
          f"{'% of all trans':>15}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*15}")

    top_active = graph.most_active_symbols(top_k=15)
    for rank, (sym, cnt) in enumerate(top_active[:15]):
        pct     = cnt / graph.total_transitions * 100
        entropy = graph.symbol_entropy(sym)
        print(f"  {rank+1:>4}  #{sym:>06d}  {cnt:>10,}  {entropy:>8.3f}  {pct:>14.2f}%")

    # ── 6. Query top symbols for next-state probabilities ────────────────────
    # Use the most active symbol + a few specific ones discovered earlier
    probe_symbols: list[int] = [top_active[0][0], top_active[1][0], 199, 317]
    probe_symbols = list(dict.fromkeys(probe_symbols))  # dedup, preserve order

    print("\n" + "=" * 70)
    print("  Step 5 — Next-State Probability Distributions (top 5 each)")
    print("=" * 70)

    for probe in probe_symbols:
        out_total = graph.outbound_total(probe)
        entropy   = graph.symbol_entropy(probe)
        print(f"\n  ┌─ Symbol #{probe:06d}  (outbound={out_total:,}  entropy={entropy:.3f} bits)")

        if out_total == 0:
            print(f"  │  [No outbound transitions observed]")
            print(f"  └{'─'*60}")
            continue

        top5 = graph.get_probabilities(probe, top_k=5)
        print(f"  │   {'Rank':>4}  {'Next Symbol':>12}  {'Probability':>12}  "
              f"{'Count':>10}  {'Bar'}")
        print(f"  │   {'─'*4}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*20}")

        for rank, (nxt, prob, cnt) in enumerate(top5):
            bar_len = int(prob * 40)
            bar     = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  │   {rank+1:>4}  #{nxt:>10d}  {prob:>11.2%}  {cnt:>10,}  {bar}")

            # Show the centroid of the next symbol so we can interpret it
            c        = q.centroid(nxt)
            labels   = ["ΔPrc%", "ΔμPrc%", "ΔAvgD%", "ΔBid%", "ΔAsk%", "OBI", "ΔOBI", "ΔBin"]
            c_str    = "  ".join(f"{l}={v:>+6.3f}" for l, v in zip(labels, c))
            print(f"  │              centroid → [{c_str}]")

        # Also show how often the symbol loops back to itself (self-transitions)
        self_cnt  = graph.get_count(probe, probe)
        self_prob = self_cnt / out_total if out_total > 0 else 0
        print(f"  │")
        print(f"  │   Self-loop:  #{probe:06d} → #{probe:06d}  "
              f"p={self_prob:.2%}  count={self_cnt:,}")
        print(f"  └{'─'*60}")

    # ── 7. Markov chain simulation ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Step 6 — Markov Chain Walk (greedy, 20 steps from most-active node)")
    print("=" * 70)

    start_sym = top_active[0][0]
    rng       = np.random.default_rng(seed=42)
    current   = start_sym

    print(f"\n  Starting from #{start_sym:06d} — following highest-probability edges:\n")
    print(f"  {'Step':>5}  {'From':>8}  {'→':>2}  {'To':>8}  {'P':>8}  {'Entropy':>8}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*2}  {'─'*8}  {'─'*8}  {'─'*8}")

    for step in range(20):
        nexts = graph.get_probabilities(current, top_k=10)
        if not nexts:
            print(f"  {step+1:>5}  #{current:>06d}  →  [dead end]")
            break

        # Probabilistic walk (sample from the distribution, not just greedy)
        next_syms  = [n[0] for n in nexts]
        next_probs = np.array([n[1] for n in nexts])
        next_probs /= next_probs.sum()   # re-normalise top-10 slice
        chosen      = int(rng.choice(next_syms, p=next_probs))
        chosen_prob = dict((n[0], n[1]) for n in nexts)[chosen]
        entropy     = graph.symbol_entropy(chosen)

        print(f"  {step+1:>5}  #{current:>06d}  →  #{chosen:>06d}  "
              f"{chosen_prob:>7.2%}  {entropy:>8.3f}")
        current = chosen

    print(f"\n✅ MarketTransitionGraph built and validated.")
    print(f"   Load with: graph = MarketTransitionGraph.load('{GRAPH_PATH}', q)")
