"""
srm_mcts.py

Monte Carlo Tree Search (MCTS) planner for the Symbolic Reasoning Model (SRM).

Architecture role
─────────────────
  MarketTransitionGraph ──→ P(S_{t+1} | S_t)
                                    │
                                    ▼
                            MCTSPlanner.search(root_symbol)
                                    │
                          ┌─────────┴──────────┐
                          │  4-phase MCTS loop  │
                          └─────────┬──────────┘
                                    │
                                    ▼
                    best_action ∈ {HOLD=0, BUY=1, SELL=2}
                    + full annotated search tree for introspection

Tree semantics
──────────────
  The tree alternates between two node types:

    DECISION node  — "We are at market symbol S; which action should we take?"
                     Children are one ACTION node per action ∈ {0,1,2}.
                     UCB1 is evaluated here to select the best action child.

    ACTION node    — "We chose action A; what market state comes next?"
                     Children are DECISION nodes for each sampled next symbol,
                     weighted by P(S' | S) from MarketTransitionGraph.
                     Next symbols are sampled stochastically during expansion.

This two-level structure cleanly separates "what we choose" (actions) from
"what the market does" (stochastic transitions).

Simulation (rollout)
────────────────────
  During the simulation phase we roll out K steps (default K=5) using the
  transition graph to sample next symbols, accumulating reward at each step.
  Reward is currently a PLACEHOLDER that combines:
    • A small signal derived from symbol entropy (high entropy → risky → -r)
    • A random component (±1.0)
  This will be replaced by the Intrinsic Curiosity Module (ICM) heuristic.

Dynamic simulation budget
──────────────────────────
  High-entropy symbols (many near-equally-likely futures) warrant deeper search.
  Low-entropy symbols (essentially deterministic next state) need few sims.
  Budget:  n_sims = base × clamp(entropy / log2(vocab), sim_min, sim_max)
  where all knobs are configurable at construction time.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Local
from srm_quantizer import SymbolicQuantizer
from srm_transition_graph import MarketTransitionGraph

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ACTIONS     = {0: "HOLD", 1: "BUY", 2: "SELL"}
N_ACTIONS   = len(ACTIONS)
UCB_C       = 1.41          # Exploration constant (√2 ≈ 1.414)
ROLLOUT_K   = 5             # Steps per simulation rollout
LOG2_VOCAB  = math.log2(512)  # Normaliser for entropy-based sim budget


# ──────────────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    """
    A single node in the MCTS search tree.

    Parameters
    ----------
    symbol      : int   — Market symbol (Quantizer ID) at this state.
    action      : int | None — Action taken to reach this node (None = root).
    parent      : Node | None — Parent node in the tree.

    Attributes
    ----------
    N  : int   — Visit count.
    W  : float — Cumulative reward (sum of all backpropagated rewards).
    Q  : float — Mean reward  W / N  (0.0 if never visited).
    children : dict  — Maps action_id (int) → list[Node] for DECISION nodes,
                       or symbol_id (int) → Node for ACTION nodes.
    is_action_node : bool — True if this is an ACTION node (chosen-action level),
                            False for DECISION nodes (market-state level).
    """
    symbol:         int
    action:         Optional[int]  = None
    parent:         Optional["Node"] = field(default=None, repr=False)
    is_action_node: bool           = False

    # MCTS statistics
    N:  int   = 0
    W:  float = 0.0

    # Children:
    #   DECISION node → {action_id: Node}             (one child per action)
    #   ACTION node   → {next_symbol: Node}           (one child per next state)
    children: dict = field(default_factory=dict)

    @property
    def Q(self) -> float:
        """Mean action value (expected reward)."""
        return self.W / self.N if self.N > 0 else 0.0

    def ucb1(self, parent_N: int, c: float = UCB_C) -> float:
        """
        UCB1 score for tree selection.
        Unvisited nodes return +∞ to guarantee they get expanded first.
        """
        if self.N == 0:
            return float("inf")
        return self.Q + c * math.sqrt(math.log(parent_N) / self.N)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        act_str = ACTIONS.get(self.action, "root") if self.action is not None else "root"
        kind    = "ACT" if self.is_action_node else "DEC"
        return (
            f"Node[{kind} sym=#{self.symbol:04d} "
            f"act={act_str} N={self.N} Q={self.Q:+.3f}]"
        )


# ──────────────────────────────────────────────────────────────────────────────
# MCTSPlanner
# ──────────────────────────────────────────────────────────────────────────────

class MCTSPlanner:
    """
    Monte Carlo Tree Search planner over the 512-symbol Solana market vocabulary.

    Parameters
    ----------
    quantizer : SymbolicQuantizer
        Fitted quantizer — used for centroid lookups and metadata.
    graph : MarketTransitionGraph
        Fitted transition graph — provides P(S' | S) for expansion + rollout.
    base_simulations : int
        Number of MCTS simulations for an average-entropy symbol.  Default: 200.
    sim_scale_min : float
        Minimum simulation multiplier (applied for very low-entropy symbols).
    sim_scale_max : float
        Maximum simulation multiplier (applied for very high-entropy symbols).
    rollout_depth : int
        Number of market steps to simulate during the rollout phase.
    ucb_c : float
        UCB1 exploration constant.  Default: 1.41 (≈ √2).
    top_k_expansion : int
        Number of most-probable next symbols to expand per action node.
        Keeps the branching factor tractable (don't expand all 512 symbols).
    random_state : int
        Seed for the NumPy RNG used in stochastic sampling.
    """

    def __init__(
        self,
        quantizer:         SymbolicQuantizer,
        graph:             MarketTransitionGraph,
        base_simulations:  int   = 200,
        sim_scale_min:     float = 0.25,
        sim_scale_max:     float = 4.0,
        rollout_depth:     int   = ROLLOUT_K,
        ucb_c:             float = UCB_C,
        top_k_expansion:   int   = 8,
        random_state:      int   = 42,
    ) -> None:
        self.quantizer        = quantizer
        self.graph            = graph
        self.base_simulations = base_simulations
        self.sim_scale_min    = sim_scale_min
        self.sim_scale_max    = sim_scale_max
        self.rollout_depth    = rollout_depth
        self.ucb_c            = ucb_c
        self.top_k_expansion  = top_k_expansion
        self._rng             = np.random.default_rng(random_state)

        # Per-call diagnostics (populated by search())
        self._last_root:         Optional[Node] = None
        self._last_n_sims:       int            = 0
        self._last_elapsed_ms:   float          = 0.0

    # ── Dynamic simulation budget ─────────────────────────────────────────────

    def _compute_sim_budget(self, symbol: int) -> int:
        """
        Scale the number of simulations by the entropy of the root symbol.

        Rationale:
          • entropy ≈ 0  → next state is nearly deterministic → few sims needed
          • entropy ≈ log2(512) ≈ 9 → next state is uniform-random → many sims
        """
        entropy  = self.graph.symbol_entropy(symbol)
        ratio    = entropy / LOG2_VOCAB                             # ∈ [0, 1]
        scale    = self.sim_scale_min + ratio * (self.sim_scale_max - self.sim_scale_min)
        n_sims   = max(1, int(self.base_simulations * scale))
        return n_sims

    # ── Phase 1: Selection ────────────────────────────────────────────────────

    def _select(self, node: Node) -> tuple[Node, list[Node]]:
        """
        Walk down the tree following UCB1 until we reach a leaf.

        Returns the leaf node and the path from root → leaf (inclusive).
        Rule:
          - At a DECISION node: select the action child with highest UCB1.
          - At an ACTION node:  sample a next-symbol child proportionally to
            the empirical transition counts (stochastic env transition).
        """
        path = [node]

        while not node.is_leaf():
            if not node.is_action_node:
                # DECISION node → pick action by UCB1
                best_child = max(
                    node.children.values(),
                    key=lambda c: c.ucb1(node.N, self.ucb_c),
                )
                node = best_child
            else:
                # ACTION node → stochastic transition sampling
                children   = list(node.children.values())
                counts     = np.array(
                    [self.graph.get_count(node.symbol, c.symbol) for c in children],
                    dtype=np.float64,
                )
                total = counts.sum()
                probs = counts / total if total > 0 else np.ones(len(children)) / len(children)
                node  = children[self._rng.choice(len(children), p=probs)]

            path.append(node)

        return node, path

    # ── Phase 2: Expansion ────────────────────────────────────────────────────

    def _expand(self, node: Node) -> None:
        """
        Expand a leaf node by adding its children.

        DECISION node → add one ACTION child per action ∈ {0, 1, 2}.
        ACTION node   → add DECISION children for the top-K most probable
                        next market symbols from the transition graph.
        """
        if not node.is_action_node:
            # Expand: one action child per available action
            for action_id in range(N_ACTIONS):
                child = Node(
                    symbol=node.symbol,
                    action=action_id,
                    parent=node,
                    is_action_node=True,
                )
                node.children[action_id] = child
        else:
            # Expand: next market states from transition graph
            transitions = self.graph.get_probabilities(
                node.symbol, top_k=self.top_k_expansion
            )
            if not transitions:
                # Dead node (symbol never seen transitioning) → add self-loop
                child = Node(
                    symbol=node.symbol,
                    action=node.action,
                    parent=node,
                    is_action_node=False,
                )
                node.children[node.symbol] = child
            else:
                for next_sym, _prob, _cnt in transitions:
                    child = Node(
                        symbol=next_sym,
                        action=node.action,
                        parent=node,
                        is_action_node=False,
                    )
                    node.children[next_sym] = child

    # ── Phase 3: Simulation (rollout) ─────────────────────────────────────────

    def _simulate(self, node: Node) -> float:
        """
        Roll out a random market trajectory from ``node`` for ``rollout_depth``
        steps, accumulating a heuristic reward signal at each step.

        Current reward heuristic (PLACEHOLDER — will be replaced by ICM):
        ─────────────────────────────────────────────────────────────────
          r_step = random_noise(±1.0) − entropy_penalty
          where entropy_penalty = symbol_entropy / LOG2_VOCAB × 0.5

        Logic:
          • High-entropy symbols are penalised slightly (uncertain = risky).
          • Random noise ensures the tree doesn't collapse to a single path
            before the real reward function is wired in.
          • Action-adjusted bonus: BUY/SELL get a tiny exploration bonus
            (+0.05) vs HOLD (0.0) to prevent all branches being identical.

        Returns
        -------
        float : cumulative discounted reward (γ=0.9 per step).
        """
        discount      = 0.9
        cumulative_r  = 0.0
        gamma_k       = 1.0
        current_sym   = node.symbol
        action        = node.action  # None for DECISION nodes (root or mid-tree)

        # Action-specific base bonus (tiny exploration signal)
        action_bonus = {0: 0.00, 1: 0.05, 2: 0.05}.get(action or 0, 0.0)

        for _ in range(self.rollout_depth):
            entropy         = self.graph.symbol_entropy(current_sym)
            entropy_penalty = (entropy / LOG2_VOCAB) * 0.5

            # Sample next symbol using the transition graph
            next_transitions = self.graph.get_probabilities(
                current_sym, top_k=self.top_k_expansion
            )
            if next_transitions:
                next_syms  = [t[0] for t in next_transitions]
                next_probs = np.array([t[1] for t in next_transitions], dtype=np.float64)
                next_probs /= next_probs.sum()
                current_sym = int(self._rng.choice(next_syms, p=next_probs))
            # else: dead end — stay at current_sym

            # Placeholder reward
            noise  = float(self._rng.uniform(-1.0, 1.0))
            r_step = noise - entropy_penalty + action_bonus
            cumulative_r += gamma_k * r_step
            gamma_k      *= discount

        return cumulative_r

    # ── Phase 4: Backpropagation ──────────────────────────────────────────────

    @staticmethod
    def _backpropagate(path: list[Node], reward: float) -> None:
        """
        Walk the path from leaf → root and update N and W for every node.
        """
        for node in reversed(path):
            node.N += 1
            node.W += reward

    # ── Main search loop ──────────────────────────────────────────────────────

    def search(
        self,
        root_symbol:     int,
        num_simulations: Optional[int] = None,
    ) -> tuple[int, Node]:
        """
        Run MCTS from ``root_symbol`` and return the best action.

        Parameters
        ----------
        root_symbol : int
            The current market symbol produced by SymbolicQuantizer.encode().
        num_simulations : int | None
            Override the simulation budget.  If None, computes dynamically
            from the root symbol's entropy.

        Returns
        -------
        best_action : int — 0=HOLD, 1=BUY, 2=SELL
        root_node   : Node — Root of the fully-searched tree (for introspection).
        """
        if not self.graph.is_built:
            raise RuntimeError("MarketTransitionGraph must be built before searching.")

        # Dynamic compute budget
        if num_simulations is None:
            num_simulations = self._compute_sim_budget(root_symbol)

        root = Node(symbol=root_symbol, action=None, is_action_node=False)

        t0 = time.perf_counter()

        for _ in range(num_simulations):
            # ── 1. Selection ──────────────────────────────────────────────────
            leaf, path = self._select(root)

            # ── 2. Expansion ──────────────────────────────────────────────────
            if leaf.N > 0 or leaf is root:
                # Only expand if already visited (or is root)
                self._expand(leaf)
                if leaf.children:
                    # Step into the first unexplored child for simulation
                    leaf = next(iter(leaf.children.values()))
                    path.append(leaf)

            # ── 3. Simulation ─────────────────────────────────────────────────
            reward = self._simulate(leaf)

            # ── 4. Backpropagation ────────────────────────────────────────────
            self._backpropagate(path, reward)

        # Choose best action: the action child of root with highest Q
        if not root.children:
            best_action = 0  # fallback: HOLD
        else:
            best_action = max(
                root.children.keys(),
                key=lambda a: root.children[a].Q,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Persist diagnostics
        self._last_root       = root
        self._last_n_sims     = num_simulations
        self._last_elapsed_ms = elapsed_ms

        return best_action, root

    # ── Tree introspection ────────────────────────────────────────────────────

    def print_tree(
        self,
        node:       Optional[Node] = None,
        depth:      int  = 0,
        max_depth:  int  = 3,
        top_k:      int  = 3,
        _prefix:    str  = "",
        _is_last:   bool = True,
    ) -> None:
        """
        Pretty-print the search tree up to ``max_depth`` levels.
        At each decision node shows the top-``top_k`` action children by Q.
        At each action node shows the top-``top_k`` next-state children by visit.
        """
        if node is None:
            node = self._last_root
        if node is None:
            print("[No tree — run search() first]")
            return

        connector = "└── " if _is_last else "├── "

        if depth == 0:
            kind = "ROOT"
            label = (
                f"Symbol #{node.symbol:04d}  "
                f"N={node.N}  Q={node.Q:+.3f}  "
                f"entropy={self.graph.symbol_entropy(node.symbol):.3f}"
            )
        elif node.is_action_node:
            act   = ACTIONS.get(node.action, "?")
            label = (
                f"ACTION[{act}]  "
                f"N={node.N}  Q={node.Q:+.3f}  "
                f"UCB1={node.ucb1(node.parent.N if node.parent else 1, self.ucb_c):.3f}"
            )
            kind = "ACT"
        else:
            label = (
                f"Symbol #{node.symbol:04d}  "
                f"N={node.N}  Q={node.Q:+.3f}"
            )
            kind = "DEC"

        if depth == 0:
            print(f"[{kind}] {label}")
        else:
            print(f"{_prefix}{connector}[{kind}] {label}")

        if depth >= max_depth or not node.children:
            return

        # Select top-k children to display
        if not node.is_action_node:
            # DECISION → order action children by Q descending
            sorted_children = sorted(
                node.children.values(), key=lambda c: c.Q, reverse=True
            )[:top_k]
        else:
            # ACTION → order next-state children by visit count descending
            sorted_children = sorted(
                node.children.values(), key=lambda c: c.N, reverse=True
            )[:top_k]

        extension = "    " if _is_last else "│   "
        for i, child in enumerate(sorted_children):
            is_last_child = (i == len(sorted_children) - 1)
            self.print_tree(
                node=child,
                depth=depth + 1,
                max_depth=max_depth,
                top_k=top_k,
                _prefix=_prefix + extension,
                _is_last=is_last_child,
            )

    def summary(self, root: Optional[Node] = None) -> None:
        """Print a concise one-line summary of the last search result."""
        node = root or self._last_root
        if node is None:
            return

        entropy  = self.graph.symbol_entropy(node.symbol)
        budget   = self._last_n_sims
        elapsed  = self._last_elapsed_ms

        if node.children:
            ranked = sorted(
                node.children.items(), key=lambda kv: kv[1].Q, reverse=True
            )
            best_a, best_node = ranked[0]
            print(f"\n{'═'*60}")
            print(f"  MCTS Search Summary")
            print(f"{'═'*60}")
            print(f"  Root symbol : #{node.symbol:04d}")
            print(f"  Entropy     : {entropy:.3f} bits  "
                  f"(budget scale: {entropy/LOG2_VOCAB:.2f}×)")
            print(f"  Simulations : {budget}")
            print(f"  Wall time   : {elapsed:.1f} ms  "
                  f"({elapsed/budget:.2f} ms/sim)")
            print(f"  Root N      : {node.N}")
            print()
            print(f"  Action rankings (by Q = mean reward):")
            for action_id, action_node in ranked:
                marker = " ◄ BEST" if action_id == best_a else ""
                print(
                    f"    [{ACTIONS[action_id]:<4}]  "
                    f"Q={action_node.Q:>+7.4f}  "
                    f"N={action_node.N:>5}  "
                    f"W={action_node.W:>+8.3f}{marker}"
                )
            print(f"\n  ⚡ Decision: {ACTIONS[best_a]}")
            print(f"{'═'*60}")

    def __repr__(self) -> str:
        return (
            f"MCTSPlanner("
            f"base_sims={self.base_simulations}, "
            f"rollout_depth={self.rollout_depth}, "
            f"ucb_c={self.ucb_c}, "
            f"top_k_expand={self.top_k_expansion})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Execution block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    from pathlib import Path

    QUANT_PATH = Path(__file__).parent / "srm_quantizer_512.pkl"
    GRAPH_PATH = Path(__file__).parent / "srm_graph_512.pkl"

    # ── 1. Load pre-trained components ────────────────────────────────────────
    print("=" * 60)
    print("  MCTS Planner — Initialisation")
    print("=" * 60)

    q     = SymbolicQuantizer.load(QUANT_PATH)
    graph = MarketTransitionGraph.load(GRAPH_PATH, q)

    planner = MCTSPlanner(
        quantizer        = q,
        graph            = graph,
        base_simulations = 200,
        sim_scale_min    = 0.25,  # 50  sims for entropy≈0 symbols
        sim_scale_max    = 4.0,   # 800 sims for entropy≈max symbols
        rollout_depth    = 5,
        ucb_c            = 1.41,
        top_k_expansion  = 8,
        random_state     = 42,
    )
    print(f"\n{planner}\n")

    # ── 2. Define probe symbols ────────────────────────────────────────────────
    # #000199 = high-entropy "tick event" node  (entropy ≈ 7.6)
    # #000243 = low-entropy  "stasis"    node   (entropy ≈ 0.3)
    # #000317 = medium-entropy oscillation node (entropy ≈ 5.9)
    probes = [
        (199, "High-entropy  'tick event' node"),
        (317, "Medium-entropy oscillation node"),
        (243, "Low-entropy   'stasis' node"),
    ]

    for symbol, description in probes:
        entropy = graph.symbol_entropy(symbol)
        budget  = planner._compute_sim_budget(symbol)

        print("\n" + "═" * 60)
        print(f"  Probe: Symbol #{symbol:04d}  — {description}")
        print(f"         entropy={entropy:.3f} bits  →  budget={budget} sims")
        print("═" * 60)

        # ── 3. Run MCTS ────────────────────────────────────────────────────────
        best_action, root = planner.search(symbol)

        # ── 4. Summary ─────────────────────────────────────────────────────────
        planner.summary(root)

        # ── 5. Print annotated search tree ────────────────────────────────────
        print(f"\n  Search Tree (depth=3, top-3 children each):\n")
        planner.print_tree(root, max_depth=3, top_k=3)

    # ── 6. Live walkthrough: step through a tape, search at each tick ─────────
    print("\n\n" + "═" * 60)
    print("  Live Walkthrough — 15 ticks from a volatile tape")
    print("═" * 60)

    from srm_environment import SRMEnvironment
    tape_path = Path(__file__).parent / "solana-tapes" / \
                "market_tape_volatile-bear-dump-stable_2026-02-23_16-54-01.jsonl"

    env   = SRMEnvironment(tape_path)
    state, info = env.reset()

    print(f"\n  {'Tick':>5}  {'Symbol':>8}  {'H':>7}  {'Entropy':>7}  "
          f"{'Budget':>7}  {'Decision':>8}  {'ms':>7}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*7}  "
          f"{'─'*7}  {'─'*8}  {'─'*7}")

    for tick_i in range(15):
        sym     = q.encode(state)
        entropy = graph.symbol_entropy(sym)
        budget  = planner._compute_sim_budget(sym)

        action, root = planner.search(sym)

        q_vals   = {a: root.children[a].Q for a in root.children}
        hold_q   = q_vals.get(0, float("nan"))

        print(
            f"  {info['tick_index']:>5}  "
            f"#{sym:>06d}  "
            f"{hold_q:>+7.3f}  "
            f"{entropy:>7.3f}  "
            f"{budget:>7}  "
            f"{ACTIONS[action]:>8}  "
            f"{planner._last_elapsed_ms:>7.1f}"
        )

        state, _, done, _, info = env.step(action)
        if done:
            break

    print(f"\n✅  MCTSPlanner operational.")
    print(f"    Key pattern: high-entropy ticks draw larger search budgets,")
    print(f"    low-entropy ticks are resolved with minimal compute.")
