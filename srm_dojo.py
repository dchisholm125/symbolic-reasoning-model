"""
srm_dojo.py

High-speed offline training loop for the Intrinsic Curiosity Module (ICM).
Extracts market physics from all historical .jsonl tapes to pre-train our
forward prediction network.

Pure CPU execution (NumPy + PyTorch CPU backend). No LLM overhead.
"""

import sys
import time
from pathlib import Path
from typing import Any

from srm_environment import SRMEnvironment
from srm_quantizer import SymbolicQuantizer
from srm_transition_graph import MarketTransitionGraph
from srm_mcts import MCTSPlanner
from srm_icm import IntrinsicCuriosityModule

# ──────────────────────────────────────────────────────────────────────────────
# Component: ShadowEvaluator
# ──────────────────────────────────────────────────────────────────────────────

class ShadowEvaluator:
    def __init__(self, eval_horizon: int = 5):
        self.eval_horizon = eval_horizon
        self.pending_evals: dict[int, dict[str, Any]] = {}
        self.total_wins: int = 0
        self.total_losses: int = 0

    def record_prediction(self, tick_index: int, decision: int, current_price: float) -> None:
        self.pending_evals[tick_index] = {
            "decision": decision,
            "price": current_price,
        }

    def evaluate_tick(self, current_tick: int, current_price: float) -> None:
        target_tick = current_tick - self.eval_horizon
        if target_tick not in self.pending_evals:
            return

        record = self.pending_evals.pop(target_tick)
        decision = record["decision"]
        old_price = record["price"]
        
        price_delta = current_price - old_price
        
        # Grading logic
        if decision == 1:   # BUY
            if price_delta > 0:
                self.total_wins += 1
            else:
                self.total_losses += 1
        elif decision == 2: # SELL
            if price_delta < 0:
                self.total_wins += 1
            else:
                self.total_losses += 1

    def accuracy(self) -> float:
        total = self.total_wins + self.total_losses
        if total == 0:
            return 0.0
        return (self.total_wins / total) * 100.0

# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  SRM Dojo - Offline Training Engine")
    print("=" * 64)

    ROOT = Path(__file__).parent
    TAPES_DIR = ROOT / "solana-tapes"
    
    tapes = sorted(TAPES_DIR.glob("*.jsonl"))
    if not tapes:
        print(f"No tapes found in {TAPES_DIR}")
        sys.exit(1)
        
    print(f"Found {len(tapes)} tapes.")

    quantizer = SymbolicQuantizer.load(ROOT / "srm_quantizer_512.pkl")
    graph = MarketTransitionGraph.load(ROOT / "srm_graph_512.pkl", quantizer)
    
    icm_path = ROOT / "srm_icm.pt"
    icm = IntrinsicCuriosityModule()
    if icm_path.exists():
        icm.load(str(icm_path))
    else:
        print("[Dojo] No existing ICM checkpoint found. Starting fresh.")
        
    planner = MCTSPlanner(quantizer, graph, base_simulations=150, random_state=42)
    evaluator = ShadowEvaluator(eval_horizon=5)
    
    UPDATE_FREQ = 256
    LOG_FREQ = 1000
    SAVE_FREQ = 50000
    
    global_ticks = 0
    t0_run = time.perf_counter()
    
    for tape_idx, tape_path in enumerate(tapes):
        print(f"\n[Dojo] Loading Tape {tape_idx+1}/{len(tapes)}: {tape_path.name}")
        try:
            env = SRMEnvironment(tape_path)
            state, info = env.reset()
        except Exception as e:
            print(f"Failed to load tape {tape_path.name}: {e}")
            continue
            
        ticks_this_tape = 0
        
        while True:
            # 1. State -> Symbol
            symbol = int(quantizer.encode(state))
            
            # 2. MCTS Planner Action
            action, _ = planner.search(symbol)
            
            # 3. Step environment -> S_{t+1}
            next_state, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
                
            current_price = info.get("activePrice", 0.0)
            
            # 4. Push to ICM
            icm.push(state, action, next_state)
            
            # 5. Record and grade
            evaluator.record_prediction(global_ticks, action, current_price)
            evaluator.evaluate_tick(global_ticks, current_price)
            
            global_ticks += 1
            ticks_this_tape += 1
            
            # 6. Update ICM
            if global_ticks % UPDATE_FREQ == 0:
                icm.update()
                
            # 7. Logging
            if global_ticks % LOG_FREQ == 0:
                loss_str = f"{icm._loss_ema:.6f}" if icm._loss_ema > 0 else "N/A"
                acc = evaluator.accuracy()
                print(f"[Dojo] Ticks: {global_ticks:>7}  |  "
                      f"ICM Loss (EMA): {loss_str:>9}  |  "
                      f"Win Rate: {acc:>5.1f}%  ({evaluator.total_wins}W/{evaluator.total_losses}L)")
                      
            # 8. Save Weights
            if global_ticks % SAVE_FREQ == 0:
                icm.save(str(icm_path))
                
            state = next_state
            
        # End of tape -> save weights
        icm.save(str(icm_path))
        
    total_time = time.perf_counter() - t0_run
    if total_time > 0:
        speed = global_ticks / total_time
    else:
        speed = 0.0
        
    print("\n" + "=" * 64)
    print("  Dojo Training Complete")
    print("=" * 64)
    print(f"Total Ticks: {global_ticks}")
    print(f"Total Time:  {total_time:.1f}s")
    print(f"Speed:       {speed:.1f} ticks/sec")
    print(f"Final Loss:  {icm._loss_ema:.6f}")
    print(f"Final Acc:   {evaluator.accuracy():.1f}%")

if __name__ == "__main__":
    main()
