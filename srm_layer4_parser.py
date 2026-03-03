import time
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import Dict, Any

class SRM_Input_Parser:
    """
    Layer 4: The 'Ear'. 
    Uses a tiny 86MB ONNX embedding model running purely on the CPU to map 
    messy human language into rigid Phase 1 JSON Symbols in under 10ms.
    """
    def __init__(self, model_path: str = "./layer4_onnx_model"):
        # Load the Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load the ONNX Model on CPU (Leaves VRAM 100% free for Layer 3)
        self.session = ort.InferenceSession(f"{model_path}/model.onnx", providers=['CPUExecutionProvider'])
        
        # Define the "Anchors" for our semantic space
        self.intent_anchors = {
            "query_state": "Tell me the status of our current trading operation and why we are holding or buying.",
            "execute_action": "Execute a market buy order immediately and override safety protocols.",
            "declare_fact": "The market is crashing, update your risk parameters."
        }
        
        # Pre-compute the embeddings for our anchors so we don't recalculate them on the fly
        self.anchor_embeddings = {
            intent: self._get_embedding(text) 
            for intent, text in self.intent_anchors.items()
        }
        print("Layer 4 ONNX Ear Model Loaded Successfully.")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Helper to tokenize and run ONNX inference."""
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
        
        # ONNX Inference
        outputs = self.session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
        })
        
        # Mean Pooling to get a single vector representing the sentence
        token_embeddings = outputs[0]
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = np.repeat(attention_mask[:, :, np.newaxis], token_embeddings.shape[-1], axis=-1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        embedding = sum_embeddings / sum_mask
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / norm

    def parse_natural_language(self, user_text: str) -> Dict[str, Any]:
        """Maps messy English to the closest Phase 1 JSON Symbol."""
        start_time = time.perf_counter()
        
        # Embed the user's messy input
        user_embedding = self._get_embedding(user_text)
        
        # Find the highest cosine similarity
        best_intent = "invalid_input"
        best_score = -1.0
        
        for intent, anchor_emb in self.anchor_embeddings.items():
            # Dot product of normalized vectors = Cosine Similarity
            score = float(np.dot(user_embedding, anchor_emb.T)[0][0])
            if score > best_score:
                best_score = score
                best_intent = intent
                
        # Confidence Threshold Check
        if best_score < 0.40:
            best_intent = "invalid_input"
            
        # Construct the rigid Phase 1 Input Symbol
        input_symbol = {
            "intent_class": best_intent,
            "parameters": {}
        }
        
        # Simple entity router (In a larger build, we'd use a Named Entity Recognizer here)
        if best_intent == "query_state":
            input_symbol["parameters"]["target"] = "SOL_accumulation_status"

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        input_symbol["execution_time_ms"] = round(elapsed_ms, 4)
        input_symbol["debug_confidence_score"] = round(best_score, 4)
        
        return input_symbol

# --- Execution ---
if __name__ == "__main__":
    parser = SRM_Input_Parser()
    
    messy_user_input = "Yo, why did we stop buying?"
    print(f"\nUser Said: '{messy_user_input}'")
    
    symbol = parser.parse_natural_language(messy_user_input)
    
    print(f"ONNX debug score: {symbol['debug_confidence_score']} matched '{symbol['intent_class']}'")
    print(f"Layer 4 parsed text in {symbol['execution_time_ms']}ms\n")
    print("Generated Phase 1 Input Symbol:")
    print(json.dumps(symbol, indent=2))
