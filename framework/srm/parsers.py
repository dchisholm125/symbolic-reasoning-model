from typing import Dict, List, Any
import os

try:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

class IntentRouter:
    """
    Layer 4: 'The Ear'.
    Automatically maps messy natural language to strict predefined symbols.
    Uses ONNX / Cosine Similarity if installed, otherwise falls back to fast substring matches.
    """
    def __init__(self, intents: Dict[str, List[str]], model_path: str = "./layer4_onnx_model"):
        self.intents = intents
        self.model_path = model_path
        self._initialize_onnx()

    def _initialize_onnx(self):
        if HAS_ONNX and os.path.exists(os.path.join(self.model_path, "model.onnx")):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.session = ort.InferenceSession(
                os.path.join(self.model_path, "model.onnx"), 
                providers=["CPUExecutionProvider"]
            )
            # Precompute embeddings
            self.anchor_embeddings = {
                intent: [self._get_embedding(text) for text in phrases]
                for intent, phrases in self.intents.items()
            }
            self.use_onnx = True
        else:
            self.use_onnx = False

    def _get_embedding(self, text: str) -> 'np.ndarray':
        """Internal ONNX execution using Mean Pooling."""
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        if "token_type_ids" in inputs:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"]

        outputs = self.session.run(None, ort_inputs)
        token_embeddings = outputs[0]
        attention_mask = inputs["attention_mask"]
        
        input_mask_expanded = np.repeat(attention_mask[:, :, np.newaxis], token_embeddings.shape[2], axis=2)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        embedding = sum_embeddings / sum_mask
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return (embedding / norm)[0]

    def parse(self, text: str) -> str:
        """
        Takes raw string text, returns a strict intent symbol string.
        """
        text = text.lower()
        if self.use_onnx:
            user_embedding = self._get_embedding(text)
            best_intent = "invalid_intent"
            best_score = -1.0
            
            for intent, anchors in self.anchor_embeddings.items():
                for anchor in anchors:
                    score = float(np.dot(user_embedding, anchor) / (np.linalg.norm(user_embedding) * np.linalg.norm(anchor)))
                    if score > best_score:
                        best_score = score
                        best_intent = intent
            
            return best_intent if best_score > 0.40 else "invalid_intent"
        else:
            # Fallback primitive match
            for intent, phrases in self.intents.items():
                if any(phrase in text for phrase in phrases):
                    return intent
            return "invalid_intent"
