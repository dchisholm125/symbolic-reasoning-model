# SRM Architecture v1: The Context-Window Collapse

Status: Active Architecture Definition
Objective: Eliminate the quadratic scaling of KV Cache memory ("Context Window") by entirely offloading state tracking and long-term memory to a purely symbolic, Python-based Layer 1 state machine.

## 1. The Power of State Memory over KV Cache

Current Large Language Models attempt to achieve "memory" by constantly re-feeding massive chat transcripts into their context window. This uses immense amounts of VRAM and introduces quadratic latency.

**The SRM Solution:**
We decouple memory from language entirely. Layer 1 (The Conceptualization Engine) acts as the single source of truth for the "state of the world" via ultra-fast, lightweight dictionaries. Layer 3 (The LLM/SLM) processes only the instantaneous output of Layer 1. The language generator needs a context window of just ~100 tokens, as it only translates the exact instruction provided to it at that millisecond. 

*Memory becomes stateful, rather than transcript-based.*

## 2. The Hardest Challenge: The "Fast Parser" (Layer 4)

Extracting structural logic from messy human natural language is notoriously difficult without a massive instruction-tuned LLM. A pure classification model (like RoBERTa) will require significant domain-specific fine-tuning to perfectly generate symbols from complex requests (e.g., "Remember that rusted key I found? Does it open the door?"). 

If the architecture fails, it will likely be because the input from Layer 4 failed to accurately map natural language to the strict internal schema of Layer 1.

## 3. Phase 1: The Symbolic Alphabet (Schema)

To ensure the pipeline is robust, we strictly define the internal "Alphabet" used to represent the model's state and communicate between layers. This is the **Phase 1 JSON Schema**.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SRM Layer 1 Symbolic Alphabet",
  "description": "The strict structural interface for the Symbolic Reasoning Model. No natural language is permitted in values.",
  "type": "object",
  "properties": {
    "SystemState": {
      "description": "The persistent Layer 1 memory. The LLM never sees this directly.",
      "type": "object",
      "properties": {
        "entities": {
          "type": "object",
          "description": "Key-value map of known actors/items and their properties.",
          "additionalProperties": {
            "type": "object",
            "properties": {
              "id": { "type": "string" },
              "type": { "type": "string", "enum": ["agent", "item", "environment", "metric"] },
              "attributes": { "type": "object" }
            }
          }
        },
        "temporal_flags": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Active states (e.g., 'combat_active', 'volatility_high')"
        }
      }
    },
    "InputSymbol": {
      "description": "The output of Layer 4 (Input Parser). This is what triggers Layer 1.",
      "type": "object",
      "properties": {
        "intent_class": {
          "type": "string",
          "enum": ["query_state", "execute_action", "declare_fact", "invalid_input"]
        },
        "target_entity": { "type": "string" },
        "parameters": { "type": "object" }
      },
      "required": ["intent_class"]
    },
    "OutputSymbol": {
      "description": "The output of Layer 1 (Reasoning). This is passed to Layer 2 for formatting.",
      "type": "object",
      "properties": {
        "logic_status": {
          "type": "string",
          "enum": ["approved", "rejected", "state_updated", "requires_clarification"]
        },
        "state_delta": {
          "type": "object",
          "description": "What changed in Layer 1 memory as a result of the input."
        },
        "directive_code": {
          "type": "string",
          "description": "A strict code telling Layer 3 how to respond (e.g., 'CONFIRM_MATCH', 'DENY_ACCESS_MISSING_KEY')."
        }
      },
      "required": ["logic_status", "directive_code"]
    }
  }
}
```

## 4. Phase 2: The Dummy Core

The immediate objective is to implement "Layer 1" (The Conceptualization Engine/SRM Core) in pure Python to demonstrate deterministic state-tracking and symbolic deduction running in microseconds, avoiding the use of LLMs.

**Target Execution:** `SRM_Core` initialized with an inventory holding a rusted key, evaluating an `InputSymbol` to open an environmental "iron_gate" entity. 

*No AI models. Pure logical execution.*
