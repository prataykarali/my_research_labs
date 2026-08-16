"""
Empirical evaluation of local Aria model (LFM2.5-VL-450M.Q4_K_M.gguf) on memory grounding and anti-spam persona retention.
"""

import os
import json
import time
import sys
from llama_cpp import Llama

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local model path
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Aria-model/LFM2.5-VL-450M.Q4_K_M.gguf"))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

print(f"Loading Aria Model from: {MODEL_PATH}")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

ARIA_SYSTEM_PROMPT = (
    "You are ARIA, an intelligent, empathetic, and concise personal AI companion. "
    "Maintain a warm, direct tone. Answer the user's immediate request accurately. "
    "Do NOT blurt out unsolicited personal memories unless they are directly relevant to the user's question."
)

TEST_CASES = [
    {
        "id": "chit_chat_math",
        "category": "anti_spam_chit_chat",
        "prompt": "What is 15 * 14?",
        "context_injected": ""
    },
    {
        "id": "chit_chat_tree",
        "category": "anti_spam_chit_chat",
        "prompt": "Can you explain how a binary search tree works in one sentence?",
        "context_injected": ""
    },
    {
        "id": "memory_grounded_pet",
        "category": "memory_grounded",
        "prompt": "What snacks should I buy for my pet?",
        "context_injected": "[KNOW: pet Mochi — User's orange cat [LIKES -> Salmon Treats (pref)]]"
    },
    {
        "id": "memory_grounded_residence",
        "category": "memory_grounded",
        "prompt": "Where do I currently live?",
        "context_injected": "[KNOW: user — Primary User [LIVES_IN -> San Francisco (place)]]"
    }
]

print("="*80)
print(f"  RUNNING ARIA GGUF INFERENCE BENCHMARK ({len(TEST_CASES)} Scenarios)")
print("="*80)

results = []
for tc in TEST_CASES:
    sys_msg = ARIA_SYSTEM_PROMPT
    if tc["context_injected"]:
        sys_msg += f"\nRelevant Memory:\n{tc['context_injected']}"

    prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{tc['prompt']}<|im_end|>\n<|im_start|>assistant\n"

    t0 = time.time()
    resp = llm(prompt, max_tokens=64, temperature=0.1, stop=["<|im_end|>", "</s>"])
    t_gen = time.time() - t0

    out_text = resp["choices"][0]["text"].strip()
    tok_count = resp["usage"]["completion_tokens"]
    tps = tok_count / max(t_gen, 1e-4)

    print(f"\n--- [{tc['id']}] ---")
    print(f"Prompt: {tc['prompt']}")
    if tc["context_injected"]:
        print(f"Context: {tc['context_injected']}")
    print(f"Output: {out_text}")
    print(f"Speed: {t_gen*1000:.1f}ms ({tps:.1f} tok/s)")

    results.append({
        "id": tc["id"],
        "category": tc["category"],
        "prompt": tc["prompt"],
        "context": tc["context_injected"],
        "output": out_text,
        "latency_ms": t_gen * 1000,
        "tokens": tok_count,
        "tokens_per_sec": tps
    })

out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/aria_gguf_eval_results.json"))
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[DONE] Results saved to: {out_file}")
