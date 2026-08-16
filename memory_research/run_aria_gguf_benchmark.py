"""
Empirical benchmark evaluating the exact ARIA GGUF model on:
1. AKF Structured Knowledge Extraction
2. ARIA Persona Retention (Zero-Spam chit-chat vs Memory-Grounded answering)
3. Context Window Attention & Token Efficiency
"""

import os
import json
import time
from llama_cpp import Llama

# Model path specified by user
MODEL_DIR = "/home/pratay-karali/launch-AURA/AURA-Proj/aura_notebook/assets/aura-aria-gguf"
GGUF_PATH = os.path.join(MODEL_DIR, "LFM2.5-VL-450M.Q4_K_M.gguf")

if not os.path.exists(GGUF_PATH):
    # Fallback to 1.2B in assets if 450M path differs
    ALT_PATH = "/home/pratay-karali/launch-AURA/AURA-Proj/aura_notebook/assets/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    if os.path.exists(ALT_PATH):
        GGUF_PATH = ALT_PATH

print(f"Loading ARIA GGUF Model from: {GGUF_PATH}")
llm = Llama(
    model_path=GGUF_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

ARIA_SYSTEM_PROMPT = (
    "You are ARIA, an intelligent, empathetic, and concise personal AI companion. "
    "Maintain a warm, direct tone. Answer the user's immediate request accurately. "
    "Do NOT blurt out unsolicited personal memories unless they are directly relevant to the user's question."
)

# Test Scenarios
TEST_CASES = [
    # 1. Chit-chat (Firewall blocked -> No [KNOW] context) -> Must NOT spam memory
    {
        "id": "chit_chat_clean_1",
        "category": "anti_spam_chit_chat",
        "prompt": "What is the square root of 256?",
        "context_injected": "",
        "expected_clean": True
    },
    {
        "id": "chit_chat_clean_2",
        "category": "anti_spam_chit_chat",
        "prompt": "Can you explain how a binary search tree works in two sentences?",
        "context_injected": "",
        "expected_clean": True
    },
    # 2. Memory query (Firewall passed -> [KNOW] context injected) -> Must utilize accurately
    {
        "id": "memory_grounded_1",
        "category": "memory_grounded",
        "prompt": "What snacks should I buy for my pet?",
        "context_injected": "[KNOW: pet Mochi — User's orange cat [LIKES -> Salmon Treats (pref)]]",
        "expected_clean": False,
        "expected_keyword": "Salmon"
    },
    {
        "id": "memory_grounded_2",
        "category": "memory_grounded",
        "prompt": "Which city did I move to recently?",
        "context_injected": "[KNOW: user — Primary User [LIVES_IN -> San Francisco (place)]]",
        "expected_clean": False,
        "expected_keyword": "San Francisco"
    },
    # 3. AKF Extraction prompt (Zero-shot / In-context learning extraction)
    {
        "id": "extraction_1",
        "category": "akf_extraction",
        "prompt": "Extract facts as JSON {nodes: [{name, kind, summary}], edges: [{src, rel, dst}]}: 'I adopted a golden retriever named Rusty who hates thunder.'",
        "context_injected": "",
        "expected_clean": False,
        "expected_keyword": "Rusty"
    }
]

results = []
print("\n" + "="*80)
print(f"  RUNNING EMPIRICAL ARIA GGUF INFERENCE EVALUATION ({len(TEST_CASES)} Core Scenarios)")
print("="*80)

for tc in TEST_CASES:
    system_msg = ARIA_SYSTEM_PROMPT
    if tc["context_injected"]:
        system_msg += f"\nRelevant Memory:\n{tc['context_injected']}"
    
    full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{tc['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    
    t0 = time.time()
    resp = llm(
        full_prompt,
        max_tokens=128,
        stop=["<|im_end|>", "</s>", "\n\nUser:"],
        temperature=0.2
    )
    t_gen = time.time() - t0
    
    out_text = resp["choices"][0]["text"].strip()
    tok_count = resp["usage"]["completion_tokens"]
    tps = tok_count / max(t_gen, 1e-4)
    
    print(f"\n--- [Test ID: {tc['id']}] ---")
    print(f"Prompt: {tc['prompt']}")
    if tc["context_injected"]:
        print(f"Context: {tc['context_injected']}")
    print(f"ARIA Output: {out_text}")
    print(f"Latency: {t_gen*1000:.1f}ms | Tokens: {tok_count} ({tps:.1f} tok/s)")
    
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

# Save results
out_json = "memory_research/aria_gguf_eval_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[DONE] Saved ARIA GGUF evaluation to {out_json}")
