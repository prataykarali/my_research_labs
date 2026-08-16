"""
Comparative benchmark between LFM2.5-VL-450M and LFM2.5-1.2B-Instruct on memory grounding & persona retention.
"""

import os
import json
import time
from llama_cpp import Llama

MODELS = {
    "LFM2.5-VL-450M": "/home/pratay-karali/launch-AURA/AURA-Proj/aura_notebook/assets/aura-aria-gguf/LFM2.5-VL-450M.Q4_K_M.gguf",
    "LFM2.5-1.2B-Instruct": "/home/pratay-karali/launch-AURA/AURA-Proj/aura_notebook/assets/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
}

ARIA_SYSTEM_PROMPT = (
    "You are ARIA, a helpful, empathetic personal AI. "
    "Use the following known facts to answer the user accurately if relevant.\n"
)

TESTS = [
    {
        "type": "chit_chat",
        "prompt": "What is 15 * 14?",
        "context": ""
    },
    {
        "type": "memory_grounded",
        "prompt": "What snacks should I buy for my pet?",
        "context": "[KNOW: pet Mochi — User's orange cat [LIKES -> Salmon Treats (pref)]]"
    },
    {
        "type": "memory_grounded",
        "prompt": "Where do I currently live?",
        "context": "[KNOW: user — Primary User [LIVES_IN -> San Francisco (place)]]"
    }
]

all_comparisons = {}

for m_name, m_path in MODELS.items():
    if not os.path.exists(m_path):
        print(f"Skipping {m_name}, file not found at {m_path}")
        continue
    
    print(f"\nEvaluating {m_name} ({os.path.getsize(m_path)/(1024*1024):.1f} MB)...")
    llm = Llama(model_path=m_path, n_ctx=2048, n_threads=4, verbose=False)
    m_results = []
    
    for t in TESTS:
        sys_txt = ARIA_SYSTEM_PROMPT
        if t["context"]:
            sys_txt += f"Personal Memories:\n{t['context']}\n"
        
        prompt = f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{t['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        t0 = time.time()
        res = llm(prompt, max_tokens=64, temperature=0.1, stop=["<|im_end|>", "</s>"])
        dt = time.time() - t0
        
        ans = res["choices"][0]["text"].strip()
        toks = res["usage"]["completion_tokens"]
        print(f"  [{t['type']}] Prompt: '{t['prompt']}' -> '{ans}' ({dt*1000:.1f}ms, {toks/max(dt,1e-4):.1f} tok/s)")
        m_results.append({
            "type": t["type"],
            "prompt": t["prompt"],
            "context": t["context"],
            "answer": ans,
            "latency_ms": dt * 1000,
            "tok_per_sec": toks / max(dt, 1e-4)
        })
    all_comparisons[m_name] = m_results

with open("memory_research/lfm_model_comparison_results.json", "w") as f:
    json.dump(all_comparisons, f, indent=2)

print("\nSaved comparison to memory_research/lfm_model_comparison_results.json")
