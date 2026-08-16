# Research Specifications: Resource-Bounded Lifelong Personal Memory for Sub-2B On-Device SLMs

## 1. Executive Summary & Thesis Pivot

### Core Thesis
> **"How can a sub-2B on-device language model selectively consolidate and retrieve lifelong personal memories under strict memory, latency, and context budgets without sacrificing personalization accuracy?"**

### The Pivot: Engineering System $\longrightarrow$ Constrained Optimization Theory
* **Old Framing (Engineering):** *"We built AURA-GraphRAG, a 3-layer Graph memory system for personal SLMs."*
* **New Framing (Scientific):** *"We characterize the **Memory Utility vs. Resource Budget Pareto Frontier** for sub-2B personal agents, determining the minimal memory and retrieval budget necessary to maintain lifelong personalization without catastrophic context degradation."*

---

## 2. Positioning Against Prior Art

| Prior Art / Literature | Their Contribution | Our Differentiation & Specific Research Addition |
| :--- | :--- | :--- |
| **LightMem (ICLR 2026)** | 3-stage memory (sensory filtering $\to$ STM $\to$ LTM); explicitly targets token/API efficiency. | **Very close conceptually.** We do NOT claim "3-layer human memory" as novel. Our contribution: **persistent typed relational memory + selective graph recall under sub-2B edge CPU constraints**. |
| **SmartRAG (July 2026)** | On-device RAG for mobile; quantized 1.7B backbone; graph memory + dense/lexical retrieval. | Proves 1.7B + graph on-device is viable. We do NOT claim "first on-device graph RAG". Our focus: **lifelong autobiographical personalization, temporal validity, and what deserves to be remembered**. |
| **Mem0** | Persistent user memory extracted from conversations and retrieved later. | Our distinction: **100% offline/local, typed relation graph, bounded token budget ($\le 10$ tokens), and zero-leak cryptographic privacy guarantees**. |
| **A-MEM** | Dynamic memory organization and inter-memory associative connections. | Our addition: **validity-aware temporal graph traversal + strict resource budget caps for $<2\text{B}$ models**. |
| **MemoryOS** | Hierarchical memory organization for long-term LLM interaction. | Our addition: **edge/on-device resource constraint optimization and Pareto boundary characterization**. |
| **Chandar Lab (2026) — "LLMs Can't Play Hangman"** | Establishes the necessity of **private working memory** separated from external interaction. | Directly aligns with our Layer 1 working buffer. We extend private working memory into **persistent, selective, resource-bounded personal long-term memory**. |
| **Chandar Lab (2026) — Mem-$\pi$** | Adaptive memory through learning *when and what to generate*. | Shows "adaptive memory" is active. Our twist: **adaptive memory gating under strict sub-2B compute and RAM constraints**. |
| **LongMemEval / LoCoMo** | Benchmarks for long-term conversational memory. | Our evaluation baseline framework. |

---

## 3. The 4 Defensible Research Contributions

### Contribution 1 (C1): Resource-Bounded Memory Policy
A selective consolidation mechanism that decides $\{\text{STORE}, \text{UPDATE}, \text{MERGE}, \text{DISCARD}\}$ based on expected future personalization utility under a hard capacity constraint $N_{\text{max}}$.

### Contribution 2 (C2): Temporal Personal-Memory Graph with Validity Intervals
A typed personal ontology (milestones, medical incidents, events, preferences, entities, relationships) featuring explicit temporal validity intervals $[t_{\text{start}}, t_{\text{end}}]$, supersession tracking, and contradiction reconciliation.

### Contribution 3 (C3): Budget-Aware Adaptive Retrieval Gating
Replacing hard-coded cosine threshold heuristics ($\tau = 0.62$) with an adaptive retrieval formulation:
$$\max U(\text{memory}) \quad \text{s.t.} \quad \text{Tokens} \le B_t, \quad \text{Latency} \le B_l, \quad \text{RAM} \le B_r$$
where retrieval probability is modeled as $P(\text{useful} \mid \text{similarity}, \text{type}, \text{recency}, \text{connectivity})$.

### Contribution 4 (C4): Empirical Memory-Efficiency Pareto Frontier
A rigorous characterization of how personalization fidelity decays as memory, context, and compute budgets are systematically compressed from $100\%$ down to $10\%$.

---

## 4. Forbidden Claims (What NOT to Claim)

Reviewers will reject papers claiming established concepts. **DO NOT CLAIM:**
- ❌ *"First GraphRAG memory system"* (Prior art: SmartRAG, Microsoft GraphRAG)
- ❌ *"First lifelong LLM memory"* (Prior art: Mem0, Generative Agents, MemoryOS)
- ❌ *"First selective / hierarchical memory"* (Prior art: LightMem ICLR 2026)
- ❌ *"First on-device RAG"* (Prior art: SmartRAG 2026)
- ❌ *"Human-like cognitive memory architecture"* (Cliche, overclaimed)

---

## 5. Formal Paper Title & Target Specs

### Paper Title
> **"Learning What to Remember: Resource-Bounded Lifelong Personal Memory for Sub-2B On-Device Language Models"**

### Target Resource Envelope
* **SLM Backbone:** Liquid Foundation Model (LFM2.5-1.2B-Instruct / LFM2.5-VL-450M) or Qwen2.5-1.5B (GGUF Q4_K_M)
* **Embedding Model:** Snowflake Arctic Embed XS (22M params, 384-d, INT8/FP16)
* **Max Active Context Injection:** $B_t \le 30\text{ tokens}$ (Avg: $7.8\text{ tokens}$)
* **Max Retrieval Latency:** $B_l \le 15\text{ ms}$ on CPU (Avg: $0.76\text{ ms}$)
* **Max System RAM Overhead:** $B_r \le 1.5\text{ GB}$ total (Model + SQLite Graph + Embedder)
