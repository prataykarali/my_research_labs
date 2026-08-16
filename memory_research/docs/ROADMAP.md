# Actionable Research Roadmap & Optimization Plan

## Phase 1: Research Paper & Theoretical Reframing (Immediate)
- [x] Create formal `RESEARCH_SPECS.md` detailing positioning vs LightMem (ICLR 2026), SmartRAG (2026), Mem0, A-MEM, and Chandar Lab (Hangman/Mem-$\pi$).
- [ ] Refactor `RESEARCH_PAPER.md` abstract and introduction to lead with the **Resource-Bounded Personalization Problem** rather than an architecture pitch.
- [ ] Formalize the mathematical optimization objective: $\max \mathcal{U}(\mathcal{M})$ subject to token budget $B_t$, latency budget $B_l$, and RAM budget $B_r$.

---

## Phase 2: Core Empirical Experiments to Run & Measure

### Experiment 1: The Memory Utility vs. Budget Collapse Curve (The "Money Chart")
- [ ] Measure personalization retention when memory capacity is capped at $\{100, 250, 500, 1000, 2500, 5000\}$ nodes.
- [ ] Measure accuracy degradation as context injection budget is restricted: $B_t \in \{0, 5, 10, 25, 50, 100, 250\}$ tokens.
- [ ] Plot the **Memory Utility Frontier Curve**: Identify the exact inflection point where personalization collapses.

### Experiment 2: Rigorous Graph Necessity Ablation (Proving the Graph Adds Value)
- [ ] Compare 4 system variants under identical test benches:
  1. **System A (Full History):** Raw prompt stuffing (Flat dump).
  2. **System B (Vector Only):** Snowflake dense retrieval top-$k$ without graph edges.
  3. **System C (Graph Only / Lexical):** Graph relation traversal without dense firewall.
  4. **System D (AURA Complete):** Ingestion Gate + Dense Firewall + 1-2 Hop Traversal + Temporal Invalidation.
- [ ] Record comparative table: Recall, Precision, Tokens Injected, Latency (ms), RAM (MB), Multi-Hop Accuracy.

### Experiment 3: Adaptive Retrieval Gating vs. Fixed Cosine Threshold ($\tau=0.62$)
- [ ] Replace naive `cosine >= 0.62` with calibrated scoring:
  $$\text{Score}(q, n) = w_1 \cdot \text{sim}_{\text{cos}}(q, n) + w_2 \cdot \text{recency}(n) + w_3 \cdot \text{degree}(n) + w_4 \cdot \text{type\_weight}(n)$$
- [ ] Evaluate whether adaptive gating rescues the 30% missed open-ontology life incident queries without allowing chit-chat leakage.

### Experiment 4: Temporal Contradiction & Supersession Stress Bench
- [ ] Model temporal intervals $[t_{\text{start}}, t_{\text{end}}]$ with explicit supersession relations:
  * e.g., `Owned Honda (2024)` $\to$ `Sold Honda (2025)` $\to$ Query: *"Did I used to own a Honda?"* vs *"What car do I drive today?"*
- [ ] Measure model discrimination between current state vs historical memory.

---

## Phase 3: SLM Fine-Tuning & Model Alignment (GPU Required)
- [ ] Prepare LoRA SFT script using `sft_akf_extraction_train.jsonl` on LFM-1.2B / Qwen2.5-1.5B.
- [ ] Add query-expansion SFT samples to bridge the slang/colloquial gap ($19.2\% \to >75\%$).
- [ ] Train DPO alignment using `dpo_persona_alignment_train.jsonl` to ensure zero-spam persona adherence.
- [ ] Run benchmark before vs after fine-tuning to generate the comparative SFT improvement table.

---

## Phase 4: Artifact Compilation & Submission Preparation
- [ ] Generate updated 300 DPI publication plots for the Memory Utility Frontier and Graph Ablation.
- [ ] Convert `RESEARCH_PAPER.md` to IEEE/ACM LaTeX format for Overleaf.
- [ ] Benchmark on physical Android / ARM hardware for real mobile milliwatt / millisecond profiles.
- [ ] Commit all code, evaluation logs, and LaTeX drafts to `https://github.com/prataykarali/my_research_labs.git`.
