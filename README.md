# My Research Labs: Resource-Bounded Lifelong Personal Memory

> **Research Paper & Experimental Laboratory**  
> *Learning What to Remember: Resource-Bounded Lifelong Personal Memory for Sub-2B On-Device Language Models*

---

## 🔬 Core Research Objective
How can a sub-2B on-device language model maintain dynamic, lifelong personal memory without being flooded with every fact the user ever mentioned?

This repository contains the complete experimental framework, empirical benchmarks (20,000+ evaluated turns), publication figures, GGUF inference evaluation harnesses, and academic manuscript characterizing the **Memory Utility vs. Resource Budget Pareto Frontier**.

---

## 📁 Repository Organization

```
my_research_labs/
├── Aria-model/                  # Local on-device GGUF SLM weights (ignored in git)
│   ├── LFM2.5-VL-450M.Q4_K_M.gguf
│   └── LFM2.5-VL-450M.F16-mmproj.gguf
│
└── memory_research/             # Modular Research Package
    ├── core/                    # Core Engine Architecture
    │   ├── engine.py            # EdgeMemEngine (SQLite graph, temporal validity intervals)
    │   ├── embedder.py          # Snowflake Arctic XS 384-d dense vector wrapper
    │   └── gates.py             # SmartIngestionGate & AdaptiveRetrievalGate
    │
    ├── cli/                     # Interactive CLI Interface
    │   └── main.py              # CLI tool: init-db, ingest-node, ingest-edge, retrieve, wipe
    │
    ├── benchmarks/              # Empirical Benchmark Suites
    │   ├── run_frontier.py      # Experiment 1: Memory Utility vs Budget Frontier
    │   ├── run_ablation.py      # Experiment 2: 4-Way Graph Necessity Ablation
    │   ├── run_multi_user.py    # 10-user multi-tenant isolation bench (10,000 cases)
    │   ├── run_dynamic_10k.py   # 10,000-turn dynamic multi-turn benchmark
    │   ├── run_advanced_scaling.py # Quantization, hop depth, energy, & topic drift
    │   └── run_open_ontology.py # Life incidents, medical, career, & events bench
    │
    ├── evaluations/             # Live On-Device GGUF SLM Inference
    │   ├── eval_aria_gguf.py    # Evaluates Aria model (LFM2.5-VL-450M) on memory grounding
    │   └── eval_lfm_comparison.py # LFM-450M vs LFM-1.2B head-to-head comparison
    │
    ├── training/                # SLM Fine-Tuning Datasets
    │   └── generate_sft_dpo_datasets.py # SFT (AKF extraction) & DPO (anti-spam persona)
    │
    ├── figures/                 # 14 Publication-Quality Figures (300 DPI)
    │   ├── generate_figures.py  # Reproduction script for all 14 figures
    │   ├── fig1_precision_recall_tradeoff.png
    │   ├── fig2_token_pollution_comparison.png
    │   ├── fig3_multihop_reasoning_accuracy.png
    │   ├── fig4_latency_and_footprint.png
    │   ├── fig5_overall_radar_comparison.png
    │   ├── fig6_multi_user_isolation.png
    │   ├── fig7_smart_ingestion_gate.png
    │   ├── fig8_dynamic_chaining_temporal.png
    │   ├── fig9_edge_case_privacy.png
    │   ├── fig10_lfm_aria_generation_evaluation.png
    │   ├── fig11_quantization_and_scaling.png
    │   ├── fig12_hop_depth_and_energy.png
    │   ├── fig13_memory_utility_frontier.png
    │   └── fig14_graph_necessity_ablation.png
    │
    ├── results/                 # JSON Output Logs for All Benchmarks
    ├── tests/                   # Unit and Integration Test Suite
    │   └── test_unit.py         # 8 automated tests (100% passing)
    │
    └── docs/                    # Academic Documentation
        ├── RESEARCH_PAPER.md    # Full Research Paper (6 tables, 14 figures)
        ├── RESEARCH_SPECS.md    # Literature positioning vs LightMem, SmartRAG, Chandar Lab
        └── ROADMAP.md           # 4-phase research roadmap
```

---

## 📊 Summary of Key Empirical Findings

| Benchmark Dimension | Measured Metric | Measured Value | Baseline / Target |
| :--- | :--- | :---: | :---: |
| **Token Economy** | Average Tokens Injected / Turn | **$7.8\text{ tokens}$** | $234.0\text{ tokens}$ ($96.7\%$ savings) |
| **Context Pollution** | False-Positive Chit-Chat Injections | **$0.0\%$ ($0/6000$)** | $100.0\%$ (Always-Inject) |
| **Retrieval Precision** | Macro Precision | **$99.8\%$** | $45.0\%$ (Flat Dump) |
| **Tenant Privacy** | Multi-User Isolation Leak Rate | **$0.0\%$ ($0/500$)** | $0.0\%$ Leak Target |
| **Smart Ingestion** | Ephemeral Noise Filtered | **$70.0\%$ ($1050/1500$)** | $0.0\%$ (Stores everything) |
| **Permanent Retention**| Autobiographical Facts Kept | **$100.0\%$ ($1500/1500$)** | $100.0\%$ Target |
| **Multi-Hop Traversal** | 1–2 Hop Reasoning Accuracy | **$94.5\% - 99.2\%$** | $38.5\%$ (Naïve Vector RAG) |
| **INT8 Quantization** | Vector RAM ($N=5,000$ Nodes) | **$1.91\text{ MB}$ ($99.992\%$ fidelity)** | $7.50\text{ MB}$ (FP32) |
| **CPU Retrieval Speed**| Mean Search + Hop Latency | **$0.76\text{ ms}$** | $< 15\text{ ms}$ Budget |
| **GGUF SLM Speed** | LFM-450M CPU Inference | **$17.8\text{ tok/s}$** | $> 5\text{ tok/s}$ Budget |

---

## 🚀 Getting Started

### 1. Run Automated Unit Tests
```bash
python3 memory_research/tests/test_unit.py
```

### 2. Run the Interactive CLI
```bash
python3 memory_research/cli/main.py --db my_memory.db init-db
python3 memory_research/cli/main.py --db my_memory.db ingest-node --name "Mochi" --kind "pet" --summary "Orange cat who loves salmon treats"
python3 memory_research/cli/main.py --db my_memory.db retrieve --query "What treats does my cat like?"
```

### 3. Run the On-Device Aria GGUF Evaluation
```bash
python3 memory_research/evaluations/eval_aria_gguf.py
```

### 4. Regenerate All 14 Figures
```bash
python3 memory_research/figures/generate_figures.py
```
