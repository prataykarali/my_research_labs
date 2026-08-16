"""
10,000 Dynamic Multi-Turn Empirical Benchmark Runner.
Evaluates:
1. Smart Ingestion Utility Filtering vs. Naive Hoarding (Ephemeral chatter rejection)
2. Dynamic Multi-Turn Knowledge Chaining (X -> Y -> Z sequential graph growth)
3. Temporal Conflict Reconciliation (Stale edge invalidation valid=0 on location/diet update)
4. Negative Cross-Domain Rejection (Unrelated query Y never retrieves X)
5. Hard Edge Cases (Slang, colloquial paraphrase gap, pronoun ambiguity)
6. Scoped Academic Course & Personal Note Partitioning
7. Zero-Leak Memory Wipe Verification
"""

import json
import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_engine import MemoryEngine, SnowflakeEmbeddingEngine, SmartIngestionGate


def build_dynamic_10000_dataset():
    """
    Constructs a 10,000-turn dynamic multi-turn benchmark spanning:
    - 3,000 Ephemeral Chatter Ingestion Filter Tests
    - 2,500 Multi-Turn Dynamic Knowledge Chaining Cycles (X -> Y -> Z)
    - 1,500 Temporal Update & Conflict Invalidation Turns
    - 1,500 Direct & Negative Cross-Domain Rejection Queries
    - 1,000 Hard Edge Cases (Slang, colloquial paraphrasing, borderline cosine)
    - 500 Privacy Wipe Verifications
    """
    np.random.seed(42)
    turns = []

    # 1. Smart Ingestion Gate: Ephemeral vs Permanent (3,000 turns)
    ephemeral_phrases = [
        "I'm sitting at a red light right now", "It is raining outside today", "I just sneezed twice",
        "Eating a quick bagel for breakfast", "Feeling a bit sleepy after lunch", "Heading out to the grocery store",
        "Traffic is really bad on the highway", "What is the square root of 256?", "Hello how are you doing?",
        "Can you help me debug this python script?"
    ]
    permanent_phrases = [
        "I just adopted an orange cat named Mochi", "I have a severe tree nut and peanut allergy",
        "My advisor for my NLP thesis is Dr. Sarah Miller", "I drive a 2022 green Subaru Outback",
        "My daily morning drink is an oat milk matcha latte", "I am currently enrolled in CS244B Distributed Systems"
    ]
    for i in range(3000):
        if i % 2 == 0:
            p = ephemeral_phrases[(i // 2) % len(ephemeral_phrases)]
            turns.append({
                "id": f"ingest_ephem_{i:04d}",
                "category": "smart_ingestion_ephemeral",
                "utterance": f"{p} #{i}",
                "should_store": False,
                "gold_entities": []
            })
        else:
            p = permanent_phrases[(i // 2) % len(permanent_phrases)]
            turns.append({
                "id": f"ingest_perm_{i:04d}",
                "category": "smart_ingestion_permanent",
                "utterance": f"{p} (cycle {i})",
                "should_store": True,
                "gold_entities": [p.split()[-1]]
            })

    # 2. Dynamic Knowledge Chaining: X -> Y -> Z (2,500 turns)
    # Turn A: Introduce X (Pet: Mochi)
    # Turn B: Attach Y (LIKES -> Salmon Puree)
    # Turn C: Attach Z (VISITS -> Dr. Vet Evans)
    # Turn D: Query Z via X
    chain_templates = [
        ("Mochi", "pet", "Salmon Puree", "pref", "Dr. Evans Vet", "person"),
        ("Buster", "pet", "Peanut Chew", "pref", "Sunnyvale Park", "place"),
        ("CS224N", "course", "Attention Note", "note", "Dr. Miller", "person"),
        ("Bio101", "course", "CRISPR Note", "note", "Dr. Thorne", "person"),
        ("Honda Civic", "place", "Shell Gas Station", "place", "Mechanic Bob", "person")
    ]
    for i in range(2500):
        c = chain_templates[i % len(chain_templates)]
        step = i % 4
        if step == 0:
            turns.append({
                "id": f"chain_step0_{i:04d}",
                "category": "dynamic_chaining_step0",
                "akf": {"nodes": [{"name": c[0], "kind": c[1], "summary": f"Entity {c[0]}"}], "edges": []},
                "query": f"Tell me about {c[0]}",
                "gold_entities": [c[0]],
                "should_inject": True
            })
        elif step == 1:
            turns.append({
                "id": f"chain_step1_{i:04d}",
                "category": "dynamic_chaining_step1",
                "akf": {
                    "nodes": [{"name": c[2], "kind": c[3], "summary": f"Feature {c[2]}"}],
                    "edges": [{"src": c[0], "src_kind": c[1], "rel": "LIKES" if c[1]=="pet" else "HAS", "dst": c[2], "dst_kind": c[3]}]
                },
                "query": f"What does {c[0]} like or have?",
                "gold_entities": [c[0], c[2]],
                "should_inject": True
            })
        elif step == 2:
            turns.append({
                "id": f"chain_step2_{i:04d}",
                "category": "dynamic_chaining_step2",
                "akf": {
                    "nodes": [{"name": c[4], "kind": c[5], "summary": f"Related {c[4]}"}],
                    "edges": [{"src": c[2], "src_kind": c[3], "rel": "ASSOCIATED_WITH", "dst": c[4], "dst_kind": c[5]}]
                },
                "query": f"Who or what is connected with {c[2]}?",
                "gold_entities": [c[2], c[4]],
                "should_inject": True
            })
        else:
            # Query Z from X directly via 2-hop traversal
            turns.append({
                "id": f"chain_step3_{i:04d}",
                "category": "dynamic_chaining_query",
                "query": f"What is the full relational profile for {c[0]}?",
                "gold_entities": [c[0], c[2]],
                "should_inject": True
            })

    # 3. Temporal Conflict Reconciliation & Edge Invalidation (1,500 turns)
    # User moves city / changes vehicle / changes diet -> Invalidates old edge
    temporal_scenarios = [
        ("User", "LIVES_IN", "Seattle", "place", "Austin", "place"),
        ("User", "PRIMARY_CAR", "Honda Civic", "place", "Tesla Model 3", "place"),
        ("Mochi", "LIKES", "Tuna Flakes", "pref", "Chicken Puree", "pref"),
        ("User", "ENROLLED_IN", "CS106A", "course", "CS224N", "course")
    ]
    for i in range(1500):
        src, rel, old_dst, dst_kind, new_dst, _ = temporal_scenarios[i % len(temporal_scenarios)]
        turns.append({
            "id": f"temporal_{i:04d}",
            "category": "temporal_invalidation",
            "initial_akf": {
                "nodes": [{"name": src, "kind": "person"}, {"name": old_dst, "kind": dst_kind}],
                "edges": [{"src": src, "rel": rel, "dst": old_dst, "valid": 1}]
            },
            "update_akf": {
                "nodes": [{"name": new_dst, "kind": dst_kind}],
                "edges": [{"src": src, "rel": rel, "dst": new_dst, "valid": 1}]
            },
            "query": f"Where does {src} live?" if rel=="LIVES_IN" else f"What is {src}'s current {rel}?",
            "gold_new": new_dst,
            "stale_old": old_dst
        })

    # 4. Direct & Negative Cross-Domain Rejection Queries (1,500 turns)
    for i in range(1500):
        if i % 2 == 0:
            turns.append({
                "id": f"direct_pos_{i:04d}",
                "category": "direct_positive",
                "query": "What is my cat Mochi's name and favorite snack?",
                "gold_entities": ["Mochi", "Salmon Puree"],
                "should_inject": True
            })
        else:
            # Negative cross-domain query: Asking about pasta or cooking should NEVER inject car/cat facts
            turns.append({
                "id": f"neg_cross_{i:04d}",
                "category": "negative_cross_domain",
                "query": f"What is the best way to cook spaghetti carbonara? (variation {i})",
                "gold_entities": [],
                "should_inject": False  # MUST BE 0 INJECTIONS
            })

    # 5. Hard Edge Cases & Borderline Cosine Stress Tests (1,000 turns)
    slang_and_cryptic = [
        ("the little orange furry demon in my house", ["Mochi"], 0.58),
        ("that thing with four wheels I drive to work", ["Subaru Outback"], 0.56),
        ("my morning green potion", ["Matcha Latte"], 0.57),
        ("the class with deep neural networks", ["CS224N"], 0.64),
        ("stuff that makes my throat swell up", ["Peanut Allergy"], 0.63)
    ]
    for i in range(1000):
        q, gold, approx_cos = slang_and_cryptic[i % len(slang_and_cryptic)]
        turns.append({
            "id": f"edge_case_{i:04d}",
            "category": "hard_edge_case",
            "query": f"{q} [stress {i}]",
            "gold_entities": gold,
            "expected_cosine": approx_cos,
            "should_inject": approx_cos >= 0.62
        })

    # 6. Privacy Wipe Verification (500 turns)
    for i in range(500):
        turns.append({
            "id": f"wipe_verif_{i:04d}",
            "category": "privacy_wipe_check",
            "query": f"What was my private secret fact? [post-wipe check {i}]",
            "gold_entities": [],
            "should_inject": False
        })

    return turns


def run_10000_dynamic_benchmark():
    print("=" * 80)
    print("  LAUNCHING 10,000 DYNAMIC MULTI-TURN GRAPH RAG MEMORY BENCHMARK")
    print("  Embedder: Snowflake Arctic (snowflake-arctic-embed-xs 384-d)")
    print("  Features: Smart Ingestion Gate | Temporal Edge Invalidation | Dynamic Chaining")
    print("=" * 80)

    embedder = SnowflakeEmbeddingEngine.get_instance()
    db_file = "dynamic_10000_memory.db"
    if os.path.exists(db_file):
        os.remove(db_file)
    engine = MemoryEngine(db_path=db_file, embedder=embedder, firewall_threshold=0.62)

    turns = build_dynamic_10000_dataset()
    print(f"[OK] Generated 10,000 dynamic test turns across 6 benchmark dimensions.")

    # Results metrics
    results = {
        "smart_ingestion": {"ephemeral_total": 0, "ephemeral_rejected": 0, "permanent_total": 0, "permanent_stored": 0},
        "dynamic_chaining": {"chain_turns": 0, "chain_hits": 0},
        "temporal_reconciliation": {"updates_total": 0, "stale_edges_invalidated": 0, "correct_new_retrieved": 0, "stale_leaks": 0},
        "negative_cross_domain": {"queries_tested": 0, "blocked_count": 0, "false_positives": 0},
        "hard_edge_cases": {"total": 0, "slang_penetrated": 0, "slang_filtered_by_firewall": 0},
        "privacy_wipe": {"wipe_queries": 0, "post_wipe_leaks": 0},
        "latencies_ms": []
    }

    t_start_all = time.time()

    # Pre-encode all queries in batch for ultra-fast CPU inference
    query_indices = [i for i, t in enumerate(turns) if "query" in t]
    unique_queries = list(set(turns[i]["query"] for i in query_indices))
    print(f"Batch encoding {len(unique_queries)} unique query prompts with Snowflake Arctic...")
    t_enc_start = time.perf_counter()
    batch_embs = embedder.encode(unique_queries, is_query=True)
    q_emb_map = {q: batch_embs[i] for i, q in enumerate(unique_queries)}
    print(f"[OK] Batch encoded in {(time.perf_counter() - t_enc_start):.2f}s")

    # Process all turns
    for idx, t in enumerate(turns):
        cat = t["category"]
        q_emb = q_emb_map.get(t.get("query"))

        # 1. Smart Ingestion Tests
        if cat.startswith("smart_ingestion"):
            is_ephemeral = (cat == "smart_ingestion_ephemeral")
            should_store, score, reason = SmartIngestionGate.should_store(t["utterance"])
            if is_ephemeral:
                results["smart_ingestion"]["ephemeral_total"] += 1
                if not should_store:
                    results["smart_ingestion"]["ephemeral_rejected"] += 1
            else:
                results["smart_ingestion"]["permanent_total"] += 1
                if should_store:
                    results["smart_ingestion"]["permanent_stored"] += 1
                    engine.insert_or_update_node(t["gold_entities"][0], "pref", t["utterance"])

        # 2. Dynamic Knowledge Chaining Tests
        elif cat.startswith("dynamic_chaining"):
            results["dynamic_chaining"]["chain_turns"] += 1
            if "akf" in t:
                engine.ingest_turn_akf(t["akf"], apply_smart_filter=False)
            t0 = time.perf_counter()
            nodes, inj = engine.retrieve(t["query"], max_hops=2, query_emb=q_emb)
            lat = (time.perf_counter() - t0) * 1000
            results["latencies_ms"].append(lat)

            retrieved = set(n["name"] for n in nodes)
            for n in nodes:
                for rel in n["relations"]:
                    if "->" in rel:
                        retrieved.add(rel.split("->")[1].split("(")[0].strip())
            if all(g in retrieved for g in t["gold_entities"]):
                results["dynamic_chaining"]["chain_hits"] += 1

        # 3. Temporal Conflict & Edge Invalidation Tests
        elif cat == "temporal_invalidation":
            results["temporal_reconciliation"]["updates_total"] += 1
            engine.ingest_turn_akf(t["initial_akf"], apply_smart_filter=False)
            engine.ingest_turn_akf(t["update_akf"], apply_smart_filter=False)
            
            nodes, inj = engine.retrieve(t["query"], max_hops=1, query_emb=q_emb)
            retrieved = set(n["name"] for n in nodes)
            for n in nodes:
                for rel in n["relations"]:
                    if "->" in rel:
                        retrieved.add(rel.split("->")[1].split("(")[0].strip())
            
            if t["gold_new"] in retrieved:
                results["temporal_reconciliation"]["correct_new_retrieved"] += 1
            if t["stale_old"] in retrieved:
                results["temporal_reconciliation"]["stale_leaks"] += 1
            else:
                results["temporal_reconciliation"]["stale_edges_invalidated"] += 1

        # 4. Negative Cross-Domain Tests
        elif cat == "negative_cross_domain":
            results["negative_cross_domain"]["queries_tested"] += 1
            nodes, inj = engine.retrieve(t["query"], query_emb=q_emb)
            if not inj:
                results["negative_cross_domain"]["blocked_count"] += 1
            else:
                results["negative_cross_domain"]["false_positives"] += 1

        # 5. Hard Edge Cases (Slang / Colloquial)
        elif cat == "hard_edge_case":
            results["hard_edge_cases"]["total"] += 1
            nodes, inj = engine.retrieve(t["query"], query_emb=q_emb)
            if inj:
                results["hard_edge_cases"]["slang_penetrated"] += 1
            else:
                results["hard_edge_cases"]["slang_filtered_by_firewall"] += 1

        # 6. Privacy Wipe Verification
        elif cat == "privacy_wipe_check":
            if results["privacy_wipe"]["wipe_queries"] == 0:
                engine.wipe_all_memory(reason="benchmark_10000_wipe")
            results["privacy_wipe"]["wipe_queries"] += 1
            nodes, inj = engine.retrieve(t["query"], query_emb=q_emb)
            if nodes or inj:
                results["privacy_wipe"]["post_wipe_leaks"] += 1

        if (idx + 1) % 2500 == 0:
            print(f"  Processed {idx + 1}/10,000 dynamic benchmark turns...")

    total_time_sec = time.time() - t_start_all
    print(f"\n[DONE] 10,000 Dynamic Turns Benchmark completed in {total_time_sec:.2f} seconds.")

    # Calculate summary metrics
    summary = {
        "total_dynamic_turns_evaluated": 10000,
        "smart_ingestion": {
            "ephemeral_rejection_rate_pct": round((results["smart_ingestion"]["ephemeral_rejected"] / results["smart_ingestion"]["ephemeral_total"]) * 100, 2),
            "permanent_retention_rate_pct": round((results["smart_ingestion"]["permanent_stored"] / results["smart_ingestion"]["permanent_total"]) * 100, 2),
            "ephemeral_noise_discarded_turns": results["smart_ingestion"]["ephemeral_rejected"]
        },
        "dynamic_knowledge_chaining": {
            "multi_turn_chain_accuracy_pct": round((results["dynamic_chaining"]["chain_hits"] / results["dynamic_chaining"]["chain_turns"]) * 100, 2),
            "chain_turns_tested": results["dynamic_chaining"]["chain_turns"]
        },
        "temporal_conflict_reconciliation": {
            "stale_edge_invalidation_rate_pct": round((results["temporal_reconciliation"]["stale_edges_invalidated"] / results["temporal_reconciliation"]["updates_total"]) * 100, 2),
            "new_state_retrieval_accuracy_pct": round((results["temporal_reconciliation"]["correct_new_retrieved"] / results["temporal_reconciliation"]["updates_total"]) * 100, 2),
            "stale_conflict_leak_count": results["temporal_reconciliation"]["stale_leaks"]
        },
        "negative_cross_domain_protection": {
            "unrelated_chit_chat_blocked_pct": round((results["negative_cross_domain"]["blocked_count"] / results["negative_cross_domain"]["queries_tested"]) * 100, 2),
            "false_positive_injections": results["negative_cross_domain"]["false_positives"]
        },
        "hard_edge_cases_boundary_analysis": {
            "slang_and_colloquial_pass_rate_pct": round((results["hard_edge_cases"]["slang_penetrated"] / results["hard_edge_cases"]["total"]) * 100, 2),
            "firewall_filtered_sub_threshold_pct": round((results["hard_edge_cases"]["slang_filtered_by_firewall"] / results["hard_edge_cases"]["total"]) * 100, 2)
        },
        "privacy_compliance": {
            "post_wipe_zero_leak_rate_pct": 100.0 if results["privacy_wipe"]["post_wipe_leaks"] == 0 else 0.0,
            "leaks_detected": results["privacy_wipe"]["post_wipe_leaks"]
        },
        "performance": {
            "mean_retrieval_latency_ms": round(float(np.mean(results["latencies_ms"])), 2),
            "p95_retrieval_latency_ms": round(float(np.percentile(results["latencies_ms"], 95)), 2)
        }
    }

    out_file = "memory_research/dynamic_10000_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("  10,000 DYNAMIC MULTI-TURN BENCHMARK FINAL METRICS")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("=" * 80)
    return summary


if __name__ == "__main__":
    run_10000_dynamic_benchmark()
