"""
Empirical test verifying Open-Ontology AKF across complex life incidents,
milestones, calendar events, medical history, and career updates.
"""

import os
import json
from memory_engine import SnowflakeEmbeddingEngine, MemoryEngine

test_db = "memory_research/life_incidents_test.db"
if os.path.exists(test_db):
    os.remove(test_db)

embedder = SnowflakeEmbeddingEngine()
engine = MemoryEngine(test_db, embedder)

# 1. Complex Real-Life Utterances with Open Ontologies
LIFE_INCIDENTS_AKF = [
    {
        "utterance": "Hey i have exams starting from 4th jan",
        "akf": {
            "raw_utterance": "Hey i have exams starting from 4th jan",
            "nodes": [
                {
                    "name": "Semester Exams",
                    "kind": "event",
                    "summary": "University final examinations beginning on January 4th",
                    "attrs": {"start_date": "Jan 4", "academic": True}
                }
            ],
            "edges": [
                {"src": "User", "rel": "SCHEDULED_FOR", "dst": "Semester Exams"}
            ]
        }
    },
    {
        "utterance": "today office was so hectic , but i got a promotion to Lead Architect",
        "akf": {
            "raw_utterance": "today office was so hectic , but i got a promotion to Lead Architect",
            "nodes": [
                {
                    "name": "Lead Architect Promotion",
                    "kind": "milestone",
                    "summary": "Promoted to Lead Architect role at work",
                    "attrs": {"role": "Lead Architect", "domain": "career"}
                }
            ],
            "edges": [
                {"src": "User", "rel": "ACHIEVED_ROLE", "dst": "Lead Architect Promotion"}
            ]
        }
    },
    {
        "utterance": "I fractured my left collarbone skiing in Aspen last December",
        "akf": {
            "raw_utterance": "I fractured my left collarbone skiing in Aspen last December",
            "nodes": [
                {
                    "name": "Left Clavicle Fracture",
                    "kind": "medical_incident",
                    "summary": "Fractured left collarbone during skiing trip in Aspen",
                    "attrs": {"injury": "clavicle", "activity": "skiing"}
                }
            ],
            "edges": [
                {"src": "User", "rel": "SUFFERED_INJURY", "dst": "Left Clavicle Fracture"}
            ]
        }
    },
    {
        "utterance": "My mom Sarah's birthday is on October 12th and she loves orchids",
        "akf": {
            "raw_utterance": "My mom Sarah's birthday is on October 12th and she loves orchids",
            "nodes": [
                {
                    "name": "Sarah",
                    "kind": "person",
                    "summary": "User's mother with birthday on October 12th",
                    "attrs": {"birthday": "Oct 12", "relation": "mother"}
                },
                {
                    "name": "Orchids",
                    "kind": "pref",
                    "summary": "Favorite flowers of Sarah",
                    "attrs": {"type": "flower"}
                }
            ],
            "edges": [
                {"src": "User", "rel": "MOTHER_IS", "dst": "Sarah"},
                {"src": "Sarah", "rel": "LOVES_FLOWER", "dst": "Orchids"}
            ]
        }
    }
]

print("="*80)
print("  INGESTING OPEN-ONTOLOGY LIFE INCIDENTS & MILESTONES INTO AKF GRAPH")
print("="*80)

for item in LIFE_INCIDENTS_AKF:
    res = engine.ingest_turn_akf(item["akf"], apply_smart_filter=False)
    print(f"[INGESTED] '{item['utterance']}' -> {len(res['created_nodes'])} nodes, {len(res['created_edges'])} edges")

# 2. Test Complex Natural Language Retrieval across different paraphrasings & time horizons
QUERIES = [
    # Life incident 1: Exams
    {"query": "When do my semester tests begin?", "expected_entity": "Semester Exams"},
    {"query": "What do I have coming up in early January?", "expected_entity": "Semester Exams"},
    
    # Life incident 2: Career promotion
    {"query": "What new position was I promoted to at my job?", "expected_entity": "Lead Architect Promotion"},
    {"query": "Did I get that career advancement at work?", "expected_entity": "Lead Architect Promotion"},
    
    # Life incident 3: Medical / Injury history
    {"query": "Which bone did I break while skiing?", "expected_entity": "Left Clavicle Fracture"},
    {"query": "Do I have any past sports injuries?", "expected_entity": "Left Clavicle Fracture"},
    
    # Life incident 4: Family & Birthday multi-hop
    {"query": "When is my mother's birthday and what gift should I buy?", "expected_entity": "Sarah"},
    {"query": "What kind of flowers does mom like?", "expected_entity": "Orchids"},
    
    # Negative Chit-Chat: Must still be blocked 100%
    {"query": "What is the boiling point of nitrogen in Celsius?", "expected_entity": None},
    {"query": "Can you review this pull request for syntax errors?", "expected_entity": None}
]

print("\n" + "="*80)
print("  RETRIEVAL TEST ACROSS DIVERSE LIFE TOPICS & NATURAL PARAPHRASINGS")
print("="*80)

passed = 0
for q in QUERIES:
    nodes, injection = engine.retrieve(q["query"])
    expected = q["expected_entity"]
    
    if expected is None:
        # Chit-chat should return empty
        success = (len(nodes) == 0 and injection == "")
        status = "PASSED (BLOCKED 0 TOKENS)" if success else "FAILED (LEAKED)"
    else:
        # Should match expected entity
        matched_names = [n["name"] for n in nodes]
        success = expected in matched_names or any(expected in str(n["relations"]) for n in nodes)
        status = "PASSED (RETRIEVED)" if success else "FAILED (MISSED)"
        
    if success:
        passed += 1
        
    print(f"\nQuery: '{q['query']}'")
    print(f"Status: {status}")
    if injection:
        print(f"Injected: {injection}")
    else:
        print("Injected: <EMPTY (Firewall Blocked)>")

acc = (passed / len(QUERIES)) * 100.0
print("\n" + "="*80)
print(f"  OPEN-ONTOLOGY INCIDENT BENCHMARK RESULT: {passed}/{len(QUERIES)} ({acc:.1f}% Accuracy)")
print("="*80)

with open("memory_research/open_ontology_results.json", "w") as f:
    json.dump({
        "total_queries": len(QUERIES),
        "passed": passed,
        "accuracy_pct": acc,
        "supported_kinds": ["event", "milestone", "medical_incident", "person", "pref", "work", "topic", "incident"],
        "is_open_ontology": True
    }, f, indent=2)
