"""
SFT & DPO Training Data Generator for On-Device Sub-2B SLM.
Generates:
1. Task 1: AKF JSON Graph Extraction (User Utterance -> Structured Nodes & Directed Edges)
2. Task 2: Grounded Persona Response Generation without Memory Spam (DPO: Chosen Natural vs Rejected Spam)
"""

import json
import os


def generate_sft_akf_extraction_data(output_file: str = "memory_research/sft_akf_extraction_train.jsonl"):
    """
    Generates training pairs to fine-tune sub-2B SLM for Atomic Knowledge Fragment extraction.
    Input: User turn conversation
    Output: Valid AKF JSON with nodes and directed edges.
    """
    samples = [
        {
            "instruction": "Extract personal knowledge nodes and directed relations in AKF JSON format. If the utterance contains only transient or generic chatter, return an empty graph.",
            "input": "I just adopted a cute little orange tabby cat named Mochi! He absolutely loves salmon puree treats.",
            "output": json.dumps({
                "nodes": [
                    {"name": "Mochi", "kind": "pet", "summary": "User's orange tabby cat", "attrs": {"species": "cat", "color": "orange"}},
                    {"name": "Salmon Puree", "kind": "pref", "summary": "Favorite treat for cat Mochi", "attrs": {"type": "cat_snack"}}
                ],
                "edges": [
                    {"src": "Mochi", "src_kind": "pet", "rel": "LIKES", "dst": "Salmon Puree", "dst_kind": "pref"}
                ]
            })
        },
        {
            "instruction": "Extract personal knowledge nodes and directed relations in AKF JSON format. If the utterance contains only transient or generic chatter, return an empty graph.",
            "input": "I'm currently taking CS224N with Professor Sarah Miller this spring.",
            "output": json.dumps({
                "nodes": [
                    {"name": "CS224N", "kind": "course", "summary": "Natural Language Processing course", "attrs": {"term": "Spring 2026"}},
                    {"name": "Dr. Sarah Miller", "kind": "person", "summary": "Professor of CS224N", "attrs": {"role": "instructor"}}
                ],
                "edges": [
                    {"src": "Dr. Sarah Miller", "src_kind": "person", "rel": "TEACHES", "dst": "CS224N", "dst_kind": "course"}
                ]
            })
        },
        {
            "instruction": "Extract personal knowledge nodes and directed relations in AKF JSON format. If the utterance contains only transient or generic chatter, return an empty graph.",
            "input": "I'm sitting in traffic right now and it's raining outside.",
            "output": json.dumps({
                "nodes": [],
                "edges": [],
                "ephemeral": True
            })
        },
        {
            "instruction": "Extract personal knowledge nodes and directed relations in AKF JSON format. If the utterance contains only transient or generic chatter, return an empty graph.",
            "input": "What is the capital of Australia?",
            "output": json.dumps({
                "nodes": [],
                "edges": [],
                "ephemeral": True
            })
        },
        {
            "instruction": "Extract personal knowledge nodes and directed relations in AKF JSON format. If the utterance contains only transient or generic chatter, return an empty graph.",
            "input": "I moved from Seattle to Austin last weekend.",
            "output": json.dumps({
                "nodes": [
                    {"name": "Austin", "kind": "place", "summary": "User's current residence city", "attrs": {"current": True}}
                ],
                "edges": [
                    {"src": "User", "src_kind": "person", "rel": "LIVES_IN", "dst": "Austin", "dst_kind": "place", "valid": 1}
                ]
            })
        }
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[SAVED] SFT AKF Extraction dataset: {output_file} ({len(samples)} exemplar pairs)")


def generate_dpo_persona_alignment_data(output_file: str = "memory_research/dpo_persona_alignment_train.jsonl"):
    """
    Generates DPO (Direct Preference Optimization) preference pairs:
    - Chosen: Natural, concise AURA persona grounding without memory spam.
    - Rejected: Generic AI memory spam ('As an AI, I recall you mentioned X...').
    """
    dpo_samples = [
        {
            "prompt": "[KNOW: pet Mochi — User's orange cat [LIKES -> Salmon Puree (pref)]]\nUser: What snack should I grab on my way home?",
            "chosen": "Grab some salmon puree treats—Mochi will be thrilled!",
            "rejected": "According to my personal memory database records, I remember you told me that you have an orange tabby cat named Mochi and that Mochi's favorite treat is Salmon Puree. Therefore, you should buy Salmon Puree."
        },
        {
            "prompt": "[KNOW: course CS224N — NLP with Deep Learning [TEACHES -> Dr. Sarah Miller]]\nUser: When should I visit office hours for my NLP class?",
            "chosen": "Check Dr. Miller's syllabus for CS224N—her office hours are usually posted on Canvas.",
            "rejected": "I have retrieved from your long-term memories that you are taking CS224N taught by Dr. Sarah Miller. As your AI assistant who remembers everything, I suggest visiting Dr. Sarah Miller."
        },
        {
            "prompt": "User: Can you explain quicksort?",
            "chosen": "Quicksort is a divide-and-conquer algorithm that selects a pivot element and partitions the array into sub-arrays of elements less than and greater than the pivot.",
            "rejected": "I can explain quicksort. Also, I remember you drive a green Subaru Outback and have a cat named Mochi. Quicksort works by..."
        }
    ]

    with open(output_file, "w") as f:
        for s in dpo_samples:
            f.write(json.dumps(s) + "\n")
    print(f"[SAVED] DPO Persona Alignment dataset: {output_file} ({len(dpo_samples)} preference pairs)")


if __name__ == "__main__":
    generate_sft_akf_extraction_data()
    generate_dpo_persona_alignment_data()
