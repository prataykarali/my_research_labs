#!/usr/bin/env python3
"""
CLI Tool for On-Device Graph RAG Personal Memory Layer.
Demonstrates:
- AKF Node & Edge Ingestion
- Two-Pass Vector + Graph Retrieval
- Cosine Firewall (tau = 0.62) Verification
- Course / Note Isolation
- Zero-Leak Memory Wipe
"""

import argparse
import json
import sys
import os

# Add local path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_engine import MemoryEngine, SnowflakeEmbeddingEngine


def parse_args():
    parser = argparse.ArgumentParser(description="AURA Graph RAG On-Device Memory Layer CLI")
    parser.add_argument("--db", type=str, default="aura_memory.db", help="Path to SQLite memory database")
    parser.add_argument("--firewall", type=float, default=0.62, help="Cosine firewall threshold (default: 0.62)")
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Ingest node
    node_p = subparsers.add_parser("add-node", help="Add or update a knowledge graph node")
    node_p.add_argument("--name", type=str, required=True, help="Entity name (e.g., Mochi)")
    node_p.add_argument("--kind", type=str, required=True, choices=["person", "pet", "place", "pref", "note", "course", "topic"], help="Ontology kind")
    node_p.add_argument("--summary", type=str, required=True, help="Summary description")
    node_p.add_argument("--attrs", type=str, default="{}", help="JSON attributes string")

    # Add edge
    edge_p = subparsers.add_parser("add-edge", help="Add directed relationship edge between nodes")
    edge_p.add_argument("--src", type=int, required=True, help="Source node ID")
    edge_p.add_argument("--rel", type=str, required=True, help="Relation (e.g., HAS, LIKES, OWNED_BY, REQUIRES)")
    edge_p.add_argument("--dst", type=int, required=True, help="Destination node ID")

    # Ingest AKF JSON
    akf_p = subparsers.add_parser("ingest-akf", help="Ingest an Atomic Knowledge Fragment JSON string or file")
    akf_p.add_argument("--json", type=str, help="Raw AKF JSON string")
    akf_p.add_argument("--file", type=str, help="Path to AKF JSON file")

    # Query / Retrieve
    query_p = subparsers.add_parser("query", help="Retrieve memories for a user prompt with firewall & graph traversal")
    query_p.add_argument("prompt", type=str, help="User query / prompt text")
    query_p.add_argument("--hops", type=int, default=1, help="Max graph hops (default: 1)")
    query_p.add_argument("--course", type=str, default=None, help="Scope retrieval to specific course")

    # Stats
    subparsers.add_parser("stats", help="Show memory graph summary statistics")

    # Wipe
    wipe_p = subparsers.add_parser("wipe", help="Zero-leak purge of all personal memory")
    wipe_p.add_argument("--reason", type=str, default="cli_request", help="Reason for memory wipe")

    # Interactive REPL
    subparsers.add_parser("interactive", help="Start interactive CLI REPL session")

    return parser.parse_args()


def interactive_repl(engine: MemoryEngine):
    print("=" * 65)
    print("  AURA On-Device Personal Memory CLI (Snowflake Arctic Embed)")
    print(f"  Firewall Threshold: {engine.firewall_threshold}")
    print("  Commands: /ingest <json>, /query <text>, /wipe, /stats, /exit")
    print("=" * 65)

    while True:
        try:
            line = input("\n[AURA-Memory]> ").strip()
            if not line:
                continue
            if line in ["/exit", "exit", "quit", ":q"]:
                print("Exiting memory CLI.")
                break
            elif line.startswith("/ingest "):
                payload_str = line[8:].strip()
                payload = json.loads(payload_str)
                ids = engine.ingest_turn_akf(payload)
                print(f"[SUCCESS] Ingested AKF. Node IDs: {ids}")
            elif line.startswith("/query "):
                q = line[7:].strip()
                nodes, injection = engine.retrieve(q)
                print(f"\n--- Retrieved {len(nodes)} Nodes ---")
                for n in nodes:
                    print(f" * [{n['kind'].upper()}] {n['name']} (score: {n['score']:.4f}) — {n['summary']}")
                    if n["relations"]:
                        print(f"    Edges: {', '.join(n['relations'])}")
                print("\n--- LLM Context Injection Block ---")
                if injection:
                    print(injection)
                else:
                    print("(BLOCKED BY FIREWALL: No context injected to avoid pollution)")
            elif line == "/stats":
                st = engine.get_stats()
                print(f"[STATS] Nodes: {st['nodes']} | Edges: {st['edges']} | Facts: {st['facts']} | Wipes: {st['wipes']}")
                print(f"[STATS] Active Kinds: {st['kinds']}")
            elif line.startswith("/wipe"):
                res = engine.wipe_all_memory(reason="cli_interactive")
                print(f"[WIPED] Cleaned memory: {res}")
            else:
                # Default treat as query
                nodes, injection = engine.retrieve(line)
                if injection:
                    print("\n" + injection)
                else:
                    print("(Firewall: tau < 0.62 -> Zero injection)")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as ex:
            print(f"[ERROR] {ex}")


def main():
    args = parse_args()
    if not args.command and not sys.argv[1:]:
        print("Use --help for CLI instructions or run with 'interactive' subcommand.")
        sys.exit(0)

    # Initialize embedder and memory engine
    embedder = SnowflakeEmbeddingEngine.get_instance()
    engine = MemoryEngine(db_path=args.db, embedder=embedder, firewall_threshold=args.firewall)

    if args.command == "add-node":
        attrs = json.loads(args.attrs)
        nid = engine.insert_or_update_node(args.name, args.kind, args.summary, attrs=attrs)
        print(f"[OK] Added/Updated Node ID {nid}: {args.name} ({args.kind})")

    elif args.command == "add-edge":
        engine.add_edge(args.src, args.rel, args.dst)
        print(f"[OK] Added Edge: Node {args.src} --[{args.rel}]--> Node {args.dst}")

    elif args.command == "ingest-akf":
        if args.file:
            with open(args.file, "r") as f:
                data = json.load(f)
        elif args.json:
            data = json.loads(args.json)
        else:
            print("[ERROR] Please provide --json or --file")
            sys.exit(1)
        node_ids = engine.ingest_turn_akf(data)
        print(f"[OK] Ingested AKF Payload. Nodes Created/Updated: {node_ids}")

    elif args.command == "query":
        nodes, injection = engine.retrieve(args.prompt, max_hops=args.hops, scope_course=args.course)
        print(f"Matched Nodes: {len(nodes)}")
        for n in nodes:
            print(f" - [{n['kind']}] {n['name']} (sim={n['score']:.4f})")
            if n['relations']:
                print(f"    relations: {n['relations']}")
        print("\n--- Formatted Injection Prefix ---")
        print(injection if injection else "<EMPTY (Firewall Blocked)>")

    elif args.command == "stats":
        stats = engine.get_stats()
        print(json.dumps(stats, indent=2))

    elif args.command == "wipe":
        res = engine.wipe_all_memory(reason=args.reason)
        print(f"[SUCCESS] Zero-leak wipe complete: {res}")

    elif args.command == "interactive":
        interactive_repl(engine)


if __name__ == "__main__":
    main()
