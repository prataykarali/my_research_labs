"""
EdgeMem Interactive Command Line Interface.
"""

import argparse
import json
import sys
import os

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import EdgeMemEngine, SnowflakeEmbeddingEngine

def main():
    parser = argparse.ArgumentParser(description="EdgeMem On-Device Personal Memory CLI")
    parser.add_argument("--db", default="edgemem_cli.db", help="Path to SQLite memory database")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subparsers.add_parser("init-db", help="Initialize a fresh EdgeMem SQLite database")

    p_node = subparsers.add_parser("ingest-node", help="Ingest or update a node entity")
    p_node.add_argument("--name", required=True, help="Entity name (e.g. 'Mochi')")
    p_node.add_argument("--kind", required=True, help="Entity kind (e.g. 'pet', 'person', 'milestone')")
    p_node.add_argument("--summary", required=True, help="Natural language summary")
    p_node.add_argument("--attrs", default="{}", help="JSON key-value metadata attributes")

    p_edge = subparsers.add_parser("ingest-edge", help="Create directed relationship edge between nodes")
    p_edge.add_argument("--src-id", type=int, required=True, help="Source node integer ID")
    p_edge.add_argument("--rel", required=True, help="Relationship label (e.g. 'LIKES')")
    p_edge.add_argument("--dst-id", type=int, required=True, help="Destination node integer ID")

    p_ret = subparsers.add_parser("retrieve", help="Perform two-pass guarded retrieval")
    p_ret.add_argument("--query", required=True, help="User natural language prompt")
    p_ret.add_argument("--hops", type=int, default=1, help="Max graph traversal hops (default: 1)")

    subparsers.add_parser("stats", help="Show database node, edge, and fact counts")

    p_wipe = subparsers.add_parser("wipe", help="Perform total memory wipe")
    p_wipe.add_argument("--reason", default="cli_user_request", help="Reason logged for the audit trail")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "init-db":
        engine = EdgeMemEngine(args.db)
        print(f"[SUCCESS] Initialized EdgeMem database at: {args.db}")
        return

    engine = EdgeMemEngine(args.db)

    if args.command == "ingest-node":
        attrs = json.loads(args.attrs)
        nid = engine.insert_or_update_node(args.name, args.kind, args.summary, attrs)
        print(f"[INGESTED NODE] ID={nid} | {args.name} ({args.kind})")

    elif args.command == "ingest-edge":
        engine.add_edge(args.src_id, args.rel, args.dst_id)
        print(f"[INGESTED EDGE] {args.src_id} -[{args.rel}]-> {args.dst_id}")

    elif args.command == "retrieve":
        nodes, injection = engine.retrieve(args.query, max_hops=args.hops)
        print(f"\nQuery: {args.query}")
        print(f"Retrieved Nodes Count: {len(nodes)}")
        if injection:
            print(f"\n[INJECTED CONTEXT]:\n{injection}\n")
        else:
            print("\n[FIREWALL RESULT]: Suppressed (0 tokens injected, pure chit-chat)\n")

    elif args.command == "stats":
        stats = engine.get_stats()
        print(json.dumps(stats, indent=2))

    elif args.command == "wipe":
        res = engine.wipe_all_memory(args.reason)
        print(f"[WIPE COMPLETE] {res}")

if __name__ == "__main__":
    main()
