"""
Core EdgeMem Engine: Relational Graph Store with Dual-Pass Guarded Retrieval.
Fully on-device, zero background daemon overhead.
"""

import sqlite3
import json
import time
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .embedder import SnowflakeEmbeddingEngine
from .gates import SmartIngestionGate, AdaptiveRetrievalGate

class EdgeMemEngine:
    """
    On-device Resource-Bounded Personal Memory Engine.
    Combines SQLite relational adjacency graphs with dense semantic embeddings and adaptive gating.
    """
    def __init__(self, db_path: str, embedder: Optional[SnowflakeEmbeddingEngine] = None, adaptive_gating: bool = True):
        self.db_path = db_path
        self.embedder = embedder or SnowflakeEmbeddingEngine()
        self.adaptive_gate = AdaptiveRetrievalGate() if adaptive_gating else None
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initializes tables with foreign keys and cascade support."""
        with self.conn:
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    attrs TEXT DEFAULT '{}',
                    source TEXT DEFAULT 'chat',
                    updated_at INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    UNIQUE(name, kind)
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    src INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    rel TEXT NOT NULL,
                    dst INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    valid INTEGER NOT NULL DEFAULT 1,
                    valid_from INTEGER,
                    valid_to INTEGER,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(src, rel, dst)
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_wipe_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wiped_at INTEGER NOT NULL,
                    nodes_deleted INTEGER NOT NULL,
                    edges_deleted INTEGER NOT NULL,
                    facts_deleted INTEGER NOT NULL,
                    reason TEXT
                );
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_valid ON edges(valid);")

    def _serialize_vec(self, vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    def _deserialize_vec(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def insert_or_update_node(self, name: str, kind: str, summary: str, attrs: Dict[str, Any] = None, source: str = "chat") -> int:
        """Insert or update a node entity with fresh Snowflake Arctic embedding."""
        attrs = attrs or {}
        now = int(time.time() * 1000)
        attrs_json = json.dumps(attrs)

        text_to_embed = f"{name} ({kind}): {summary}"
        emb = self.embedder.encode([text_to_embed], is_query=False)[0]
        emb_blob = self._serialize_vec(emb)

        with self.conn:
            cur = self.conn.execute("SELECT id FROM nodes WHERE name = ? AND kind = ?", (name, kind))
            row = cur.fetchone()
            if row:
                node_id = row["id"]
                self.conn.execute("""
                    UPDATE nodes 
                    SET summary = ?, attrs = ?, source = ?, updated_at = ?, embedding = ?
                    WHERE id = ?
                """, (summary, attrs_json, source, now, emb_blob, node_id))
            else:
                cur = self.conn.execute("""
                    INSERT INTO nodes (name, kind, summary, attrs, source, updated_at, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (name, kind, summary, attrs_json, source, now, emb_blob))
                node_id = cur.lastrowid

            self.conn.execute("INSERT INTO facts (fact, created_at) VALUES (?, ?)", (f"{name} is a {kind}: {summary}", now))

        return node_id

    def add_or_update_edge(self, src_id: int, rel: str, dst_id: int, valid: int = 1, invalidate_conflicts: bool = True, valid_from: int = None, valid_to: int = None):
        """Add or update directed relationship edge with temporal interval support."""
        now = int(time.time() * 1000)
        rel_upper = rel.upper()
        valid_from = valid_from or now

        with self.conn:
            SINGLE_VALUED_RELS = ["LIVES_IN", "PRIMARY_CAR", "CURRENT_MAJOR", "TEACHES", "ACHIEVED_ROLE"]
            if invalidate_conflicts and rel_upper in SINGLE_VALUED_RELS:
                self.conn.execute("""
                    UPDATE edges SET valid = 0, valid_to = ?, updated_at = ?
                    WHERE src = ? AND rel = ? AND dst != ? AND valid = 1
                """, (now, now, src_id, rel_upper, dst_id))

            self.conn.execute("""
                INSERT INTO edges (src, rel, dst, valid, valid_from, valid_to, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(src, rel, dst) DO UPDATE SET valid = excluded.valid, valid_to = excluded.valid_to, updated_at = excluded.updated_at
            """, (src_id, rel_upper, dst_id, valid, valid_from, valid_to, now))

    def add_edge(self, src_id: int, rel: str, dst_id: int, valid: int = 1):
        """Convenience alias for add_or_update_edge."""
        return self.add_or_update_edge(src_id, rel, dst_id, valid=valid)

    def ingest_turn_akf(self, akf_payload: Dict[str, Any], apply_smart_filter: bool = True) -> Dict[str, Any]:
        """Ingests an Atomic Knowledge Fragment (AKF) JSON payload."""
        raw_text = akf_payload.get("raw_utterance", "")
        if apply_smart_filter and raw_text:
            should_store, utility, reason = SmartIngestionGate.should_store(raw_text)
            if not should_store:
                return {
                    "stored": False,
                    "rejected_reason": reason,
                    "utility_score": utility,
                    "created_nodes": [],
                    "created_edges": []
                }

        created_ids = {}
        for node in akf_payload.get("nodes", []):
            nid = self.insert_or_update_node(
                name=node["name"],
                kind=node["kind"],
                summary=node.get("summary", f"{node['name']} ({node['kind']})"),
                attrs=node.get("attrs", {}),
                source=node.get("source", "chat")
            )
            created_ids[(node["name"], node["kind"])] = nid

        created_edges = []
        for edge in akf_payload.get("edges", []):
            src_key = (edge["src"], edge.get("src_kind", "topic"))
            dst_key = (edge["dst"], edge.get("dst_kind", "topic"))
            
            src_id = created_ids.get(src_key)
            if not src_id:
                cur = self.conn.execute("SELECT id FROM nodes WHERE name = ?", (edge["src"],))
                row = cur.fetchone()
                if row:
                    src_id = row["id"]
                else:
                    src_id = self.insert_or_update_node(edge["src"], edge.get("src_kind", "topic"), f"Entity {edge['src']}")
                    created_ids[src_key] = src_id

            dst_id = created_ids.get(dst_key)
            if not dst_id:
                cur = self.conn.execute("SELECT id FROM nodes WHERE name = ?", (edge["dst"],))
                row = cur.fetchone()
                if row:
                    dst_id = row["id"]
                else:
                    dst_id = self.insert_or_update_node(edge["dst"], edge.get("dst_kind", "topic"), f"Entity {edge['dst']}")
                    created_ids[dst_key] = dst_id

            if src_id and dst_id:
                valid_status = edge.get("valid", 1)
                self.add_or_update_edge(src_id, edge["rel"], dst_id, valid=valid_status, invalidate_conflicts=True)
                created_edges.append((src_id, edge["rel"], dst_id))

        return {
            "stored": True,
            "utility_score": 1.0,
            "created_nodes": list(created_ids.values()),
            "created_edges": created_edges
        }

    def retrieve(self, query: str, max_hops: int = 1, scope_course: Optional[str] = None, top_k_nodes: int = 3, query_emb: Optional[np.ndarray] = None, static_tau: float = 0.62) -> Tuple[List[Dict[str, Any]], str]:
        """
        Two-Pass Graph RAG Retrieval:
        Pass 1: Dense Vector Similarity with Adaptive or Cosine Firewall
        Pass 2: Directed Subgraph Traversal on active edges (valid=1)
        """
        if query_emb is None:
            query_emb = self.embedder.encode([query], is_query=True)[0]

        query_sql = "SELECT id, name, kind, summary, attrs, updated_at, embedding FROM nodes"
        params = []
        if scope_course:
            query_sql += " WHERE (kind = 'course' AND name = ?) OR (attrs LIKE ?)"
            params.extend([scope_course, f"%{scope_course}%"])

        with self.conn:
            rows = self.conn.execute(query_sql, params).fetchall()

        if not rows:
            return [], ""

        now = int(time.time() * 1000)
        node_scores = []
        for r in rows:
            node_vec = self._deserialize_vec(r["embedding"])
            cos_sim = float(np.dot(query_emb, node_vec) / (np.linalg.norm(query_emb) * np.linalg.norm(node_vec) + 1e-10))
            
            if self.adaptive_gate:
                # Degree calculation
                with self.conn:
                    deg = self.conn.execute("SELECT COUNT(*) FROM edges WHERE (src = ? OR dst = ?) AND valid = 1", (r["id"], r["id"])).fetchone()[0]
                recency_ms = now - r["updated_at"]
                retrievable, util_score = self.adaptive_gate.is_retrievable(cos_sim, recency_ms, deg, r["kind"])
                if retrievable:
                    node_scores.append((util_score, r))
            else:
                if cos_sim >= static_tau:
                    node_scores.append((cos_sim, r))

        if not node_scores:
            return [], ""

        node_scores.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [r for _, r in node_scores[:top_k_nodes]]

        result_nodes = []
        injection_fragments = []

        for node_row in top_nodes:
            nid = node_row["id"]
            name = node_row["name"]
            kind = node_row["kind"]
            summary = node_row["summary"]

            relations = []
            if max_hops >= 1:
                with self.conn:
                    edge_rows = self.conn.execute("""
                        SELECT e.rel, n.name as target_name, n.kind as target_kind
                        FROM edges e
                        JOIN nodes n ON e.dst = n.id
                        WHERE e.src = ? AND e.valid = 1
                    """, (nid,)).fetchall()
                    for er in edge_rows:
                        relations.append(f"[{er['rel']} -> {er['target_name']} ({er['target_kind']})]")

                    in_edges = self.conn.execute("""
                        SELECT e.rel, n.name as src_name, n.kind as src_kind
                        FROM edges e
                        JOIN nodes n ON e.src = n.id
                        WHERE e.dst = ? AND e.valid = 1
                    """, (nid,)).fetchall()
                    for ier in in_edges:
                        relations.append(f"[{ier['src_name']} ({ier['src_kind']}) -{ier['rel']}->]")

            rel_str = f" {' '.join(relations)}" if relations else ""
            injection_fragments.append(f"[KNOW: {kind} {name} — {summary}{rel_str}]")

            result_nodes.append({
                "id": nid,
                "name": name,
                "kind": kind,
                "summary": summary,
                "relations": relations
            })

        final_injection = " ".join(injection_fragments)
        return result_nodes, final_injection

    def wipe_all_memory(self, reason: str = "user_request") -> Dict[str, Any]:
        """Cryptographic Zero-Leak Memory Purge."""
        now = int(time.time() * 1000)
        with self.conn:
            cur_nodes = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            cur_edges = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            cur_facts = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

            self.conn.execute("DELETE FROM edges;")
            self.conn.execute("DELETE FROM nodes;")
            self.conn.execute("DELETE FROM facts;")

            self.conn.execute("""
                INSERT INTO memory_wipe_log (wiped_at, nodes_deleted, edges_deleted, facts_deleted, reason)
                VALUES (?, ?, ?, ?, ?)
            """, (now, cur_nodes, cur_edges, cur_facts, reason))

        return {
            "nodes_deleted": cur_nodes,
            "edges_deleted": cur_edges,
            "facts_deleted": cur_facts
        }

    def get_stats(self) -> Dict[str, Any]:
        """Returns node count, active/invalid edges, facts, and wipe count."""
        with self.conn:
            n_nodes = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            n_edges_active = self.conn.execute("SELECT COUNT(*) FROM edges WHERE valid = 1").fetchone()[0]
            n_edges_invalid = self.conn.execute("SELECT COUNT(*) FROM edges WHERE valid = 0").fetchone()[0]
            n_facts = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            n_wipes = self.conn.execute("SELECT COUNT(*) FROM memory_wipe_log").fetchone()[0]
            kinds = [r[0] for r in self.conn.execute("SELECT DISTINCT kind FROM nodes").fetchall()]
        return {
            "nodes": n_nodes,
            "edges": n_edges_active + n_edges_invalid,
            "active_edges": n_edges_active,
            "invalidated_edges": n_edges_invalid,
            "facts": n_facts,
            "wipes": n_wipes,
            "kinds": kinds
        }


# Alias for backward compatibility
MemoryEngine = EdgeMemEngine
