"""
Comprehensive Unit and Integration Test Suite for On-Device Graph RAG Memory Layer.
Tests:
1. Schema integrity (nodes, edges, facts, memory_wipe_log)
2. Snowflake Arctic dense embedding representation (384-d)
3. Dual-write consistency (nodes + legacy facts)
4. Cosine Firewall threshold gating (tau = 0.62)
5. 1-2 Hop Graph Traversal & [KNOW:] formatting
6. Personal Note & Course Scoped Isolation
7. Zero-Leak Memory Wipe & Audit Logging
8. CLI Subprocess Interface Verification
"""

import unittest
import os
import sys
import subprocess
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_engine import MemoryEngine, SnowflakeEmbeddingEngine


class TestMemoryEngine(unittest.TestCase):

    def setUp(self):
        self.db_path = "test_memory_scratch.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.embedder = SnowflakeEmbeddingEngine.get_instance()
        self.engine = MemoryEngine(db_path=self.db_path, embedder=self.embedder, firewall_threshold=0.62)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_01_schema_initialization(self):
        """Verify all SQLite tables and indexes exist."""
        cur = self.engine.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = set(r[0] for r in cur.fetchall())
        self.assertIn("nodes", tables)
        self.assertIn("edges", tables)
        self.assertIn("facts", tables)
        self.assertIn("memory_wipe_log", tables)

    def test_02_snowflake_embedding_dimensions(self):
        """Verify Snowflake Arctic produces normalized 384-d vectors."""
        vec = self.embedder.encode(["Test embedding sentence"])[0]
        self.assertEqual(len(vec), 384)
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_03_dual_write_consistency(self):
        """Verify adding a node updates both nodes and legacy facts table."""
        nid = self.engine.insert_or_update_node(
            name="Mochi",
            kind="pet",
            summary="User's orange tabby cat",
            attrs={"color": "orange"}
        )
        self.assertGreater(nid, 0)
        
        # Verify node
        nrow = self.engine.conn.execute("SELECT name, kind, summary FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(nrow["name"], "Mochi")
        self.assertEqual(nrow["kind"], "pet")

        # Verify dual-written fact
        frow = self.engine.conn.execute("SELECT fact FROM facts").fetchone()
        self.assertIn("Mochi is a pet", frow["fact"])

    def test_04_cosine_firewall_rejection(self):
        """Verify casual chit-chat is blocked by tau=0.62 firewall (0 injection)."""
        self.engine.insert_or_update_node("Mochi", "pet", "User's orange cat")
        
        # Casual query
        nodes, injection = self.engine.retrieve("Hey there, how is the weather today?")
        self.assertEqual(len(nodes), 0)
        self.assertEqual(injection, "")

    def test_05_personal_query_retrieval_and_formatting(self):
        """Verify targeted personal query penetrates firewall and outputs [KNOW:]."""
        self.engine.insert_or_update_node("Mochi", "pet", "User's orange cat")
        
        nodes, injection = self.engine.retrieve("What is my cat's name?")
        self.assertGreater(len(nodes), 0)
        self.assertEqual(nodes[0]["name"], "Mochi")
        self.assertTrue(injection.startswith("[KNOW: pet Mochi — User's orange cat"))

    def test_06_graph_edge_traversal(self):
        """Verify directed edge connection between nodes is traversed."""
        n1 = self.engine.insert_or_update_node("Mochi", "pet", "User's orange cat")
        n2 = self.engine.insert_or_update_node("Salmon Treats", "pref", "Favorite snack of Mochi")
        self.engine.add_edge(n1, "LIKES", n2)

        nodes, injection = self.engine.retrieve("What snacks does my cat love?")
        self.assertGreater(len(nodes), 0)
        # Should include relation
        found_rel = False
        for n in nodes:
            if "Salmon Treats" in str(n["relations"]) or n["name"] == "Salmon Treats":
                found_rel = True
        self.assertTrue(found_rel)

    def test_07_course_and_note_scoping(self):
        """Verify kind='course' and kind='note' isolation."""
        c1 = self.engine.insert_or_update_node("CS101", "course", "Intro to Python Programming")
        n1 = self.engine.insert_or_update_node("Loops Note", "note", "For and while loop syntax", attrs={"course": "CS101"})
        c2 = self.engine.insert_or_update_node("Bio101", "course", "Intro to Biology")
        n2 = self.engine.insert_or_update_node("Cell Note", "note", "Membrane dynamics", attrs={"course": "Bio101"})

        # Retrieve scoped to CS101
        nodes, _ = self.engine.retrieve("loop syntax", scope_course="CS101")
        for n in nodes:
            self.assertNotEqual(n["name"], "Cell Note")

    def test_08_zero_leak_memory_wipe(self):
        """Verify total wipe purges data and logs event in memory_wipe_log."""
        self.engine.insert_or_update_node("Secret Fact", "pref", "Private user preference")
        stats_before = self.engine.get_stats()
        self.assertGreater(stats_before["nodes"], 0)

        wipe_res = self.engine.wipe_all_memory(reason="unit_test")
        self.assertGreater(wipe_res["nodes_deleted"], 0)

        stats_after = self.engine.get_stats()
        self.assertEqual(stats_after["nodes"], 0)
        self.assertEqual(stats_after["edges"], 0)
        self.assertEqual(stats_after["facts"], 0)
        self.assertEqual(stats_after["wipes"], 1)

        # Confirm post-wipe query returns 0
        nodes, injection = self.engine.retrieve("Secret Fact")
        self.assertEqual(len(nodes), 0)
        self.assertEqual(injection, "")

    def test_09_cli_interface_execution(self):
        """Verify CLI subcommands operate via subprocess."""
        cli_path = os.path.join(os.path.dirname(__file__), "memory_cli.py")
        
        # Test add-node
        res = subprocess.run([
            sys.executable, cli_path, "--db", self.db_path,
            "add-node", "--name", "Luna", "--kind", "pet", "--summary", "Golden Retriever dog"
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("[OK] Added/Updated Node ID", res.stdout)

        # Test stats
        res = subprocess.run([
            sys.executable, cli_path, "--db", self.db_path, "stats"
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn('"nodes": 1', res.stdout)

        # Test query
        res = subprocess.run([
            sys.executable, cli_path, "--db", self.db_path,
            "query", "What is my dog's name?"
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("[KNOW: pet Luna", res.stdout)


if __name__ == "__main__":
    unittest.main()
