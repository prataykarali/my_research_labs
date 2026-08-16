"""
Unit and Integration Test Suite for EdgeMem Engine.
"""

import unittest
import os
import shutil
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import EdgeMemEngine, SnowflakeEmbeddingEngine, SmartIngestionGate, AdaptiveRetrievalGate

class TestEdgeMemEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = SnowflakeEmbeddingEngine()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_mem.db")
        self.engine = EdgeMemEngine(self.db_path, self.embedder)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_01_schema_initialization(self):
        stats = self.engine.get_stats()
        self.assertEqual(stats["nodes"], 0)
        self.assertEqual(stats["edges"], 0)
        self.assertEqual(stats["facts"], 0)

    def test_02_node_insertion_and_embedding(self):
        nid = self.engine.insert_or_update_node("Mochi", "pet", "User's orange cat")
        self.assertIsInstance(nid, int)
        stats = self.engine.get_stats()
        self.assertEqual(stats["nodes"], 1)
        self.assertEqual(stats["facts"], 1)

    def test_03_smart_ingestion_filtering(self):
        store_hi, _, _ = SmartIngestionGate.should_store("Hello how are you?")
        self.assertFalse(store_hi)
        store_fact, _, _ = SmartIngestionGate.should_store("I have an orange cat named Mochi.")
        self.assertTrue(store_fact)

    def test_04_dense_retrieval_and_firewall(self):
        self.engine.insert_or_update_node("Mochi", "pet", "User's orange cat")
        # Relevant query
        nodes, inj = self.engine.retrieve("What is my cat's name?")
        self.assertGreater(len(nodes), 0)
        self.assertEqual(nodes[0]["name"], "Mochi")
        # Irrelevant query -> Blocked
        nodes_irr, inj_irr = self.engine.retrieve("What is the square root of 144?")
        self.assertEqual(len(nodes_irr), 0)
        self.assertEqual(inj_irr, "")

    def test_05_graph_relational_traversal(self):
        n1 = self.engine.insert_or_update_node("Mochi", "pet", "User's orange cat")
        n2 = self.engine.insert_or_update_node("Salmon Treats", "pref", "Favorite snack of Mochi")
        self.engine.add_edge(n1, "LIKES", n2)

        nodes, inj = self.engine.retrieve("What snacks does my cat love?")
        self.assertGreater(len(nodes), 0)
        self.assertTrue("Salmon Treats" in inj or nodes[0]["name"] == "Salmon Treats")

    def test_06_temporal_conflict_resolution(self):
        u1 = self.engine.insert_or_update_node("User", "person", "Primary User")
        p1 = self.engine.insert_or_update_node("New York", "place", "City in NY")
        p2 = self.engine.insert_or_update_node("San Francisco", "place", "City in CA")

        self.engine.add_or_update_edge(u1, "LIVES_IN", p1)
        self.engine.add_or_update_edge(u1, "LIVES_IN", p2)

        stats = self.engine.get_stats()
        self.assertEqual(stats["active_edges"], 1)
        self.assertEqual(stats["invalidated_edges"], 1)

    def test_07_zero_leak_memory_wipe(self):
        self.engine.insert_or_update_node("Secret", "pref", "Confidential preference")
        wipe_res = self.engine.wipe_all_memory("unit_test")
        self.assertGreater(wipe_res["nodes_deleted"], 0)
        stats = self.engine.get_stats()
        self.assertEqual(stats["nodes"], 0)
        self.assertEqual(stats["edges"], 0)

    def test_08_adaptive_retrieval_gating(self):
        gate = AdaptiveRetrievalGate(utility_threshold=0.55)
        # Moderate cosine sim for a milestone
        retrievable, score = gate.is_retrievable(cos_sim=0.56, recency_ms=1000, degree=2, kind="milestone")
        self.assertTrue(retrievable)

if __name__ == "__main__":
    unittest.main()
