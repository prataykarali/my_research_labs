"""
EdgeMem Core Package.
"""

from .embedder import SnowflakeEmbeddingEngine
from .gates import SmartIngestionGate, AdaptiveRetrievalGate
from .engine import EdgeMemEngine, MemoryEngine

__all__ = [
    "SnowflakeEmbeddingEngine",
    "SmartIngestionGate",
    "AdaptiveRetrievalGate",
    "EdgeMemEngine",
    "MemoryEngine"
]
