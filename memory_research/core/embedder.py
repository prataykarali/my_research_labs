"""
Dense Vector Embedding Engine using Snowflake Arctic XS.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class SnowflakeEmbeddingEngine:
    """Wrapper for Snowflake Arctic Embed XS (22M params, 384-d)."""
    MODEL_NAME = "Snowflake/snowflake-arctic-embed-xs"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.MODEL_NAME
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encodes texts to 384-dimensional dense vectors with asymmetric query prefixes."""
        if is_query:
            prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        else:
            prefixed = texts
        embeddings = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)
