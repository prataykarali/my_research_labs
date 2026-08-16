"""
Ingestion & Retrieval Gating Mechanisms for Resource-Bounded Personal Memory.
"""

import re
from typing import Tuple, Dict, Any, List

class SmartIngestionGate:
    """
    Evaluates whether an incoming conversation utterance contains persistent autobiographical value
    or is ephemeral conversational noise.
    """
    EPHEMERAL_PATTERNS = [
        r"^(hi|hello|hey|greetings|what's up|yo)\b",
        r"^(thanks|thank you|ok|okay|cool|sure|got it|sounds good|alright|nice)\b",
        r"^(good morning|good night|good afternoon|bye|see you)\b",
        r"^(yes|no|yep|nope|maybe|haha|lol)\b",
        r"^(how are you|how's it going|what is the weather)\b"
    ]

    FACTUAL_INDICATORS = [
        r"\b(my name is|i am|i'm|i live in|i work as|my favorite|i love|i hate|i have a|i own)\b",
        r"\b(my dog|my cat|my pet|my mom|my dad|my sister|my brother|my wife|my husband)\b",
        r"\b(diagnosed with|allergic to|prescribed|born in|birthday is|enrolled in)\b",
        r"\b(started working|promoted to|graduated from|exam on|fractured|broke my)\b"
    ]

    @classmethod
    def should_store(cls, utterance: str) -> Tuple[bool, float, str]:
        """
        Returns (should_store, utility_score, reason).
        Utility score in [0.0, 1.0].
        """
        clean = utterance.strip().lower()
        if len(clean.split()) < 3 and not any(re.search(p, clean) for p in cls.FACTUAL_INDICATORS):
            return False, 0.1, "too_short_or_filler"

        for p in cls.EPHEMERAL_PATTERNS:
            if re.search(p, clean) and not any(re.search(f, clean) for f in cls.FACTUAL_INDICATORS):
                return False, 0.15, "ephemeral_chit_chat"

        for f in cls.FACTUAL_INDICATORS:
            if re.search(f, clean):
                return True, 0.95, "explicit_personal_fact"

        return True, 0.70, "general_informative_utterance"


class AdaptiveRetrievalGate:
    """
    Calibrated multi-factor retrieval scoring replacing static cosine threshold heuristics.
    Score(q, n) = w_sim * cos_sim + w_rec * recency + w_deg * degree + w_type * type_prior
    """
    def __init__(self,
                 w_sim: float = 0.70,
                 w_rec: float = 0.10,
                 w_deg: float = 0.10,
                 w_type: float = 0.10,
                 utility_threshold: float = 0.58):
        self.w_sim = w_sim
        self.w_rec = w_rec
        self.w_deg = w_deg
        self.w_type = w_type
        self.utility_threshold = utility_threshold

        self.TYPE_PRIORS = {
            "milestone": 1.0,
            "medical_incident": 1.0,
            "event": 0.9,
            "person": 0.85,
            "pref": 0.80,
            "course": 0.75,
            "topic": 0.60
        }

    def compute_utility(self,
                        cos_sim: float,
                        recency_ms: float,
                        degree: int,
                        kind: str) -> float:
        """Computes continuous retrieval utility in [0, 1]."""
        # Normalize recency (exponential decay over 30 days = 2.592e9 ms)
        rec_score = max(0.0, min(1.0, 1.0 / (1.0 + recency_ms / 2.592e9)))
        # Normalize degree
        deg_score = min(1.0, degree / 5.0)
        # Type prior
        type_score = self.TYPE_PRIORS.get(kind, 0.5)

        score = (self.w_sim * cos_sim +
                 self.w_rec * rec_score +
                 self.w_deg * deg_score +
                 self.w_type * type_score)
        return float(score)

    def is_retrievable(self, cos_sim: float, recency_ms: float, degree: int, kind: str) -> Tuple[bool, float]:
        """Returns (should_retrieve, utility_score)."""
        # Hard baseline guard to prevent absolute noise
        if cos_sim < 0.35:
            return False, cos_sim
        score = self.compute_utility(cos_sim, recency_ms, degree, kind)
        return (score >= self.utility_threshold), score
