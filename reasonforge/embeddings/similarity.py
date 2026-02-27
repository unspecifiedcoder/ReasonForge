"""
ReasonForge - Embedding Similarity Detector

Detect plagiarism between miner submissions using sentence embeddings.
Uses sentence-transformers/all-MiniLM-L6-v2 (fast, 384-dim).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger("reasonforge.embeddings")


class SimilarityDetector:
    """
    Detect plagiarism between miner submissions using sentence embeddings.
    Uses cosine similarity on normalized embedding vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.history_embeddings: List[np.ndarray] = []
        self.max_history = 5000

    def _get_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded embedding model: %s", self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers>=2.2.0"
                )
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Encode text into a normalized embedding vector."""
        model = self._get_model()
        return model.encode(text, normalize_embeddings=True)

    def embed_submission(self, response) -> np.ndarray:
        """Encode a ReasoningTask response into a single embedding vector."""
        steps = getattr(response, "reasoning_steps", None) or []
        if isinstance(steps, list):
            steps_text = " ".join(
                s.get("reasoning", "") if isinstance(s, dict) else str(s)
                for s in steps
            )
        else:
            steps_text = str(steps)

        final_answer = getattr(response, "final_answer", "") or ""
        full_text = f"{steps_text} {final_answer}".strip()

        if not full_text:
            return np.zeros(384)  # Default embedding dimension

        return self.embed_text(full_text)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        # Vectors are already normalized, so dot product = cosine similarity
        return float(np.dot(a, b))

    def check_against_batch(
        self, response, other_responses: list
    ) -> float:
        """
        Return max cosine similarity against other responses in this batch.

        Args:
            response: The ReasoningTask response to check.
            other_responses: List of other responses in the same batch.

        Returns:
            Max similarity score (0.0 to 1.0).
        """
        if not other_responses:
            return 0.0

        try:
            target_emb = self.embed_submission(response)
            other_embs = np.array([self.embed_submission(r) for r in other_responses])

            # Cosine similarities (embeddings are normalized)
            similarities = other_embs @ target_emb
            return float(np.max(similarities))
        except Exception as e:
            logger.warning("Batch similarity check failed: %s", e)
            return 0.0

    def check_against_history(self, response) -> float:
        """Check against historical submissions (cross-epoch plagiarism)."""
        if not self.history_embeddings:
            return 0.0

        try:
            target_emb = self.embed_submission(response)
            history_matrix = np.array(self.history_embeddings[-self.max_history:])
            similarities = history_matrix @ target_emb
            return float(np.max(similarities))
        except Exception as e:
            logger.warning("History similarity check failed: %s", e)
            return 0.0

    def add_to_history(self, response) -> None:
        """Store embedding for future cross-epoch checks."""
        try:
            emb = self.embed_submission(response)
            self.history_embeddings.append(emb)
            if len(self.history_embeddings) > self.max_history:
                self.history_embeddings = self.history_embeddings[-self.max_history:]
        except Exception as e:
            logger.debug("Failed to add to history: %s", e)

    def check_text_similarity(self, text_a: str, text_b: str) -> float:
        """Direct text-to-text similarity check."""
        try:
            emb_a = self.embed_text(text_a)
            emb_b = self.embed_text(text_b)
            return self.cosine_similarity(emb_a, emb_b)
        except Exception:
            return 0.0
