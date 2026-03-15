"""
ReasonForge - Embedding Tests

Tests for similarity detection using sentence embeddings.
Note: Actual embedding model tests require sentence-transformers installed.
"""

import numpy as np


class TestSimilarityDetector:
    """Test embedding-based similarity detection."""

    def test_import(self):
        """Test that the module can be imported."""
        from reasonforge.embeddings.similarity import SimilarityDetector

        detector = SimilarityDetector.__new__(SimilarityDetector)
        assert detector is not None

    def test_cosine_similarity_identical(self):
        from reasonforge.embeddings.similarity import SimilarityDetector

        detector = SimilarityDetector.__new__(SimilarityDetector)
        # Identical normalized vectors should have similarity 1.0
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        sim = detector.cosine_similarity(a, b)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        from reasonforge.embeddings.similarity import SimilarityDetector

        detector = SimilarityDetector.__new__(SimilarityDetector)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = detector.cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_cosine_similarity_opposite(self):
        from reasonforge.embeddings.similarity import SimilarityDetector

        detector = SimilarityDetector.__new__(SimilarityDetector)
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        sim = detector.cosine_similarity(a, b)
        assert abs(sim - (-1.0)) < 1e-6

    def test_empty_history_check(self):
        from reasonforge.embeddings.similarity import SimilarityDetector

        detector = SimilarityDetector.__new__(SimilarityDetector)
        detector.history_embeddings = []

        class FakeResponse:
            reasoning_steps = [{"reasoning": "test"}]
            final_answer = "test"

        sim = detector.check_against_history(FakeResponse())
        assert sim == 0.0

    def test_empty_batch_check(self):
        from reasonforge.embeddings.similarity import SimilarityDetector

        detector = SimilarityDetector.__new__(SimilarityDetector)

        class FakeResponse:
            reasoning_steps = [{"reasoning": "test"}]
            final_answer = "test"

        sim = detector.check_against_batch(FakeResponse(), [])
        assert sim == 0.0
