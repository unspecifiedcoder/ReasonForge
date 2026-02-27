"""
ReasonForge - Embedding-Based Similarity Detection

Replaces MVP Jaccard similarity with sentence-transformer cosine similarity.
"""

from .similarity import SimilarityDetector

__all__ = ["SimilarityDetector"]
