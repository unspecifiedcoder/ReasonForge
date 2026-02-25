"""
ReasonForge - Plagiarism Detection

Simplified similarity detection using Jaccard similarity on reasoning text
and hash-based comparison. In production, this would use embedding cosine similarity.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Set, Tuple

from .types import MinerSubmission, SIMILARITY_THRESHOLD, SIMILARITY_PENALTY


class PlagiarismDetector:
    """
    Detects plagiarism/copying between miner submissions.

    - Maintains a rolling buffer of submission data (last 30 epochs)
    - check() returns max similarity score against history
    - If similarity > 0.95, flag and apply 0.5x penalty
    """

    def __init__(self, max_epochs: int = 30):
        self.max_epochs = max_epochs
        # Rolling buffer: list of (miner_id, hash, token_set) per epoch
        self._history: Deque[List[Tuple[str, str, Set[str]]]] = deque(maxlen=max_epochs)

    def _tokenize(self, submission: MinerSubmission) -> Set[str]:
        """Extract token set from submission reasoning steps."""
        tokens: Set[str] = set()
        for step in submission.steps:
            words = step.reasoning.lower().split()
            # Use 3-grams for better similarity detection
            for i in range(len(words) - 2):
                tokens.add(" ".join(words[i : i + 3]))
            # Also add individual words
            tokens.update(words)
        # Add final answer tokens
        tokens.update(submission.final_answer.lower().split())
        return tokens

    def _jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return intersection / union

    def check(self, submission: MinerSubmission, current_submissions: List[MinerSubmission] = None) -> float:
        """
        Check a submission for plagiarism against history and current epoch.

        Returns: max similarity score (0.0 to 1.0)
        """
        tokens = self._tokenize(submission)
        sub_hash = submission.compute_hash()
        max_sim = 0.0

        # Check against history
        for epoch_data in self._history:
            for other_miner_id, other_hash, other_tokens in epoch_data:
                if other_miner_id == submission.miner_id:
                    continue  # Skip own previous submissions

                # Hash-based exact match
                if sub_hash == other_hash:
                    return 1.0

                # Jaccard similarity
                sim = self._jaccard_similarity(tokens, other_tokens)
                max_sim = max(max_sim, sim)

        # Check against current epoch submissions
        if current_submissions:
            for other in current_submissions:
                if other.miner_id == submission.miner_id:
                    continue
                other_tokens = self._tokenize(other)
                other_hash = other.submission_hash or other.compute_hash()

                if sub_hash == other_hash:
                    return 1.0

                sim = self._jaccard_similarity(tokens, other_tokens)
                max_sim = max(max_sim, sim)

        return max_sim

    def is_plagiarized(self, similarity: float) -> bool:
        """Check if similarity exceeds threshold."""
        return similarity > SIMILARITY_THRESHOLD

    def get_penalty(self, similarity: float) -> float:
        """Get penalty multiplier (1.0 = no penalty, 0.5 = plagiarism penalty)."""
        if self.is_plagiarized(similarity):
            return SIMILARITY_PENALTY
        return 1.0

    def record_epoch(self, submissions: List[MinerSubmission]) -> None:
        """Record submissions from an epoch into the rolling buffer."""
        epoch_data = []
        for sub in submissions:
            tokens = self._tokenize(sub)
            sub_hash = sub.submission_hash or sub.compute_hash()
            epoch_data.append((sub.miner_id, sub_hash, tokens))
        self._history.append(epoch_data)
