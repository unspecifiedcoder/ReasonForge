"""
ReasonForge - Fact Checker

Citation and factual claim verification for scientific and ethical domains.
"""

from __future__ import annotations

import logging
import re
from typing import Dict

logger = logging.getLogger("reasonforge.verification.fact")


class FactChecker:
    """Verify citations and factual claims in miner submissions."""

    # Known facts database (expandable)
    KNOWN_FACTS: Dict[str, str] = {
        "speed of light": "299792458 m/s",
        "planck constant": "6.626e-34 J*s",
        "avogadro number": "6.022e23",
        "boltzmann constant": "1.381e-23 J/K",
        "gravitational constant": "6.674e-11 N*m^2/kg^2",
        "electron mass": "9.109e-31 kg",
        "proton mass": "1.673e-27 kg",
        "pi": "3.14159265358979",
        "euler number": "2.71828182845905",
        "golden ratio": "1.61803398874989",
        "ph of pure water": "7.0",
        "absolute zero": "-273.15 C",
        "boiling point of water": "100 C",
    }

    def verify_claims(self, text: str) -> float:
        """
        Verify factual claims in text.

        Returns:
            Score 0.0-1.0 based on claim accuracy.
        """
        if not text:
            return 0.5

        claims_found = 0
        claims_verified = 0

        text_lower = text.lower()

        for fact_key, fact_value in self.KNOWN_FACTS.items():
            if fact_key in text_lower:
                claims_found += 1
                # Check if the correct value appears near the fact mention
                if self._value_matches(text_lower, fact_key, fact_value):
                    claims_verified += 1

        if claims_found == 0:
            return 0.5  # No verifiable claims

        return claims_verified / claims_found

    def check_citations(self, text: str) -> float:
        """
        Check for presence and format of citations.

        Returns:
            Score 0.0-1.0 based on citation quality.
        """
        if not text:
            return 0.0

        score = 0.0

        # Check for standard citation formats
        # APA-like: (Author, Year)
        apa_citations = re.findall(
            r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?,\s*\d{4}\)", text
        )
        if apa_citations:
            score += min(0.4, len(apa_citations) * 0.1)

        # Numbered citations: [1], [2], etc.
        numbered_citations = re.findall(r"\[\d+\]", text)
        if numbered_citations:
            score += min(0.3, len(numbered_citations) * 0.05)

        # General reference indicators
        ref_indicators = [
            "according to",
            "study shows",
            "research indicates",
            "et al.",
            "published in",
            "journal of",
        ]
        for indicator in ref_indicators:
            if indicator.lower() in text.lower():
                score += 0.05

        return min(1.0, score)

    def verify_scientific_claims(self, domain: str, text: str) -> float:
        """
        Domain-specific scientific claim verification.
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        score = 0.3  # Base score for having content

        # Check for methodological rigor
        methodology_keywords = [
            "hypothesis",
            "method",
            "result",
            "conclusion",
            "control",
            "variable",
            "experiment",
            "observation",
            "data",
            "analysis",
            "significant",
            "evidence",
        ]
        found = sum(1 for kw in methodology_keywords if kw in text_lower)
        score += min(0.4, found * 0.05)

        # Check for quantitative content
        numbers = re.findall(r"\d+\.?\d*", text)
        if numbers:
            score += min(0.2, len(numbers) * 0.02)

        # Check for units
        units = [
            "kg",
            "m/s",
            "mol",
            "kelvin",
            "joule",
            "watt",
            "newton",
            "pascal",
            "hertz",
            "volt",
            "ampere",
        ]
        for unit in units:
            if unit in text_lower:
                score += 0.03

        return min(1.0, score)

    def _value_matches(self, text: str, fact_key: str, fact_value: str) -> bool:
        """Check if the correct value appears near a fact mention."""
        # Find the position of the fact mention
        pos = text.find(fact_key)
        if pos < 0:
            return False

        # Check a window around the mention
        window_start = max(0, pos - 200)
        window_end = min(len(text), pos + len(fact_key) + 200)
        window = text[window_start:window_end]

        # Extract the numeric part of the expected value
        expected_nums = re.findall(r"-?\d+\.?\d*(?:e[+-]?\d+)?", fact_value.lower())
        for expected in expected_nums:
            if expected in window:
                return True

        return False
