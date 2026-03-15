"""Witness generator for the ReasonForge ZK certificate circuit.

Converts a VerificationCertificate (or raw report data) into a JSON
witness file compatible with the ``certificate.circom`` circuit.  The
witness includes both public and private inputs.

The circuit expects:
  - task_hash[8]: 256-bit hash split into 8 x 32-bit limbs (big-endian)
  - overall_verdict: 0=FAILED, 1=PARTIALLY_VERIFIED, 2=VERIFIED
  - total_steps, verified_steps, timestamp: integers
  - step_verdicts[20]: array of 0/1 values, padded with zeros
  - Helper witness values for non-zero checks (inverses)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Maximum number of steps supported by the circuit.
MAX_STEPS: int = 20

# Verdict string to integer mapping.
VERDICT_MAP: Dict[str, int] = {
    "FAILED": 0,
    "PARTIALLY_VERIFIED": 1,
    "VERIFIED": 2,
}

# BN128 field prime for modular inverse computation.
_BN128_PRIME: int = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


def _mod_inverse(value: int, prime: int = _BN128_PRIME) -> int:
    """Compute the modular multiplicative inverse of *value* mod *prime*.

    Returns 0 if *value* is 0 (by convention for the circuit witness).
    """
    if value == 0:
        return 0
    return pow(value, prime - 2, prime)


def _hash_to_limbs(
    hex_hash: str, num_limbs: int = 8, bits_per_limb: int = 32
) -> List[int]:
    """Split a hex-encoded hash into *num_limbs* integer limbs.

    The hash is interpreted as a big-endian integer and split from
    least-significant to most-significant limb.  If the hash string
    is shorter than expected, it is left-padded with zeros.
    """
    # Normalise: strip 0x prefix, pad to 64 hex chars (256 bits)
    clean = hex_hash.lower().replace("0x", "").strip()
    clean = clean.zfill(num_limbs * (bits_per_limb // 4))

    # Take only the last 64 hex chars if longer
    clean = clean[-(num_limbs * (bits_per_limb // 4)) :]

    full_int = int(clean, 16)
    mask = (1 << bits_per_limb) - 1
    limbs: List[int] = []
    for _ in range(num_limbs):
        limbs.append(full_int & mask)
        full_int >>= bits_per_limb
    # limbs[0] is the least-significant 32 bits
    return limbs


def generate_witness_json(
    task_hash: str,
    overall_verdict: Union[str, int],
    total_steps: int,
    verified_steps: int,
    timestamp: int,
    step_verdicts: List[int],
) -> Dict[str, Any]:
    """Build the full witness dictionary for the circom circuit.

    Parameters
    ----------
    task_hash:
        Hex-encoded 256-bit hash (e.g. SHA-256 hex digest).
    overall_verdict:
        Either a string (``"VERIFIED"``, ``"PARTIALLY_VERIFIED"``,
        ``"FAILED"``) or an integer (2, 1, 0).
    total_steps:
        Total number of verification steps.
    verified_steps:
        Number of steps that passed verification.
    timestamp:
        Unix epoch timestamp.
    step_verdicts:
        List of 0/1 values indicating per-step pass/fail.
        Length must be ``<= MAX_STEPS``.  Will be zero-padded.

    Returns
    -------
    dict
        A JSON-serialisable dictionary suitable for writing to the
        circuit's input JSON file.
    """
    if len(step_verdicts) > MAX_STEPS:
        raise ValueError(
            f"step_verdicts length {len(step_verdicts)} exceeds MAX_STEPS={MAX_STEPS}"
        )

    # Resolve verdict to integer
    if isinstance(overall_verdict, str):
        verdict_int = VERDICT_MAP.get(overall_verdict)
        if verdict_int is None:
            raise ValueError(
                f"Unknown verdict string: {overall_verdict!r}. "
                f"Expected one of {list(VERDICT_MAP.keys())}"
            )
    else:
        verdict_int = int(overall_verdict)

    # Pad step_verdicts to MAX_STEPS
    padded_verdicts = list(step_verdicts) + [0] * (MAX_STEPS - len(step_verdicts))

    # Validate binary values
    for i, v in enumerate(padded_verdicts):
        if v not in (0, 1):
            raise ValueError(f"step_verdicts[{i}] = {v}, expected 0 or 1")

    # Split task_hash into 8 x 32-bit limbs
    task_hash_limbs = _hash_to_limbs(task_hash)

    # Compute helper witness values for non-zero checks
    is_total_positive = 1 if total_steps > 0 else 0
    total_steps_inv = _mod_inverse(total_steps % _BN128_PRIME)

    is_verified_positive = 1 if verified_steps > 0 else 0
    verified_steps_inv = _mod_inverse(verified_steps % _BN128_PRIME)

    diff = total_steps - verified_steps
    # In the finite field, diff might be negative conceptually but
    # circom works in the BN128 field, so we take diff mod prime.
    diff_field = diff % _BN128_PRIME
    is_diff_zero = 1 if diff == 0 else 0
    diff_inv = _mod_inverse(diff_field)

    witness: Dict[str, Any] = {
        # Public inputs
        "task_hash": [str(limb) for limb in task_hash_limbs],
        "overall_verdict": str(verdict_int),
        "total_steps": str(total_steps),
        "verified_steps": str(verified_steps),
        "timestamp": str(timestamp),
        # Private inputs
        "step_verdicts": [str(v) for v in padded_verdicts],
        # Helper witness signals
        "is_total_positive": str(is_total_positive),
        "total_steps_inv": str(total_steps_inv),
        "is_verified_positive": str(is_verified_positive),
        "verified_steps_inv": str(verified_steps_inv),
        "is_diff_zero": str(is_diff_zero),
        "diff_inv": str(diff_inv),
    }

    logger.debug(
        "Generated witness: verdict=%d total=%d verified=%d",
        verdict_int,
        total_steps,
        verified_steps,
    )
    return witness


def write_witness_file(
    witness: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Write the witness dictionary to a JSON file.

    Parameters
    ----------
    witness:
        The witness dictionary (from :func:`generate_witness_json`).
    output_path:
        Destination path.  Defaults to ``input.json`` in the current
        working directory.

    Returns
    -------
    Path
        The path to the written file.
    """
    if output_path is None:
        out = Path("input.json")
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(witness, indent=2), encoding="utf-8")
    logger.info("Witness file written to %s", out)
    return out
