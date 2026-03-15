"""ZK proof verification wrapper for the ReasonForge certificate circuit.

Wraps ``snarkjs`` CLI calls to verify Groth16 proofs.  Falls back
gracefully when ``snarkjs`` is not installed.

Both synchronous (:func:`verify_proof_sync`) and asynchronous
(:func:`verify_proof`) interfaces are provided.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Default verification key path.
_BUILD_DIR = Path(__file__).resolve().parent / "circuits" / "build"
_VERIFICATION_KEY = _BUILD_DIR / "verification_key.json"


def _snarkjs_available() -> bool:
    """Return ``True`` if snarkjs is on the system PATH."""
    return shutil.which("snarkjs") is not None


def verify_proof_sync(
    proof_bytes: bytes,
    verification_key: Optional[str] = None,
    verification_key_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Verify a Groth16 ZK proof synchronously.

    Parameters
    ----------
    proof_bytes:
        JSON-encoded proof payload (as produced by
        :func:`~reasonforge.certificates.zk_prover.generate_proof_sync`).
        Must contain ``"proof"`` and ``"public_signals"`` keys.
    verification_key:
        JSON string of the verification key.  If not provided,
        *verification_key_path* is used instead.
    verification_key_path:
        Path to a verification key JSON file.  Defaults to the
        built-in build directory key.

    Returns
    -------
    bool
        ``True`` if the proof is valid, ``False`` otherwise.

    Raises
    ------
    RuntimeError
        If snarkjs is not installed.
    """
    if not _snarkjs_available():
        raise RuntimeError(
            "snarkjs is not installed or not on PATH. "
            "Install with: npm install -g snarkjs"
        )

    # Parse the proof payload
    try:
        payload: Dict[str, Any] = json.loads(proof_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Failed to parse proof payload: %s", exc)
        return False

    proof_data: Optional[Dict[str, Any]] = payload.get("proof")
    public_signals: Optional[List[str]] = payload.get("public_signals")

    if proof_data is None or public_signals is None:
        logger.warning("Proof payload missing 'proof' or 'public_signals' keys")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        proof_json = tmp / "proof.json"
        public_json = tmp / "public.json"
        vk_json = tmp / "verification_key.json"

        # Write proof and public signals
        proof_json.write_text(json.dumps(proof_data, indent=2), encoding="utf-8")
        public_json.write_text(json.dumps(public_signals, indent=2), encoding="utf-8")

        # Write verification key
        if verification_key:
            vk_json.write_text(verification_key, encoding="utf-8")
        elif verification_key_path:
            vk_path = Path(verification_key_path)
            if not vk_path.is_file():
                logger.warning("Verification key not found at %s", vk_path)
                return False
            vk_json.write_text(vk_path.read_text(encoding="utf-8"), encoding="utf-8")
        elif _VERIFICATION_KEY.is_file():
            vk_json.write_text(
                _VERIFICATION_KEY.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            logger.warning(
                "No verification key provided and default key not found at %s",
                _VERIFICATION_KEY,
            )
            return False

        # Run snarkjs verify
        try:
            result = subprocess.run(
                [
                    "snarkjs",
                    "groth16",
                    "verify",
                    str(vk_json),
                    str(public_json),
                    str(proof_json),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(tmp),
            )
        except subprocess.TimeoutExpired:
            logger.warning("snarkjs verification timed out")
            return False
        except FileNotFoundError:
            logger.warning("snarkjs binary not found")
            return False

        # snarkjs exits 0 and prints "OK!" on valid proof
        stdout = result.stdout.strip()
        logger.debug(
            "snarkjs verify stdout=%r stderr=%r returncode=%d",
            stdout,
            result.stderr.strip(),
            result.returncode,
        )

        if result.returncode == 0 and "OK" in stdout.upper():
            logger.info("ZK proof verification: VALID")
            return True
        else:
            logger.info(
                "ZK proof verification: INVALID (rc=%d, out=%r)",
                result.returncode,
                stdout,
            )
            return False


async def verify_proof(
    proof_bytes: bytes,
    verification_key: Optional[str] = None,
    verification_key_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Async wrapper around :func:`verify_proof_sync`.

    Runs the synchronous verification in a thread pool executor.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        verify_proof_sync,
        proof_bytes,
        verification_key,
        verification_key_path,
    )


def is_verifier_available() -> bool:
    """Return ``True`` if snarkjs is available for verification."""
    return _snarkjs_available()
