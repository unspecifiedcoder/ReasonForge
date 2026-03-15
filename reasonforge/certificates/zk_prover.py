"""ZK proof generation wrapper for the ReasonForge certificate circuit.

Wraps ``snarkjs`` CLI calls to generate Groth16 proofs from a witness
JSON.  Falls back gracefully to stub mode when ``snarkjs`` or
``circom`` are not installed.

Both synchronous (:func:`generate_proof_sync`) and asynchronous
(:func:`generate_proof`) interfaces are provided.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .witness import generate_witness_json

logger = logging.getLogger(__name__)

# On Windows, npm-installed tools (.cmd wrappers) require shell=True.
_USE_SHELL: bool = sys.platform == "win32"

# Default paths for build artefacts produced by ``setup.py``.
_CIRCUITS_DIR = Path(__file__).resolve().parent / "circuits"
_BUILD_DIR = _CIRCUITS_DIR / "build"
_WASM_DIR = _BUILD_DIR / "certificate_js"
_PROVING_KEY = _BUILD_DIR / "certificate_proving_key.zkey"
_VERIFICATION_KEY = _BUILD_DIR / "verification_key.json"


def _snarkjs_available() -> bool:
    """Return ``True`` if snarkjs is on the system PATH."""
    return shutil.which("snarkjs") is not None


def _build_artefacts_exist() -> bool:
    """Return ``True`` if all required build artefacts are present."""
    wasm_file = _WASM_DIR / "certificate.wasm"
    return (
        wasm_file.is_file() and _PROVING_KEY.is_file() and _VERIFICATION_KEY.is_file()
    )


def _check_prerequisites() -> bool:
    """Return ``True`` if ZK proving is fully available."""
    if not _snarkjs_available():
        logger.debug("snarkjs not found on PATH")
        return False
    if not _build_artefacts_exist():
        logger.debug(
            "Build artefacts not found in %s. Run "
            "'python -m reasonforge.certificates.setup' first.",
            _BUILD_DIR,
        )
        return False
    return True


def _run_snarkjs(
    args: List[str], cwd: Optional[Path] = None
) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    """Run snarkjs with the given arguments."""
    cmd = ["snarkjs"] + args
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=str(cwd) if cwd else None,
        timeout=120,
        shell=_USE_SHELL,
    )


def generate_proof_sync(
    task_hash: str,
    overall_verdict: Union[str, int],
    total_steps: int,
    verified_steps: int,
    timestamp: int,
    step_verdicts: List[int],
    proving_key_path: Optional[Union[str, Path]] = None,
    verification_key_path: Optional[Union[str, Path]] = None,
) -> Tuple[bytes, str]:
    """Generate a Groth16 ZK proof synchronously.

    Parameters
    ----------
    task_hash:
        Hex-encoded 256-bit hash.
    overall_verdict:
        Verdict string or integer.
    total_steps, verified_steps, timestamp:
        Certificate summary fields.
    step_verdicts:
        List of 0/1 per-step results (length <= MAX_STEPS).
    proving_key_path:
        Override path to the ``.zkey`` file.
    verification_key_path:
        Override path to the verification key JSON.

    Returns
    -------
    tuple[bytes, str]
        ``(proof_json_bytes, verification_key_json_string)``.

    Raises
    ------
    RuntimeError
        If snarkjs or build artefacts are not available.
    """
    pkey = Path(proving_key_path) if proving_key_path else _PROVING_KEY
    vkey = Path(verification_key_path) if verification_key_path else _VERIFICATION_KEY

    if not _snarkjs_available():
        raise RuntimeError(
            "snarkjs is not installed or not on PATH. "
            "Install with: npm install -g snarkjs"
        )
    if not pkey.is_file():
        raise RuntimeError(
            f"Proving key not found at {pkey}. "
            "Run 'python -m reasonforge.certificates.setup' first."
        )

    wasm_file = _WASM_DIR / "certificate.wasm"
    if not wasm_file.is_file():
        raise RuntimeError(
            f"Circuit WASM not found at {wasm_file}. "
            "Run 'python -m reasonforge.certificates.setup' first."
        )

    # Generate witness JSON
    witness = generate_witness_json(
        task_hash=task_hash,
        overall_verdict=overall_verdict,
        total_steps=total_steps,
        verified_steps=verified_steps,
        timestamp=timestamp,
        step_verdicts=step_verdicts,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_json = tmp / "input.json"
        witness_file = tmp / "witness.wtns"
        proof_json = tmp / "proof.json"
        public_json = tmp / "public.json"

        # Write input JSON
        input_json.write_text(json.dumps(witness, indent=2), encoding="utf-8")

        # Step 1: Calculate witness using snarkjs wtns calculate
        _run_snarkjs(
            [
                "wtns",
                "calculate",
                str(wasm_file),
                str(input_json),
                str(witness_file),
            ],
            cwd=tmp,
        )

        # Step 2: Generate Groth16 proof
        _run_snarkjs(
            [
                "groth16",
                "prove",
                str(pkey),
                str(witness_file),
                str(proof_json),
                str(public_json),
            ],
            cwd=tmp,
        )

        # Read outputs
        proof_data: Dict[str, Any] = json.loads(proof_json.read_text(encoding="utf-8"))
        public_signals: List[str] = json.loads(public_json.read_text(encoding="utf-8"))

    # Combine proof and public signals into a single JSON payload
    proof_payload: Dict[str, Any] = {
        "proof": proof_data,
        "public_signals": public_signals,
    }
    proof_bytes = json.dumps(proof_payload).encode("utf-8")

    # Read verification key
    vk_json = vkey.read_text(encoding="utf-8") if vkey.is_file() else ""

    logger.info(
        "ZK proof generated: %d bytes, %d public signals",
        len(proof_bytes),
        len(public_signals),
    )
    return proof_bytes, vk_json


async def generate_proof(
    task_hash: str,
    overall_verdict: Union[str, int],
    total_steps: int,
    verified_steps: int,
    timestamp: int,
    step_verdicts: List[int],
    proving_key_path: Optional[Union[str, Path]] = None,
    verification_key_path: Optional[Union[str, Path]] = None,
) -> Tuple[bytes, str]:
    """Async wrapper around :func:`generate_proof_sync`.

    Runs the synchronous proof generation in a thread pool executor
    to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        generate_proof_sync,
        task_hash,
        overall_verdict,
        total_steps,
        verified_steps,
        timestamp,
        step_verdicts,
        proving_key_path,
        verification_key_path,
    )


def is_zk_available() -> bool:
    """Return ``True`` if the full ZK proving pipeline is available.

    This checks for:
    1. ``snarkjs`` on the system PATH
    2. All required build artefacts (WASM, proving key, verification key)
    """
    return _check_prerequisites()
