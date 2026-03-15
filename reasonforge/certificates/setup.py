"""One-time trusted setup for the ReasonForge ZK certificate circuit.

This script performs the full Groth16 trusted setup:

1. Generate powers of tau ceremony (BN128, 2^12)
2. Compile the circom circuit to R1CS + WASM
3. Run the Groth16 circuit-specific setup
4. Export the verification key

Usage::

    python -m reasonforge.certificates.setup

All artefacts are written to ``reasonforge/certificates/circuits/build/``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths
_CIRCUITS_DIR = Path(__file__).resolve().parent / "circuits"
_BUILD_DIR = _CIRCUITS_DIR / "build"
_CIRCOM_FILE = _CIRCUITS_DIR / "certificate.circom"


def _check_tool(name: str) -> bool:
    """Return True if *name* is available on the system PATH."""
    return shutil.which(name) is not None


def _run(
    cmd: list[str],
    cwd: Path | None = None,
    description: str = "",
) -> None:
    """Run a subprocess command, logging output."""
    desc = description or " ".join(cmd[:3])
    logger.info("Running: %s", desc)
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        timeout=300,
    )

    if result.stdout.strip():
        logger.debug("stdout: %s", result.stdout.strip())
    if result.stderr.strip():
        logger.debug("stderr: %s", result.stderr.strip())

    if result.returncode != 0:
        logger.error(
            "%s failed (rc=%d):\nstdout: %s\nstderr: %s",
            desc,
            result.returncode,
            result.stdout,
            result.stderr,
        )
        raise RuntimeError(f"{desc} failed with return code {result.returncode}")


def run_setup() -> None:
    """Execute the full trusted setup procedure."""
    # Check prerequisites
    if not _check_tool("circom"):
        print(
            "ERROR: 'circom' compiler not found on PATH.\n"
            "Install from: https://docs.circom.io/getting-started/installation/",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _check_tool("snarkjs"):
        print(
            "ERROR: 'snarkjs' not found on PATH.\nInstall with: npm install -g snarkjs",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _CIRCOM_FILE.is_file():
        print(
            f"ERROR: Circuit file not found at {_CIRCOM_FILE}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create build directory
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ReasonForge ZK Certificate Trusted Setup")
    print("=" * 60)

    # ── Step 1: Powers of Tau ceremony ──────────────────────────
    print("\n[1/6] Generating powers of tau (BN128, 2^12)...")
    pot12_0 = _BUILD_DIR / "pot12_0000.ptau"
    pot12_1 = _BUILD_DIR / "pot12_0001.ptau"
    pot12_final = _BUILD_DIR / "pot12_final.ptau"

    _run(
        [
            "snarkjs",
            "powersoftau",
            "new",
            "bn128",
            "12",
            str(pot12_0),
            "-v",
        ],
        description="Powers of tau: new",
    )

    # Contribute to the ceremony (non-interactive with random entropy)
    print("[2/6] Contributing to powers of tau ceremony...")
    _run(
        [
            "snarkjs",
            "powersoftau",
            "contribute",
            str(pot12_0),
            str(pot12_1),
            "--name=ReasonForge_Setup",
            "-v",
            "-e=reasonforge_random_entropy_for_setup",
        ],
        description="Powers of tau: contribute",
    )

    # Prepare phase 2
    print("[3/6] Preparing phase 2...")
    _run(
        [
            "snarkjs",
            "powersoftau",
            "prepare",
            "phase2",
            str(pot12_1),
            str(pot12_final),
            "-v",
        ],
        description="Powers of tau: prepare phase 2",
    )

    # ── Step 2: Compile the circom circuit ──────────────────────
    print("[4/6] Compiling circom circuit...")
    _run(
        [
            "circom",
            str(_CIRCOM_FILE),
            "--r1cs",
            "--wasm",
            "--sym",
            "-o",
            str(_BUILD_DIR),
        ],
        description="Circom compilation",
    )

    r1cs_file = _BUILD_DIR / "certificate.r1cs"
    wasm_dir = _BUILD_DIR / "certificate_js"

    if not r1cs_file.is_file():
        raise RuntimeError(f"R1CS file not found at {r1cs_file}")
    if not (wasm_dir / "certificate.wasm").is_file():
        raise RuntimeError(f"WASM file not found in {wasm_dir}")

    # Print circuit info
    try:
        _run(
            ["snarkjs", "r1cs", "info", str(r1cs_file)],
            description="R1CS info",
        )
    except RuntimeError:
        pass  # Non-critical

    # ── Step 3: Groth16 setup ───────────────────────────────────
    print("[5/6] Running Groth16 setup...")
    zkey_0 = _BUILD_DIR / "certificate_0000.zkey"
    zkey_final = _BUILD_DIR / "certificate_proving_key.zkey"

    _run(
        [
            "snarkjs",
            "groth16",
            "setup",
            str(r1cs_file),
            str(pot12_final),
            str(zkey_0),
        ],
        description="Groth16 setup",
    )

    # Contribute to the ceremony
    _run(
        [
            "snarkjs",
            "zkey",
            "contribute",
            str(zkey_0),
            str(zkey_final),
            "--name=ReasonForge_Circuit_Setup",
            "-v",
            "-e=reasonforge_circuit_random_entropy",
        ],
        description="Groth16 zkey contribute",
    )

    # ── Step 4: Export verification key ─────────────────────────
    print("[6/6] Exporting verification key...")
    vk_file = _BUILD_DIR / "verification_key.json"

    _run(
        [
            "snarkjs",
            "zkey",
            "export",
            "verificationkey",
            str(zkey_final),
            str(vk_file),
        ],
        description="Export verification key",
    )

    # ── Cleanup intermediate files ──────────────────────────────
    for intermediate in [pot12_0, pot12_1, zkey_0]:
        if intermediate.is_file():
            intermediate.unlink()
            logger.debug("Removed intermediate file: %s", intermediate)

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Trusted setup complete!")
    print("=" * 60)
    print(f"\nBuild artefacts in: {_BUILD_DIR}")
    print(f"  Circuit WASM:     {wasm_dir / 'certificate.wasm'}")
    print(f"  Proving key:      {zkey_final}")
    print(f"  Verification key: {vk_file}")
    print(f"  R1CS:             {r1cs_file}")

    # Verify the key file exists and has content
    if vk_file.is_file() and vk_file.stat().st_size > 0:
        print("\nSetup verified: all artefacts present and non-empty.")
    else:
        print("\nWARNING: Some artefacts may be missing or empty.")


def main() -> None:
    """Entry point for ``python -m reasonforge.certificates.setup``."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        run_setup()
    except RuntimeError as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nSetup interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
