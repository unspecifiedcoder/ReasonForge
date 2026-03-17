"""
ReasonForge - Certificate Module Tests

Tests for VerificationCertificate (schema), CertificateProver,
CertificateRegistry, and the ZK proof generation / verification pipeline.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess as _subprocess
import sys as _sys
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from reasonforge.certificates.prover import CertificateProver
from reasonforge.certificates.registry import CertificateRegistry
from reasonforge.certificates.schema import VerificationCertificate
from reasonforge.certificates.witness import (
    MAX_STEPS,
    VERDICT_MAP,
    _hash_to_limbs,
    _mod_inverse,
    generate_witness_json,
    write_witness_file,
)
from reasonforge.verification.verdict import StepVerdict


# ──────────────────────────────────────────────
# VerificationCertificate — creation & defaults
# ──────────────────────────────────────────────


class TestVerificationCertificateDefaults:
    def test_default_values(self) -> None:
        """A freshly created certificate has sensible defaults."""
        cert = VerificationCertificate()
        assert cert.certificate_id == ""
        assert cert.version == 1
        assert cert.task_hash == ""
        assert cert.domain == "mathematics"
        assert cert.proof_level == "standard"
        assert cert.total_steps == 0
        assert cert.verified_steps == 0
        assert cert.overall_verdict == "FAILED"
        assert cert.timestamp == 0
        assert cert.validator_count == 0
        assert cert.validator_threshold == 3
        assert cert.zk_proof == b""
        assert cert.tx_hash is None
        assert cert.block_number is None

    def test_custom_values(self) -> None:
        """Fields can be overridden at creation time."""
        cert = VerificationCertificate(
            task_hash="abc123",
            domain="code",
            proof_level="formal",
            total_steps=10,
            verified_steps=8,
            overall_verdict="PARTIALLY_VERIFIED",
            timestamp=1700000000,
            validator_count=5,
        )
        assert cert.task_hash == "abc123"
        assert cert.domain == "code"
        assert cert.proof_level == "formal"
        assert cert.total_steps == 10
        assert cert.verified_steps == 8
        assert cert.overall_verdict == "PARTIALLY_VERIFIED"
        assert cert.timestamp == 1700000000
        assert cert.validator_count == 5


# ──────────────────────────────────────────────
# VerificationCertificate — compute_certificate_id
# ──────────────────────────────────────────────


class TestComputeCertificateId:
    def test_deterministic(self) -> None:
        """Calling compute_certificate_id() twice returns the same hash."""
        cert = VerificationCertificate(
            task_hash="hash1",
            overall_verdict="VERIFIED",
            total_steps=5,
            verified_steps=5,
            timestamp=1700000000,
        )
        id1 = cert.compute_certificate_id()
        id2 = cert.compute_certificate_id()
        assert id1 == id2

    def test_matches_manual_sha256(self) -> None:
        """The ID matches an independently computed SHA-256 digest."""
        cert = VerificationCertificate(
            task_hash="hash1",
            overall_verdict="VERIFIED",
            total_steps=5,
            verified_steps=5,
            timestamp=1700000000,
        )
        payload = "hash1" + "VERIFIED" + "5" + "5" + "1700000000"
        expected = hashlib.sha256(payload.encode()).hexdigest()
        assert cert.compute_certificate_id() == expected

    def test_sets_certificate_id_field(self) -> None:
        """compute_certificate_id() stores the result in certificate_id."""
        cert = VerificationCertificate(
            task_hash="abc",
            overall_verdict="FAILED",
            total_steps=1,
            verified_steps=0,
            timestamp=100,
        )
        result = cert.compute_certificate_id()
        assert cert.certificate_id == result
        assert cert.certificate_id != ""

    def test_changes_when_task_hash_differs(self) -> None:
        """Changing task_hash produces a different certificate ID."""
        cert_a = VerificationCertificate(
            task_hash="aaa", overall_verdict="VERIFIED", timestamp=1
        )
        cert_b = VerificationCertificate(
            task_hash="bbb", overall_verdict="VERIFIED", timestamp=1
        )
        assert cert_a.compute_certificate_id() != cert_b.compute_certificate_id()

    def test_changes_when_verdict_differs(self) -> None:
        """Changing overall_verdict produces a different certificate ID."""
        cert_a = VerificationCertificate(
            task_hash="same", overall_verdict="VERIFIED", timestamp=1
        )
        cert_b = VerificationCertificate(
            task_hash="same", overall_verdict="FAILED", timestamp=1
        )
        assert cert_a.compute_certificate_id() != cert_b.compute_certificate_id()

    def test_changes_when_timestamp_differs(self) -> None:
        """Changing timestamp produces a different certificate ID."""
        cert_a = VerificationCertificate(task_hash="same", timestamp=100)
        cert_b = VerificationCertificate(task_hash="same", timestamp=200)
        assert cert_a.compute_certificate_id() != cert_b.compute_certificate_id()


# ──────────────────────────────────────────────
# VerificationCertificate — to_dict
# ──────────────────────────────────────────────


class TestToDict:
    def test_returns_dict(self) -> None:
        """to_dict() returns a plain dictionary."""
        cert = VerificationCertificate()
        result = cert.to_dict()
        assert isinstance(result, dict)

    def test_contains_all_fields(self) -> None:
        """The dict contains every dataclass field."""
        cert = VerificationCertificate(
            task_hash="t1",
            domain="logic",
            total_steps=3,
            verified_steps=2,
            overall_verdict="PARTIALLY_VERIFIED",
            timestamp=999,
        )
        d = cert.to_dict()
        assert d["task_hash"] == "t1"
        assert d["domain"] == "logic"
        assert d["total_steps"] == 3
        assert d["verified_steps"] == 2
        assert d["overall_verdict"] == "PARTIALLY_VERIFIED"
        assert d["timestamp"] == 999
        assert d["version"] == 1
        assert d["zk_proof"] == b""
        assert d["tx_hash"] is None

    def test_roundtrip_certificate_id(self) -> None:
        """After computing the ID, to_dict() reflects it."""
        cert = VerificationCertificate(task_hash="abc", timestamp=1)
        cert.compute_certificate_id()
        d = cert.to_dict()
        assert d["certificate_id"] == cert.certificate_id
        assert d["certificate_id"] != ""


# ──────────────────────────────────────────────
# CertificateProver
# ──────────────────────────────────────────────


class TestCertificateProver:
    def test_instantiation_default(self) -> None:
        """CertificateProver can be created with default params."""
        prover = CertificateProver()
        assert prover.params_path is None

    def test_instantiation_with_path(self) -> None:
        """CertificateProver stores params_path."""
        prover = CertificateProver(params_path="/tmp/params.bin")
        assert prover.params_path == "/tmp/params.bin"

    @pytest.mark.asyncio
    async def test_generate_certificate_all_verified(self) -> None:
        """All steps verified -> VERIFIED verdict."""
        prover = CertificateProver()
        verdicts = [
            StepVerdict(step_id=0, verified=True),
            StepVerdict(step_id=1, verified=True),
            StepVerdict(step_id=2, verified=True),
        ]
        cert = await prover.generate_certificate(
            task_id="task-001",
            task_hash="abc123",
            domain="mathematics",
            proof_level="standard",
            step_verdicts=verdicts,
            validator_count=3,
        )
        assert cert.overall_verdict == "VERIFIED"
        assert cert.total_steps == 3
        assert cert.verified_steps == 3
        assert cert.task_hash == "abc123"
        assert cert.domain == "mathematics"
        assert cert.proof_level == "standard"
        assert cert.validator_count == 3
        # zk_proof is either real or stub depending on environment
        assert cert.zk_proof != b""
        assert cert.certificate_id != ""

    @pytest.mark.asyncio
    async def test_generate_certificate_partial(self) -> None:
        """Some steps fail -> PARTIALLY_VERIFIED verdict."""
        prover = CertificateProver()
        verdicts = [
            StepVerdict(step_id=0, verified=True),
            StepVerdict(step_id=1, verified=False, error_message="type error"),
            StepVerdict(step_id=2, verified=True),
        ]
        cert = await prover.generate_certificate(
            task_id="task-002",
            task_hash="def456",
            domain="code",
            proof_level="formal",
            step_verdicts=verdicts,
        )
        assert cert.overall_verdict == "PARTIALLY_VERIFIED"
        assert cert.total_steps == 3
        assert cert.verified_steps == 2

    @pytest.mark.asyncio
    async def test_generate_certificate_all_failed(self) -> None:
        """No steps verified -> FAILED verdict."""
        prover = CertificateProver()
        verdicts = [
            StepVerdict(step_id=0, verified=False),
            StepVerdict(step_id=1, verified=False),
        ]
        cert = await prover.generate_certificate(
            task_id="task-003",
            task_hash="ghi789",
            domain="logic",
            proof_level="quick",
            step_verdicts=verdicts,
        )
        assert cert.overall_verdict == "FAILED"
        assert cert.total_steps == 2
        assert cert.verified_steps == 0

    @pytest.mark.asyncio
    async def test_generate_certificate_empty_verdicts(self) -> None:
        """Empty verdict list -> FAILED verdict, 0 steps."""
        prover = CertificateProver()
        cert = await prover.generate_certificate(
            task_id="task-004",
            task_hash="jkl000",
            domain="mathematics",
            proof_level="standard",
            step_verdicts=[],
        )
        assert cert.overall_verdict == "FAILED"
        assert cert.total_steps == 0
        assert cert.verified_steps == 0


# ──────────────────────────────────────────────
# CertificateRegistry
# ──────────────────────────────────────────────


class TestCertificateRegistry:
    def _make_cert(
        self,
        task_hash: str = "test",
        overall_verdict: str = "FAILED",
        timestamp: int = 0,
        zk_proof: bytes = b"",
    ) -> VerificationCertificate:
        """Helper to create a certificate with a computed ID."""
        cert = VerificationCertificate(
            task_hash=task_hash,
            overall_verdict=overall_verdict,
            timestamp=timestamp,
            zk_proof=zk_proof,
        )
        cert.compute_certificate_id()
        return cert

    def test_register_returns_id(self) -> None:
        """register() returns the certificate ID."""
        registry = CertificateRegistry()
        cert = self._make_cert(task_hash="reg1", timestamp=1)
        cert_id = registry.register(cert)
        assert cert_id == cert.certificate_id
        assert cert_id != ""

    def test_register_auto_computes_id(self) -> None:
        """register() computes ID if not already set."""
        registry = CertificateRegistry()
        cert = VerificationCertificate(task_hash="auto", timestamp=42)
        assert cert.certificate_id == ""
        cert_id = registry.register(cert)
        assert cert_id != ""
        assert cert.certificate_id == cert_id

    def test_get_existing(self) -> None:
        """get() returns a previously registered certificate."""
        registry = CertificateRegistry()
        cert = self._make_cert(task_hash="get1", timestamp=10)
        cert_id = registry.register(cert)
        retrieved = registry.get(cert_id)
        assert retrieved is cert

    def test_get_nonexistent_returns_none(self) -> None:
        """get() returns None for an unknown ID."""
        registry = CertificateRegistry()
        assert registry.get("nonexistent-id") is None

    def test_verify_with_proof(self) -> None:
        """verify() returns True for a certificate with zk_proof."""
        registry = CertificateRegistry()
        cert = self._make_cert(task_hash="v1", zk_proof=b"STUB_ZK_PROOF")
        cert_id = registry.register(cert)
        assert registry.verify(cert_id) is True

    def test_verify_without_proof(self) -> None:
        """verify() returns False for a certificate with empty zk_proof."""
        registry = CertificateRegistry()
        cert = self._make_cert(task_hash="v2", zk_proof=b"")
        cert_id = registry.register(cert)
        assert registry.verify(cert_id) is False

    def test_verify_nonexistent(self) -> None:
        """verify() returns False for an unknown ID."""
        registry = CertificateRegistry()
        assert registry.verify("does-not-exist") is False

    def test_get_stats_empty(self) -> None:
        """get_stats() on empty registry."""
        registry = CertificateRegistry()
        stats = registry.get_stats()
        assert stats["total_certificates"] == 0
        assert stats["total_verified"] == 0

    def test_get_stats_mixed(self) -> None:
        """get_stats() counts total and verified certificates."""
        registry = CertificateRegistry()
        cert_with_proof = self._make_cert(
            task_hash="s1", timestamp=1, zk_proof=b"proof"
        )
        cert_without_proof = self._make_cert(task_hash="s2", timestamp=2, zk_proof=b"")
        registry.register(cert_with_proof)
        registry.register(cert_without_proof)
        stats = registry.get_stats()
        assert stats["total_certificates"] == 2
        assert stats["total_verified"] == 1

    def test_multiple_registrations(self) -> None:
        """Multiple certificates can be registered and retrieved."""
        registry = CertificateRegistry()
        certs = []
        for i in range(5):
            cert = self._make_cert(task_hash=str(i), timestamp=i)
            registry.register(cert)
            certs.append(cert)
        for cert in certs:
            assert registry.get(cert.certificate_id) is cert
        assert registry.get_stats()["total_certificates"] == 5


# ──────────────────────────────────────────────
# Witness Generator
# ──────────────────────────────────────────────


class TestWitnessGenerator:
    """Tests for the witness generation module."""

    def test_generate_witness_verified(self) -> None:
        """Witness for a fully-VERIFIED certificate."""
        witness = generate_witness_json(
            task_hash="a" * 64,
            overall_verdict="VERIFIED",
            total_steps=3,
            verified_steps=3,
            timestamp=1700000000,
            step_verdicts=[1, 1, 1],
        )
        assert witness["overall_verdict"] == "2"
        assert witness["total_steps"] == "3"
        assert witness["verified_steps"] == "3"
        assert len(witness["step_verdicts"]) == MAX_STEPS
        # First 3 are "1", rest are "0" (padding)
        assert witness["step_verdicts"][:3] == ["1", "1", "1"]
        assert all(v == "0" for v in witness["step_verdicts"][3:])

    def test_generate_witness_partial(self) -> None:
        """Witness for a PARTIALLY_VERIFIED certificate."""
        witness = generate_witness_json(
            task_hash="b" * 64,
            overall_verdict="PARTIALLY_VERIFIED",
            total_steps=5,
            verified_steps=3,
            timestamp=1700000000,
            step_verdicts=[1, 1, 1, 0, 0],
        )
        assert witness["overall_verdict"] == "1"
        assert witness["total_steps"] == "5"
        assert witness["verified_steps"] == "3"
        assert witness["step_verdicts"][:5] == ["1", "1", "1", "0", "0"]

    def test_generate_witness_failed(self) -> None:
        """Witness for a FAILED certificate."""
        witness = generate_witness_json(
            task_hash="c" * 64,
            overall_verdict="FAILED",
            total_steps=2,
            verified_steps=0,
            timestamp=1700000000,
            step_verdicts=[0, 0],
        )
        assert witness["overall_verdict"] == "0"
        assert witness["verified_steps"] == "0"

    def test_generate_witness_empty_steps(self) -> None:
        """Witness for zero steps (FAILED)."""
        witness = generate_witness_json(
            task_hash="d" * 64,
            overall_verdict="FAILED",
            total_steps=0,
            verified_steps=0,
            timestamp=1700000000,
            step_verdicts=[],
        )
        assert witness["total_steps"] == "0"
        assert all(v == "0" for v in witness["step_verdicts"])

    def test_generate_witness_max_steps(self) -> None:
        """Witness with maximum number of steps."""
        verdicts = [1] * MAX_STEPS
        witness = generate_witness_json(
            task_hash="e" * 64,
            overall_verdict="VERIFIED",
            total_steps=MAX_STEPS,
            verified_steps=MAX_STEPS,
            timestamp=1700000000,
            step_verdicts=verdicts,
        )
        assert len(witness["step_verdicts"]) == MAX_STEPS
        assert all(v == "1" for v in witness["step_verdicts"])

    def test_generate_witness_exceeds_max_steps(self) -> None:
        """Exceeding MAX_STEPS raises ValueError."""
        with pytest.raises(ValueError, match="exceeds MAX_STEPS"):
            generate_witness_json(
                task_hash="f" * 64,
                overall_verdict="VERIFIED",
                total_steps=MAX_STEPS + 1,
                verified_steps=MAX_STEPS + 1,
                timestamp=1700000000,
                step_verdicts=[1] * (MAX_STEPS + 1),
            )

    def test_generate_witness_invalid_verdict_string(self) -> None:
        """Invalid verdict string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown verdict string"):
            generate_witness_json(
                task_hash="cc" * 32,
                overall_verdict="INVALID_VERDICT",
                total_steps=1,
                verified_steps=1,
                timestamp=1700000000,
                step_verdicts=[1],
            )

    def test_generate_witness_non_binary_step(self) -> None:
        """Non-binary step verdict raises ValueError."""
        with pytest.raises(ValueError, match="expected 0 or 1"):
            generate_witness_json(
                task_hash="dd" * 32,
                overall_verdict="VERIFIED",
                total_steps=1,
                verified_steps=1,
                timestamp=1700000000,
                step_verdicts=[2],
            )

    def test_generate_witness_integer_verdict(self) -> None:
        """Integer verdict values are accepted."""
        witness = generate_witness_json(
            task_hash="aa" * 32,
            overall_verdict=2,
            total_steps=1,
            verified_steps=1,
            timestamp=1700000000,
            step_verdicts=[1],
        )
        assert witness["overall_verdict"] == "2"

    def test_generate_witness_contains_helper_signals(self) -> None:
        """Witness contains the helper signals needed by the circuit."""
        witness = generate_witness_json(
            task_hash="bb" * 32,
            overall_verdict="VERIFIED",
            total_steps=5,
            verified_steps=5,
            timestamp=1700000000,
            step_verdicts=[1, 1, 1, 1, 1],
        )
        # All helper signals must be present as string values
        assert "is_total_positive" in witness
        assert "total_steps_inv" in witness
        assert "is_verified_positive" in witness
        assert "verified_steps_inv" in witness
        assert "is_diff_zero" in witness
        assert "diff_inv" in witness

    def test_generate_witness_task_hash_limbs(self) -> None:
        """task_hash is correctly split into 8 limbs."""
        witness = generate_witness_json(
            task_hash="00000000" * 7 + "0000000a",  # value = 10
            overall_verdict="FAILED",
            total_steps=0,
            verified_steps=0,
            timestamp=0,
            step_verdicts=[],
        )
        limbs = witness["task_hash"]
        assert len(limbs) == 8
        # Least-significant limb should be 10
        assert limbs[0] == "10"
        # All other limbs should be 0
        assert all(limb == "0" for limb in limbs[1:])

    def test_generate_witness_json_serializable(self) -> None:
        """The witness dictionary is JSON-serializable."""
        witness = generate_witness_json(
            task_hash="ab" * 32,
            overall_verdict="VERIFIED",
            total_steps=2,
            verified_steps=2,
            timestamp=1700000000,
            step_verdicts=[1, 1],
        )
        serialized = json.dumps(witness)
        deserialized = json.loads(serialized)
        assert deserialized == witness


# ──────────────────────────────────────────────
# Witness File I/O
# ──────────────────────────────────────────────


class TestWitnessFileIO:
    def test_write_witness_file(self, tmp_path: Path) -> None:
        """write_witness_file creates a valid JSON file."""
        witness = generate_witness_json(
            task_hash="ab" * 32,
            overall_verdict="VERIFIED",
            total_steps=1,
            verified_steps=1,
            timestamp=100,
            step_verdicts=[1],
        )
        out = tmp_path / "test_input.json"
        result = write_witness_file(witness, out)
        assert result == out
        assert out.is_file()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == witness

    def test_write_witness_file_creates_parents(self, tmp_path: Path) -> None:
        """write_witness_file creates parent directories."""
        out = tmp_path / "sub" / "dir" / "input.json"
        witness = generate_witness_json(
            task_hash="cd" * 32,
            overall_verdict="FAILED",
            total_steps=0,
            verified_steps=0,
            timestamp=0,
            step_verdicts=[],
        )
        result = write_witness_file(witness, out)
        assert result == out
        assert out.is_file()


# ──────────────────────────────────────────────
# Hash-to-limbs utility
# ──────────────────────────────────────────────


class TestHashToLimbs:
    def test_zero_hash(self) -> None:
        """All-zero hash produces all-zero limbs."""
        limbs = _hash_to_limbs("0" * 64)
        assert limbs == [0] * 8

    def test_known_value(self) -> None:
        """A known hash value is split correctly."""
        # 0x00000001 in the least-significant 32 bits
        hash_hex = "0" * 56 + "00000001"
        limbs = _hash_to_limbs(hash_hex)
        assert limbs[0] == 1
        assert all(limb == 0 for limb in limbs[1:])

    def test_full_value(self) -> None:
        """All-FF hash produces max limbs."""
        limbs = _hash_to_limbs("f" * 64)
        expected = (1 << 32) - 1  # 0xFFFFFFFF
        assert all(limb == expected for limb in limbs)

    def test_short_hash_padded(self) -> None:
        """Short hash strings are left-padded with zeros."""
        limbs = _hash_to_limbs("ff")
        assert limbs[0] == 255
        assert all(limb == 0 for limb in limbs[1:])

    def test_returns_8_limbs(self) -> None:
        """Always returns exactly 8 limbs."""
        limbs = _hash_to_limbs("abc")
        assert len(limbs) == 8


# ──────────────────────────────────────────────
# Modular inverse utility
# ──────────────────────────────────────────────


class TestModInverse:
    def test_inverse_of_zero(self) -> None:
        """Inverse of 0 is 0 by convention."""
        assert _mod_inverse(0) == 0

    def test_inverse_of_one(self) -> None:
        """Inverse of 1 is 1."""
        assert _mod_inverse(1) == 1

    def test_inverse_product_is_one(self) -> None:
        """a * inv(a) == 1 mod p for non-zero a."""
        from reasonforge.certificates.witness import _BN128_PRIME

        for val in [2, 3, 5, 7, 100, 12345, 999999]:
            inv = _mod_inverse(val)
            assert (val * inv) % _BN128_PRIME == 1


# ──────────────────────────────────────────────
# Verdict map
# ──────────────────────────────────────────────


class TestVerdictMap:
    def test_all_verdicts_mapped(self) -> None:
        """All three verdict strings are in the map."""
        assert "FAILED" in VERDICT_MAP
        assert "PARTIALLY_VERIFIED" in VERDICT_MAP
        assert "VERIFIED" in VERDICT_MAP

    def test_verdict_values(self) -> None:
        """Verdict integers match the specification."""
        assert VERDICT_MAP["FAILED"] == 0
        assert VERDICT_MAP["PARTIALLY_VERIFIED"] == 1
        assert VERDICT_MAP["VERIFIED"] == 2


# ──────────────────────────────────────────────
# ZK Prover — availability and graceful fallback
# ──────────────────────────────────────────────


class TestZKProverAvailability:
    """Tests for the ZK prover module's availability checks."""

    def test_import_zk_prover(self) -> None:
        """zk_prover module can be imported without error."""
        from reasonforge.certificates import zk_prover

        assert hasattr(zk_prover, "generate_proof")
        assert hasattr(zk_prover, "generate_proof_sync")
        assert hasattr(zk_prover, "is_zk_available")

    def test_import_zk_verifier(self) -> None:
        """zk_verifier module can be imported without error."""
        from reasonforge.certificates import zk_verifier

        assert hasattr(zk_verifier, "verify_proof")
        assert hasattr(zk_verifier, "verify_proof_sync")
        assert hasattr(zk_verifier, "is_verifier_available")

    def test_is_zk_available_returns_bool(self) -> None:
        """is_zk_available() returns a boolean."""
        from reasonforge.certificates.zk_prover import is_zk_available

        result = is_zk_available()
        assert isinstance(result, bool)

    def test_is_verifier_available_returns_bool(self) -> None:
        """is_verifier_available() returns a boolean."""
        from reasonforge.certificates.zk_verifier import is_verifier_available

        result = is_verifier_available()
        assert isinstance(result, bool)


class TestZKProverFallback:
    """Tests for the graceful fallback behaviour when snarkjs is unavailable."""

    @pytest.mark.asyncio
    async def test_prover_stub_fallback(self) -> None:
        """When ZK is unavailable, certificates get a stub proof."""
        # Patch ZK_AVAILABLE in the prover module to force stub mode
        import reasonforge.certificates.prover as prover_mod

        original = prover_mod.ZK_AVAILABLE
        try:
            prover_mod.ZK_AVAILABLE = False
            prover = CertificateProver()
            cert = await prover.generate_certificate(
                task_id="fallback-test",
                task_hash="abc123",
                domain="mathematics",
                proof_level="standard",
                step_verdicts=[StepVerdict(step_id=0, verified=True)],
            )
            assert cert.zk_proof == b"STUB_ZK_PROOF"
            assert cert.overall_verdict == "VERIFIED"
        finally:
            prover_mod.ZK_AVAILABLE = original

    def test_registry_verify_stub_proof(self) -> None:
        """Registry verify() accepts stub proofs when verifier unavailable."""
        registry = CertificateRegistry()
        cert = VerificationCertificate(
            task_hash="stub-test",
            overall_verdict="VERIFIED",
            total_steps=1,
            verified_steps=1,
            timestamp=100,
            zk_proof=b"STUB_ZK_PROOF",
        )
        cert.compute_certificate_id()
        registry.register(cert)
        assert registry.verify(cert.certificate_id) is True

    def test_prover_sync_raises_without_snarkjs(self) -> None:
        """generate_proof_sync raises RuntimeError without snarkjs."""
        from reasonforge.certificates.zk_prover import generate_proof_sync

        # Mock shutil.which to return None for snarkjs
        with patch(
            "reasonforge.certificates.zk_prover.shutil.which", return_value=None
        ):
            with pytest.raises(RuntimeError, match="snarkjs is not installed"):
                generate_proof_sync(
                    task_hash="aa" * 32,
                    overall_verdict="VERIFIED",
                    total_steps=1,
                    verified_steps=1,
                    timestamp=100,
                    step_verdicts=[1],
                )

    def test_verifier_sync_raises_without_snarkjs(self) -> None:
        """verify_proof_sync raises RuntimeError without snarkjs."""
        from reasonforge.certificates.zk_verifier import verify_proof_sync

        proof_payload = json.dumps({"proof": {}, "public_signals": []}).encode("utf-8")

        with patch(
            "reasonforge.certificates.zk_verifier.shutil.which", return_value=None
        ):
            with pytest.raises(RuntimeError, match="snarkjs is not installed"):
                verify_proof_sync(proof_payload)


# ──────────────────────────────────────────────
# ZK Verifier — proof parsing
# ──────────────────────────────────────────────


class TestZKVerifierParsing:
    """Tests for proof payload parsing in the verifier."""

    def test_invalid_json_returns_false(self) -> None:
        """Invalid JSON proof bytes return False (not crash)."""
        from reasonforge.certificates.zk_verifier import verify_proof_sync

        # Must patch snarkjs as available so we get past that check
        with patch(
            "reasonforge.certificates.zk_verifier.shutil.which",
            return_value="/usr/bin/snarkjs",
        ):
            result = verify_proof_sync(b"not valid json")
        assert result is False

    def test_missing_keys_returns_false(self) -> None:
        """Missing proof/public_signals keys return False."""
        from reasonforge.certificates.zk_verifier import verify_proof_sync

        payload = json.dumps({"only_proof": {}}).encode("utf-8")
        with patch(
            "reasonforge.certificates.zk_verifier.shutil.which",
            return_value="/usr/bin/snarkjs",
        ):
            result = verify_proof_sync(payload)
        assert result is False


# ──────────────────────────────────────────────
# Circuit file existence
# ──────────────────────────────────────────────


class TestCircuitFile:
    """Tests that the circom circuit file exists."""

    def test_circuit_file_exists(self) -> None:
        """The certificate.circom file exists in the expected location."""
        circuit_path = (
            Path(__file__).resolve().parent.parent
            / "reasonforge"
            / "certificates"
            / "circuits"
            / "certificate.circom"
        )
        assert circuit_path.is_file(), f"Circuit file not found at {circuit_path}"

    def test_circuit_contains_main_component(self) -> None:
        """The circuit file declares a main component."""
        circuit_path = (
            Path(__file__).resolve().parent.parent
            / "reasonforge"
            / "certificates"
            / "circuits"
            / "certificate.circom"
        )
        content = circuit_path.read_text(encoding="utf-8")
        assert "component main" in content
        assert "CertificateVerifier" in content

    def test_circuit_declares_public_inputs(self) -> None:
        """The circuit declares the expected public inputs."""
        circuit_path = (
            Path(__file__).resolve().parent.parent
            / "reasonforge"
            / "certificates"
            / "circuits"
            / "certificate.circom"
        )
        content = circuit_path.read_text(encoding="utf-8")
        assert "task_hash" in content
        assert "overall_verdict" in content
        assert "total_steps" in content
        assert "verified_steps" in content
        assert "timestamp" in content


# ──────────────────────────────────────────────
# Setup module
# ──────────────────────────────────────────────


class TestSetupModule:
    """Tests for the setup module."""

    def test_setup_module_importable(self) -> None:
        """The setup module can be imported."""
        from reasonforge.certificates import setup

        assert hasattr(setup, "run_setup")
        assert hasattr(setup, "main")

    def test_setup_checks_for_circom(self) -> None:
        """The setup script checks for circom availability."""
        from reasonforge.certificates.setup import _check_tool

        result = _check_tool("circom")
        assert isinstance(result, bool)


# ──────────────────────────────────────────────
# Circuit compilation (skip if circom not installed)
# ──────────────────────────────────────────────


# On Windows, npm-installed tools (.cmd wrappers) require shell=True.
_USE_SHELL: bool = _sys.platform == "win32"


def _tool_works(name: str) -> bool:
    """Return True if a tool can actually be executed (not just found on PATH)."""
    if shutil.which(name) is None:
        return False
    try:
        _subprocess.run(
            [name, "--version"],
            capture_output=True,
            timeout=10,
            shell=_USE_SHELL,
        )
        return True
    except (FileNotFoundError, _subprocess.TimeoutExpired, OSError):
        return False


def _find_circom_v2() -> Optional[str]:
    """Find a working circom v2 compiler (``circom2`` or ``circom``)."""
    for candidate in ("circom", "circom2"):
        if shutil.which(candidate) is None:
            continue
        try:
            result = _subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                shell=_USE_SHELL,
            )
            if "2." in result.stdout:
                return candidate
        except (FileNotFoundError, _subprocess.TimeoutExpired, OSError):
            continue
    return None


_CIRCOM_CMD = _find_circom_v2()
_CIRCOM_INSTALLED = _CIRCOM_CMD is not None
_SNARKJS_INSTALLED = _tool_works("snarkjs")


@pytest.mark.skipif(
    not _CIRCOM_INSTALLED,
    reason="circom v2 compiler not installed",
)
class TestCircuitCompilation:
    """Tests that require circom v2 to be installed."""

    def test_circuit_compiles(self, tmp_path: Path) -> None:
        """The circuit compiles successfully with circom v2."""
        assert _CIRCOM_CMD is not None  # for type checker
        circuit_path = (
            Path(__file__).resolve().parent.parent
            / "reasonforge"
            / "certificates"
            / "circuits"
            / "certificate.circom"
        )
        result = _subprocess.run(
            [
                _CIRCOM_CMD,
                str(circuit_path),
                "--r1cs",
                "--wasm",
                "--sym",
                "-o",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            shell=_USE_SHELL,
        )
        assert result.returncode == 0, (
            f"circom compilation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # Check outputs were created
        assert (tmp_path / "certificate.r1cs").is_file()
        assert (tmp_path / "certificate_js" / "certificate.wasm").is_file()


# ──────────────────────────────────────────────
# Real ZK proof generation & verification (skip if tools not installed)
# ──────────────────────────────────────────────


@pytest.mark.skipif(
    not (_CIRCOM_INSTALLED and _SNARKJS_INSTALLED),
    reason="circom and/or snarkjs not installed",
)
class TestRealZKProof:
    """End-to-end tests for real ZK proof generation and verification.

    These tests require circom and snarkjs to be installed, and will
    run the full trusted setup if build artefacts are not present.
    """

    @pytest.fixture(autouse=True)
    def ensure_build(self) -> None:
        """Ensure circuit build artefacts exist."""
        from reasonforge.certificates.zk_prover import _build_artefacts_exist

        if not _build_artefacts_exist():
            from reasonforge.certificates.setup import run_setup

            run_setup()

    def test_proof_generation_produces_bytes(self) -> None:
        """generate_proof_sync produces non-empty proof bytes."""
        from reasonforge.certificates.zk_prover import generate_proof_sync

        proof_bytes, vk = generate_proof_sync(
            task_hash="a" * 64,
            overall_verdict="VERIFIED",
            total_steps=3,
            verified_steps=3,
            timestamp=1700000000,
            step_verdicts=[1, 1, 1],
        )
        assert len(proof_bytes) > 0
        assert len(vk) > 0

        # Proof should be valid JSON
        payload = json.loads(proof_bytes)
        assert "proof" in payload
        assert "public_signals" in payload

    def test_proof_verification_valid(self) -> None:
        """A valid proof verifies successfully."""
        from reasonforge.certificates.zk_prover import generate_proof_sync
        from reasonforge.certificates.zk_verifier import verify_proof_sync

        proof_bytes, vk = generate_proof_sync(
            task_hash="b" * 64,
            overall_verdict="PARTIALLY_VERIFIED",
            total_steps=5,
            verified_steps=3,
            timestamp=1700000000,
            step_verdicts=[1, 1, 1, 0, 0],
        )
        assert verify_proof_sync(proof_bytes, verification_key=vk) is True

    def test_proof_verification_tampered_signals(self) -> None:
        """A proof with tampered public signals fails verification."""
        from reasonforge.certificates.zk_prover import generate_proof_sync
        from reasonforge.certificates.zk_verifier import verify_proof_sync

        proof_bytes, vk = generate_proof_sync(
            task_hash="c" * 64,
            overall_verdict="VERIFIED",
            total_steps=2,
            verified_steps=2,
            timestamp=1700000000,
            step_verdicts=[1, 1],
        )

        # Tamper with public signals
        payload = json.loads(proof_bytes)
        if payload["public_signals"]:
            # Change the first public signal
            original = payload["public_signals"][0]
            payload["public_signals"][0] = str(int(original) + 1)
        tampered = json.dumps(payload).encode("utf-8")

        assert verify_proof_sync(tampered, verification_key=vk) is False

    def test_proof_for_failed_verdict(self) -> None:
        """Proof generation works for FAILED verdict."""
        from reasonforge.certificates.zk_prover import generate_proof_sync
        from reasonforge.certificates.zk_verifier import verify_proof_sync

        proof_bytes, vk = generate_proof_sync(
            task_hash="d" * 64,
            overall_verdict="FAILED",
            total_steps=3,
            verified_steps=0,
            timestamp=1700000000,
            step_verdicts=[0, 0, 0],
        )
        assert verify_proof_sync(proof_bytes, verification_key=vk) is True

    @pytest.mark.asyncio
    async def test_async_proof_generation(self) -> None:
        """Async proof generation works."""
        from reasonforge.certificates.zk_prover import generate_proof
        from reasonforge.certificates.zk_verifier import verify_proof

        proof_bytes, vk = await generate_proof(
            task_hash="e" * 64,
            overall_verdict="VERIFIED",
            total_steps=1,
            verified_steps=1,
            timestamp=1700000000,
            step_verdicts=[1],
        )
        assert len(proof_bytes) > 0
        assert await verify_proof(proof_bytes, verification_key=vk) is True


# ──────────────────────────────────────────────
# Integration: Pipeline -> Report -> ZK Certificate -> Registry
# ──────────────────────────────────────────────


class TestIntegrationPipeline:
    """Integration tests for the full certificate lifecycle."""

    @pytest.mark.asyncio
    async def test_pipeline_to_certificate_to_registry(self) -> None:
        """Full pipeline: generate verdicts -> certificate -> registry."""
        from reasonforge.certificates.prover import CertificateProver
        from reasonforge.certificates.registry import CertificateRegistry

        prover = CertificateProver()
        registry = CertificateRegistry()

        # Generate a certificate with step verdicts
        verdicts = [StepVerdict(step_id=i, verified=True) for i in range(5)]
        cert = await prover.generate_certificate(
            task_id="integration-test",
            task_hash="ff" * 32,
            domain="mathematics",
            proof_level="standard",
            step_verdicts=verdicts,
            validator_count=3,
        )

        assert cert.overall_verdict == "VERIFIED"
        assert cert.total_steps == 5
        assert cert.verified_steps == 5
        assert cert.zk_proof != b""
        assert cert.certificate_id != ""

        # Register and verify
        cert_id = registry.register(cert)
        assert registry.verify(cert_id) is True

        # Retrieve and check
        retrieved = registry.get(cert_id)
        assert retrieved is cert
        assert retrieved is not None
        assert retrieved.overall_verdict == "VERIFIED"

    def test_codeverify_certifier_integration(self) -> None:
        """CodeVerificationCertifier creates and registers a certificate."""
        from reasonforge.codeverify.certificate import CodeVerificationCertifier
        from reasonforge.codeverify.report import CodeVerificationReport

        report = CodeVerificationReport(
            code_hash="ab" * 32,
            verdict="VERIFIED",
            timestamp="2024-01-01T00:00:00+00:00",
            tests_generated=3,
            tests_passed=3,
        )

        certifier = CodeVerificationCertifier()
        cert_id = certifier.certify_and_register(report)

        assert cert_id != ""
        assert certifier.verify_certificate(cert_id) is True

        cert = certifier.get_certificate(cert_id)
        assert cert is not None
        assert cert.overall_verdict == "VERIFIED"
        assert cert.total_steps == 3
        assert cert.verified_steps == 3
        assert cert.zk_proof != b""

    @pytest.mark.asyncio
    async def test_partial_verification_lifecycle(self) -> None:
        """Partial verification produces correct certificate."""
        prover = CertificateProver()
        registry = CertificateRegistry()

        verdicts = [
            StepVerdict(step_id=0, verified=True),
            StepVerdict(step_id=1, verified=False),
            StepVerdict(step_id=2, verified=True),
            StepVerdict(step_id=3, verified=False),
        ]
        cert = await prover.generate_certificate(
            task_id="partial-test",
            task_hash="cc" * 32,
            domain="code",
            proof_level="standard",
            step_verdicts=verdicts,
        )

        assert cert.overall_verdict == "PARTIALLY_VERIFIED"
        assert cert.verified_steps == 2
        assert cert.total_steps == 4

        cert_id = registry.register(cert)
        assert registry.verify(cert_id) is True

    @pytest.mark.asyncio
    async def test_failed_verification_lifecycle(self) -> None:
        """Failed verification produces correct certificate."""
        prover = CertificateProver()
        registry = CertificateRegistry()

        verdicts = [
            StepVerdict(step_id=0, verified=False),
            StepVerdict(step_id=1, verified=False),
        ]
        cert = await prover.generate_certificate(
            task_id="failed-test",
            task_hash="dd" * 32,
            domain="logic",
            proof_level="quick",
            step_verdicts=verdicts,
        )

        assert cert.overall_verdict == "FAILED"
        assert cert.verified_steps == 0
        assert cert.total_steps == 2

        cert_id = registry.register(cert)
        assert registry.verify(cert_id) is True

    def test_registry_stats_after_multiple_certs(self) -> None:
        """Registry stats are accurate after multiple registrations."""
        registry = CertificateRegistry()

        # Certificate with proof
        cert1 = VerificationCertificate(
            task_hash="stat1",
            overall_verdict="VERIFIED",
            timestamp=1,
            zk_proof=b"STUB_ZK_PROOF",
        )
        cert1.compute_certificate_id()

        # Certificate without proof
        cert2 = VerificationCertificate(
            task_hash="stat2",
            overall_verdict="FAILED",
            timestamp=2,
            zk_proof=b"",
        )
        cert2.compute_certificate_id()

        registry.register(cert1)
        registry.register(cert2)

        stats = registry.get_stats()
        assert stats["total_certificates"] == 2
        assert stats["total_verified"] == 1
