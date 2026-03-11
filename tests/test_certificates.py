"""
ReasonForge - Certificate Module Tests

Tests for VerificationCertificate (schema), CertificateProver, and
CertificateRegistry.
"""

from __future__ import annotations

import hashlib

import pytest

from reasonforge.certificates.prover import CertificateProver
from reasonforge.certificates.registry import CertificateRegistry
from reasonforge.certificates.schema import VerificationCertificate
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
        assert cert.zk_proof == b"STUB_ZK_PROOF"
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
