# ReasonForge — The Proof Layer for AI

## PLAN_PROOF_LAYER.md

> **What this is**: The plan that transforms ReasonForge from a Bittensor subnet into
> a new category of infrastructure — machine-verifiable trust for AI reasoning.
>
> **Prerequisite**: MVP (PLAN.md) complete. Production subnet (PLAN_PRODUCTION.md) in progress or complete.
>
> **One-line thesis**: Any AI model's reasoning chain in → cryptographic proof of correctness out.
> HTTPS did this for web traffic. We do this for AI thought.
>
> **Build time**: 12-16 weeks for core. 6 months to production-grade.

---

## Table of Contents

```
PART I   — THE PIVOT
  1.  Why We're Narrowing
  2.  What We Kill
  3.  What We Keep
  4.  The New Architecture

PART II  — FORMAL VERIFICATION ENGINE
  5.  NL-to-Formal Translation Pipeline
  6.  Lean 4 Verification Backend
  7.  Code Verification Backend
  8.  First-Order Logic Backend
  9.  Step-Level Process Supervision
  10. Verification Verdicts & Failure Localization

PART III — ZK VERIFICATION CERTIFICATES
  11. Certificate Schema
  12. ZK Circuit Design
  13. Recursive Proof Composition
  14. On-Chain Certificate Registry
  15. Certificate Verification Contract

PART IV  — BITTENSOR INTEGRATION (REVISED)
  16. New Synapse Protocol
  17. Miner Role: Translator
  18. Validator Role: Verifier
  19. Revised Incentive Mechanism
  20. Weight Computation

PART V   — ENTERPRISE API PRODUCT
  21. Verification-as-a-Service API
  22. SDK (Python / TypeScript / Rust)
  23. Model Provider Integrations
  24. Compliance Report Generator

PART VI  — BUILD ORDER & MILESTONES
  25. Directory Structure
  26. Phase-by-Phase Build Order
  27. Success Criteria
  28. Dependency Map
```

---

# PART I — THE PIVOT

---

## 1. Why We're Narrowing

The original ReasonForge design scores AI reasoning across 6 domains using heuristics.
The problem: heuristic scoring is gameable, unverifiable, and no enterprise will pay for it.

The insight: **only 3 types of reasoning can be mechanically proven correct**:

| Domain | Verification Method | Decidable? | Existing Tools |
|--------|---------------------|------------|----------------|
| Mathematics | Formal proof checkers (Lean 4, Coq, Isabelle) | **Yes** | Mature |
| Code | Execution + property-based testing + static analysis | **Yes** | Mature |
| Formal Logic | SAT/SMT solvers, model checkers | **Yes** | Mature |
| Scientific | ??? | No — requires domain expertise | None |
| Strategic | ??? | No — requires simulation | None |
| Ethical | ??? | No — inherently subjective | None |

We don't score reasoning. We **prove** it. If we can't prove it, we don't touch it.
This constraint is our moat. Everyone else is building better scorers. We build provers.

---

## 2. What We Kill

Remove entirely from the codebase:

```
- Domain: SCIENTIFIC      → Cut. No mechanical verification possible.
- Domain: STRATEGIC       → Cut. Game-theoretic verification is research-stage.
- Domain: CAUSAL          → Cut. Causal inference verification requires SCMs we can't auto-generate.
- Domain: ETHICAL          → Cut. Inherently subjective. No proof exists.
- Heuristic novelty scoring → Cut. Replace with proof/no-proof binary.
- Heuristic quality scoring → Cut. Replace with formal verification verdict.
- Consensus-based scoring (Eq. 12) → Restructure. Consensus on PROOF VALIDITY, not subjective quality.
- CMS dimensions (Q, A, N, E) → Replace with new scoring dimensions (see Section 19).
```

---

## 3. What We Keep

From the MVP:

```
✓ ScoringEngine framework      → Refactor with new formulas, keep architecture
✓ Emission distribution (Eq. 5) → Keep as-is
✓ PEB mechanism (Eq. 4)         → Keep as-is
✓ Trap problems (Eq. 9)         → Keep, now with formally verified ground truth
✓ Slashing (Eq. 10)             → Keep as-is
✓ Simulator                     → Refactor for new scoring
✓ CLI runner                    → Keep
✓ Test infrastructure           → Extend
```

From the production plan:

```
✓ Bittensor protocol layer     → Rewrite Synapses for new roles
✓ Base neuron class             → Keep
✓ State persistence             → Keep
✓ Docker deployment             → Keep
✓ Monitoring                    → Keep
✓ CI/CD                         → Keep
```

---

## 4. The New Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXTERNAL WORLD                           │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ OpenAI   │  │ Anthropic│  │ DeepSeek │  │ Any LLM/Agent │  │
│  │ o1 / GPT │  │ Claude   │  │ R1       │  │ Framework     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬────────┘  │
│       │              │              │               │           │
│       └──────────────┴──────────────┴───────────────┘           │
│                              │                                  │
│              "Here is my reasoning chain"                       │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              REASONFORGE VERIFICATION API                 │  │
│  │                                                           │  │
│  │  POST /v1/verify                                          │  │
│  │  {                                                        │  │
│  │    "reasoning_chain": [...steps...],                      │  │
│  │    "domain": "mathematics" | "code" | "logic",            │  │
│  │    "original_query": "...",                                │  │
│  │    "claimed_answer": "...",                                │  │
│  │    "proof_level": "formal" | "standard" | "quick"         │  │
│  │  }                                                        │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BITTENSOR SUBNET                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    VALIDATORS                            │   │
│  │                                                          │   │
│  │  1. Receive verification request                         │   │
│  │  2. Dispatch to N miners (translators)                   │   │
│  │  3. Collect formal translations                          │   │
│  │  4. Run mechanical verification (Lean4/Sandbox/SMT)      │   │
│  │  5. Generate verification verdict                        │   │
│  │  6. Produce ZK certificate                               │   │
│  │  7. Set on-chain weights                                 │   │
│  │                                                          │   │
│  └───────────────┬──────────────────────┬───────────────────┘   │
│                  │                      │                       │
│          ┌───────▼───────┐      ┌───────▼───────┐              │
│          │   MINERS      │      │   MINERS      │              │
│          │ (Translators) │      │ (Translators) │   × N        │
│          │               │      │               │              │
│          │ NL reasoning  │      │ NL reasoning  │              │
│          │     ↓         │      │     ↓         │              │
│          │ Lean 4 proof  │      │ Lean 4 proof  │              │
│          │   — OR —      │      │   — OR —      │              │
│          │ Test suite    │      │ Test suite    │              │
│          │   — OR —      │      │   — OR —      │              │
│          │ FOL formula   │      │ FOL formula   │              │
│          └───────────────┘      └───────────────┘              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  ZK CERTIFICATE LAYER                    │   │
│  │                                                          │   │
│  │  Inputs:                                                 │   │
│  │    - Verification verdict (pass/fail per step)           │   │
│  │    - Validator signatures (N of M)                       │   │
│  │    - Task metadata hash                                  │   │
│  │                                                          │   │
│  │  Output:                                                 │   │
│  │    - ZK-SNARK proof that:                                │   │
│  │      ✓ Reasoning was formally verified                   │   │
│  │      ✓ N independent validators confirmed                │   │
│  │      ✓ Each step has mechanical proof                    │   │
│  │      — WITHOUT revealing the reasoning chain             │   │
│  │                                                          │   │
│  │  On-chain:                                               │   │
│  │    - Certificate registry (EVM contract)                 │   │
│  │    - Verify in O(1): verifyProof(certificate) → bool     │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETURNED TO CALLER                            │
│                                                                 │
│  {                                                              │
│    "certificate_id": "0xabc...def",                             │
│    "verdict": "VERIFIED" | "PARTIAL" | "FAILED",                │
│    "steps_verified": 7,                                         │
│    "steps_total": 7,                                            │
│    "failure_points": [],           // empty if all pass         │
│    "proof": "0x...",               // ZK-SNARK proof bytes      │
│    "registry_tx": "0x...",         // On-chain registration     │
│    "verification_time_ms": 4200,                                │
│    "validators_participated": 5,                                │
│    "confidence": 1.0,              // Binary: proved or not     │
│    "lean4_proofs": [...],          // Optional: raw proofs      │
│    "verify_url": "https://verify.reasonforge.ai/0xabc...def"   │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**The fundamental role change:**

| | Old Design | New Design |
|--|------------|------------|
| **Miners do** | Solve reasoning problems | Translate NL reasoning → formal proofs |
| **Validators do** | Score reasoning quality (heuristic) | Run mechanical verification (deterministic) |
| **Output is** | Subjective score (0.0 - 1.0) | Binary proof certificate (VERIFIED / FAILED) |
| **Customer gets** | "This reasoning scored 0.87" | "This reasoning is mathematically proven correct" |

---

# PART II — FORMAL VERIFICATION ENGINE

---

## 5. NL-to-Formal Translation Pipeline

This is the core innovation. Miners receive a natural language reasoning chain and
translate each step into a formally verifiable representation.

### 5.1 — Translation Interface

```python
class TranslationRequest:
    """What the miner receives."""
    task_id: str
    original_query: str              # The question that was asked
    reasoning_chain: list[dict]      # [{step_id, content, claimed_conclusion}]
    domain: str                      # "mathematics" | "code" | "logic"
    difficulty: int                  # 1-10
    proof_level: str                 # "formal" | "standard" | "quick"

class TranslationResult:
    """What the miner returns."""
    task_id: str
    translations: list[StepTranslation]
    compilation_status: str          # "COMPILED" | "PARTIAL" | "FAILED"
    full_proof: Optional[str]        # Complete Lean4/code/FOL artifact

class StepTranslation:
    """One reasoning step translated to formal representation."""
    step_id: int
    original_content: str            # NL text (echoed back for alignment)
    formal_representation: str       # Lean4 / Python+tests / FOL formula
    dependencies: list[int]          # Which previous steps this depends on
    translation_confidence: float    # Miner's self-reported confidence
    compilation_check: bool          # Did this step compile in isolation?
    notes: Optional[str]             # Miner's notes on translation choices
```

### 5.2 — Domain-Specific Translation Strategies

**Mathematics → Lean 4:**

```
NL: "Since n is even, we can write n = 2k for some integer k"
                    ↓ (Miner translates)
Lean 4:
  theorem step_3 (n : ℤ) (h : Even n) : ∃ k : ℤ, n = 2 * k := by
    exact h

NL: "Substituting n = 2k into n² gives (2k)² = 4k²"
                    ↓
Lean 4:
  theorem step_4 (k : ℤ) : (2 * k) ^ 2 = 4 * k ^ 2 := by
    ring
```

**Code → Executable Tests + Property Checks:**

```
NL: "We use a hash map to count frequencies, achieving O(n) time"
                    ↓ (Miner translates)
Python:
  def count_frequencies(arr: list[int]) -> dict[int, int]:
      freq = {}
      for x in arr:
          freq[x] = freq.get(x, 0) + 1
      return freq

  # Correctness tests
  def test_basic():
      assert count_frequencies([1,2,2,3]) == {1:1, 2:2, 3:1}

  def test_empty():
      assert count_frequencies([]) == {}

  # Property-based test
  from hypothesis import given, strategies as st

  @given(st.lists(st.integers(min_value=-1000, max_value=1000)))
  def test_sum_preserved(arr):
      freq = count_frequencies(arr)
      assert sum(freq.values()) == len(arr)

  # Complexity verification (statistical)
  def test_linear_time():
      import time
      for n in [1000, 10000, 100000]:
          arr = list(range(n))
          start = time.perf_counter()
          count_frequencies(arr)
          elapsed = time.perf_counter() - start
          # Should scale linearly (within 3x for 10x input)
      ratio = elapsed_100k / elapsed_10k
      assert ratio < 15  # Linear would be ~10, allow margin
```

**Formal Logic → FOL + SMT-LIB:**

```
NL: "All mammals are warm-blooded. Whales are mammals. Therefore whales are warm-blooded."
                    ↓ (Miner translates)
SMT-LIB:
  (declare-sort Animal)
  (declare-fun Mammal (Animal) Bool)
  (declare-fun WarmBlooded (Animal) Bool)
  (declare-fun IsWhale (Animal) Bool)

  ; Premise 1: All mammals are warm-blooded
  (assert (forall ((x Animal)) (=> (Mammal x) (WarmBlooded x))))

  ; Premise 2: Whales are mammals
  (assert (forall ((x Animal)) (=> (IsWhale x) (Mammal x))))

  ; Negation of conclusion (to prove by contradiction)
  (declare-const w Animal)
  (assert (IsWhale w))
  (assert (not (WarmBlooded w)))

  (check-sat)  ; Expected: UNSAT (meaning conclusion is valid)
```

### 5.3 — Translation Quality Tiers

```python
class ProofLevel(str, Enum):
    FORMAL = "formal"      # Full Lean4/Coq proof. Strongest guarantee. Slowest.
    STANDARD = "standard"  # Executable tests + property checks. Good balance.
    QUICK = "quick"        # Type checking + basic assertions. Fast, weaker guarantee.
```

| Level | Math | Code | Logic |
|-------|------|------|-------|
| **Formal** | Full Lean 4 proof, every step | Property-based tests + formal spec (TLA+/Dafny) | Complete FOL proof in Lean 4 |
| **Standard** | Lean 4 key lemmas + SymPy numerical | Unit tests + hypothesis + complexity | SMT-LIB + Z3 satisfiability |
| **Quick** | SymPy symbolic verification | Type checking + basic assertions | Propositional logic SAT check |

Pricing scales with proof level. Formal costs 10× quick. This is how we monetize.

---

## 6. Lean 4 Verification Backend

### 6.1 — Lean 4 Project Template

Every math verification task gets a fresh Lean 4 project:

```
lean_workspace/
├── lakefile.lean        # Lake build config
├── ReasonForge.lean     # Main entry
├── ReasonForge/
│   ├── Context.lean     # Problem statement + given assumptions
│   ├── Step1.lean       # Translation of reasoning step 1
│   ├── Step2.lean       # Translation of reasoning step 2
│   ├── ...
│   ├── StepN.lean       # Translation of reasoning step N
│   └── Chain.lean       # Full chain: imports all steps, proves final theorem
└── lean-toolchain        # Lean version pinning
```

### 6.2 — Chain.lean (The Key File)

```lean
-- Chain.lean: Proves the entire reasoning chain is valid
-- Auto-generated by validator from miner translations

import ReasonForge.Context
import ReasonForge.Step1
import ReasonForge.Step2
import ReasonForge.Step3

-- The final theorem that connects all steps
-- If this compiles, the entire reasoning chain is formally verified.
theorem reasoning_chain_valid
  (assumptions : ProblemContext)
  (s1 : Step1Result assumptions)
  (s2 : Step2Result s1)
  (s3 : Step3Result s2)
  : FinalConclusion s3 := by
  exact final_proof s1 s2 s3
```

### 6.3 — Lean4Verifier Class

```python
class Lean4Verifier:
    """
    Production Lean 4 verification backend.
    Compiles miner translations, extracts per-step verdicts.
    """

    def __init__(self, lean_toolchain: str = "leanprover/lean4:v4.8.0"):
        self.toolchain = lean_toolchain
        self.workspace_dir = Path("lean_workspaces")
        self.workspace_dir.mkdir(exist_ok=True)
        self.timeout = 120  # seconds per verification

    async def verify_chain(
        self,
        task_id: str,
        translations: list[StepTranslation],
        context: str,
    ) -> VerificationVerdict:
        """
        Full verification pipeline:
        1. Create Lean 4 project from translations
        2. Compile with `lake build`
        3. Parse output for per-step success/failure
        4. Return structured verdict
        """
        workspace = self.workspace_dir / task_id
        workspace.mkdir(exist_ok=True)

        try:
            # 1. Generate project files
            self._generate_lakefile(workspace)
            self._generate_context(workspace, context)
            for trans in translations:
                self._generate_step(workspace, trans)
            self._generate_chain(workspace, translations)

            # 2. Compile
            result = await self._run_lake_build(workspace)

            # 3. Parse results
            step_verdicts = self._parse_compilation_output(result, translations)

            # 4. Determine overall verdict
            all_passed = all(sv.verified for sv in step_verdicts)
            partial = any(sv.verified for sv in step_verdicts)

            return VerificationVerdict(
                task_id=task_id,
                overall="VERIFIED" if all_passed else ("PARTIAL" if partial else "FAILED"),
                step_verdicts=step_verdicts,
                total_steps=len(translations),
                verified_steps=sum(1 for sv in step_verdicts if sv.verified),
                failure_points=[sv for sv in step_verdicts if not sv.verified],
                raw_output=result.stdout,
                compilation_time_ms=result.elapsed_ms,
            )

        finally:
            # Cleanup workspace (or archive for audit)
            shutil.rmtree(workspace, ignore_errors=True)

    async def _run_lake_build(self, workspace: Path) -> CompilationResult:
        """Run `lake build` with timeout and resource limits."""
        start = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            "lake", "build",
            cwd=workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
            elapsed = int((time.monotonic() - start) * 1000)
            return CompilationResult(
                success=process.returncode == 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                elapsed_ms=elapsed,
            )
        except asyncio.TimeoutError:
            process.kill()
            return CompilationResult(
                success=False,
                stdout="",
                stderr="TIMEOUT",
                elapsed_ms=self.timeout * 1000,
            )

    def _parse_compilation_output(
        self, result: CompilationResult, translations: list[StepTranslation]
    ) -> list[StepVerdict]:
        """
        Parse Lean 4 compiler output to determine which steps succeeded.
        Lean reports errors with file:line:col format → map back to steps.
        """
        verdicts = []
        for trans in translations:
            step_file = f"Step{trans.step_id}.lean"
            # Check if this file had any errors in compiler output
            has_error = step_file in result.stderr and "error" in result.stderr
            verdicts.append(StepVerdict(
                step_id=trans.step_id,
                verified=not has_error and result.success,
                error_message=self._extract_error(result.stderr, step_file) if has_error else None,
                formal_representation=trans.formal_representation,
            ))
        return verdicts
```

---

## 7. Code Verification Backend

### 7.1 — CodeVerifier Class

```python
class CodeVerifier:
    """
    Verify code reasoning through execution, testing, and static analysis.
    All execution happens in Docker sandbox — zero trust on miner code.
    """

    def __init__(self, sandbox_image: str = "reasonforge-sandbox:latest"):
        self.sandbox = CodeSandbox(image=sandbox_image)
        self.timeout = 60

    async def verify_chain(
        self,
        task_id: str,
        translations: list[StepTranslation],
        original_code_claim: str,
    ) -> VerificationVerdict:
        """
        For each step:
        1. Extract code + tests from translation
        2. Run in sandbox
        3. Check: tests pass? Properties hold? Types check?
        """
        step_verdicts = []

        for trans in translations:
            # Parse the translation into code and test components
            components = self._parse_code_translation(trans)

            # Execute in sandbox
            exec_result = await self.sandbox.execute(
                code=components.implementation,
                tests=components.tests,
                property_tests=components.property_tests,
                timeout=self.timeout,
            )

            # Determine verdict
            tests_passed = exec_result.tests_passed == exec_result.tests_total
            properties_hold = exec_result.property_violations == 0

            step_verdicts.append(StepVerdict(
                step_id=trans.step_id,
                verified=tests_passed and properties_hold,
                error_message=exec_result.error if not tests_passed else None,
                details={
                    "tests_passed": exec_result.tests_passed,
                    "tests_total": exec_result.tests_total,
                    "property_violations": exec_result.property_violations,
                    "coverage_percent": exec_result.coverage,
                    "execution_time_ms": exec_result.elapsed_ms,
                },
            ))

        all_passed = all(sv.verified for sv in step_verdicts)
        return VerificationVerdict(
            task_id=task_id,
            overall="VERIFIED" if all_passed else "PARTIAL" if any(sv.verified for sv in step_verdicts) else "FAILED",
            step_verdicts=step_verdicts,
            total_steps=len(translations),
            verified_steps=sum(1 for sv in step_verdicts if sv.verified),
            failure_points=[sv for sv in step_verdicts if not sv.verified],
        )
```

### 7.2 — Sandbox Execution Protocol

```python
class SandboxExecution:
    """
    Docker sandbox for untrusted code execution.

    Security model:
    - No network access
    - No filesystem beyond /tmp (tmpfs, 64MB)
    - CPU limited to 50% of one core
    - Memory limited to 512MB
    - Process count limited to 50
    - No capabilities
    - Read-only rootfs
    - Non-root user
    - Timeout enforced externally
    """

    SANDBOX_CONFIG = {
        "network_disabled": True,
        "read_only": True,
        "mem_limit": "512m",
        "memswap_limit": "512m",         # No swap
        "cpu_period": 100000,
        "cpu_quota": 50000,               # 50% of one core
        "pids_limit": 50,
        "tmpfs": {"/tmp": "size=64m"},
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "user": "sandbox",
    }
```

---

## 8. First-Order Logic Backend

### 8.1 — FOLVerifier Class

```python
class FOLVerifier:
    """
    Verify logical reasoning using SMT solvers (Z3, CVC5).
    Strategy: translate logical argument to SMT-LIB, prove by refutation.
    """

    def __init__(self):
        self.solver_timeout = 30  # seconds

    async def verify_chain(
        self,
        task_id: str,
        translations: list[StepTranslation],
    ) -> VerificationVerdict:
        """
        For each step:
        1. Parse SMT-LIB from translation
        2. Assert premises + negation of conclusion
        3. If UNSAT → conclusion follows from premises (valid step)
        4. If SAT → conclusion does NOT follow (invalid step)
        5. If UNKNOWN → inconclusive
        """
        step_verdicts = []

        for trans in translations:
            smt_code = trans.formal_representation

            result = await self._run_z3(smt_code)

            if result.status == "unsat":
                # UNSAT means negated conclusion is impossible → step is valid
                verified = True
                error = None
            elif result.status == "sat":
                # SAT means there exists a counterexample → step is invalid
                verified = False
                error = f"Counterexample found: {result.model}"
            else:
                # UNKNOWN — solver couldn't decide
                verified = False
                error = f"Solver returned UNKNOWN after {self.solver_timeout}s"

            step_verdicts.append(StepVerdict(
                step_id=trans.step_id,
                verified=verified,
                error_message=error,
                details={"smt_status": result.status, "solver_time_ms": result.elapsed_ms},
            ))

        all_passed = all(sv.verified for sv in step_verdicts)
        return VerificationVerdict(
            task_id=task_id,
            overall="VERIFIED" if all_passed else "PARTIAL" if any(sv.verified for sv in step_verdicts) else "FAILED",
            step_verdicts=step_verdicts,
            total_steps=len(translations),
            verified_steps=sum(1 for sv in step_verdicts if sv.verified),
            failure_points=[sv for sv in step_verdicts if not sv.verified],
        )

    async def _run_z3(self, smt_code: str) -> SMTResult:
        """Execute Z3 on SMT-LIB input."""
        with tempfile.NamedTemporaryFile(suffix=".smt2", mode="w", delete=False) as f:
            f.write(smt_code)
            f.flush()
            process = await asyncio.create_subprocess_exec(
                "z3", f"-T:{self.solver_timeout}", f.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            os.unlink(f.name)

            output = stdout.decode().strip()
            if output.startswith("unsat"):
                return SMTResult(status="unsat", model=None)
            elif output.startswith("sat"):
                return SMTResult(status="sat", model=output)
            else:
                return SMTResult(status="unknown", model=None)
```

---

## 9. Step-Level Process Supervision

### 9.1 — The Process Supervision Model

This is the key differentiator from outcome-based verification.

```
Traditional (Outcome-based):
  "Is the final answer correct?" → Yes/No
  Problem: A correct answer can come from wrong reasoning (lucky guess)
  Problem: Can't identify WHERE reasoning breaks down

ReasonForge (Process-based):
  "Is each individual reasoning step provably correct?" → [Yes, Yes, No, Yes, ...]
  Benefit: Identifies exact failure point
  Benefit: Correct process guarantees correct outcome
  Benefit: Partial verification is still valuable
```

### 9.2 — Dependency Graph

Reasoning steps aren't linear — step 5 might depend on steps 2 and 3 but not 4.
Miners declare dependencies. Validators verify the dependency graph is valid.

```python
@dataclass
class StepDependencyGraph:
    """DAG of reasoning step dependencies."""

    steps: dict[int, StepTranslation]  # step_id → translation
    edges: dict[int, list[int]]        # step_id → [dependency step_ids]

    def validate_dag(self) -> bool:
        """Check: no cycles, all dependencies exist, topological order valid."""
        ...

    def get_verification_order(self) -> list[int]:
        """Return topological sort — verify leaves first, then dependents."""
        ...

    def invalidation_cascade(self, failed_step: int) -> set[int]:
        """If step N fails, which downstream steps are also invalid?"""
        ...
```

### 9.3 — Failure Localization

```python
@dataclass
class FailureReport:
    """Detailed report when verification fails."""

    failed_step_id: int
    original_reasoning: str         # What the AI said
    formal_translation: str         # What the miner translated it to
    verification_error: str         # Why it failed (compiler/solver output)
    suggested_fix: Optional[str]    # If the error is common, suggest a fix
    cascade_impact: list[int]       # Which downstream steps are invalidated
    last_valid_step: int            # The deepest step that still holds
    partial_correctness: float      # Fraction of chain that is verified (0.0 - 1.0)
```

This is what enterprises pay for. Not "your reasoning scored 0.87" but
"Steps 1-4 are mathematically proven correct. Step 5 contains an error:
the substitution of x=2k assumes k is positive, but k could be negative.
Steps 6-8 depend on Step 5 and are therefore unverified."

---

## 10. Verification Verdicts & Failure Localization

### 10.1 — Verdict Schema

```python
@dataclass
class VerificationVerdict:
    """The core output of the verification pipeline."""

    task_id: str
    overall: str                     # "VERIFIED" | "PARTIAL" | "FAILED"

    # Per-step results
    step_verdicts: list[StepVerdict]
    total_steps: int
    verified_steps: int

    # Failure analysis
    failure_points: list[StepVerdict]  # Steps that failed
    failure_report: Optional[FailureReport] = None

    # Metadata
    domain: str                      # "mathematics" | "code" | "logic"
    proof_level: str                 # "formal" | "standard" | "quick"
    verification_time_ms: int = 0

    # Raw artifacts (for audit)
    raw_output: Optional[str] = None  # Lean4 compiler output, test logs, etc.

    # Translators involved (miner UIDs)
    translator_uids: list[int] = field(default_factory=list)

    # For certificate generation
    verdict_hash: str = ""           # SHA-256 of canonical verdict representation

    def compute_verdict_hash(self) -> str:
        """Deterministic hash of the verdict for ZK proof input."""
        canonical = json.dumps({
            "task_id": self.task_id,
            "overall": self.overall,
            "steps": [
                {"id": sv.step_id, "verified": sv.verified}
                for sv in self.step_verdicts
            ],
        }, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

@dataclass
class StepVerdict:
    """Verification result for a single reasoning step."""

    step_id: int
    verified: bool                  # Binary: proved or not
    error_message: Optional[str]    # If failed, why
    formal_representation: str      # The formal translation that was checked
    details: dict = field(default_factory=dict)  # Backend-specific details
```

---

# PART III — ZK VERIFICATION CERTIFICATES

---

## 11. Certificate Schema

### 11.1 — What Goes Into a Certificate

```python
@dataclass
class VerificationCertificate:
    """
    A ZK-SNARK proof that reasoning was formally verified,
    without revealing the reasoning itself.
    """

    # Identity
    certificate_id: str              # Unique ID (hash of contents)
    version: int = 1                 # Schema version

    # What was verified (public inputs to ZK circuit)
    task_hash: str                   # SHA-256 of original query + reasoning chain
    domain: str                      # "mathematics" | "code" | "logic"
    proof_level: str                 # "formal" | "standard" | "quick"
    total_steps: int                 # Number of reasoning steps
    verified_steps: int              # Number that passed verification
    overall_verdict: str             # "VERIFIED" | "PARTIAL" | "FAILED"
    timestamp: int                   # Unix timestamp

    # Who verified (public inputs)
    validator_count: int             # Number of validators that participated
    validator_threshold: int         # Minimum required (e.g., 3 of 5)
    validator_commitment: str        # Merkle root of validator public keys

    # The proof itself
    zk_proof: bytes                  # SNARK proof bytes
    verification_key: str            # Reference to verification key

    # On-chain registration
    chain_id: int                    # EVM chain ID
    registry_address: str            # Certificate registry contract address
    tx_hash: Optional[str]           # Registration transaction hash
    block_number: Optional[int]      # Block number of registration

    # Verification URL
    verify_url: str                  # Public URL to verify certificate
```

### 11.2 — What the ZK Proof Proves (Without Revealing)

The proof attests to ALL of the following without revealing any details:

```
1. A reasoning chain of N steps was submitted
2. Each step was translated into a formal representation by M independent miners
3. A formal verification tool (Lean4 / sandbox / Z3) was run on each translation
4. K out of N steps compiled/passed/proved successfully
5. V out of W validators independently confirmed the verification
6. The verification was performed after timestamp T
7. The validators' combined stake exceeds threshold S
```

What remains HIDDEN:
```
- The original reasoning chain (privacy)
- The formal translations (IP protection)
- Individual validator scores (anonymity)
- The specific model that produced the reasoning (model-agnostic)
- Any proprietary data in the query (confidentiality)
```

---

## 12. ZK Circuit Design

### 12.1 — Circuit Architecture

We use Groth16 (for constant-size proofs) via circom or Halo2 (for recursive composition).

```
Circuit: ReasoningVerificationProof

Public Inputs:
  - task_hash: Field               (SHA-256 truncated to field element)
  - verdict_hash: Field            (SHA-256 of verification verdict)
  - validator_root: Field          (Merkle root of validator commitments)
  - min_validators: uint32         (threshold)
  - timestamp: uint64

Private Inputs (witness):
  - step_verdicts: bool[MAX_STEPS]     (per-step pass/fail)
  - validator_signatures: Sig[MAX_VALS] (BLS/EdDSA signatures on verdict_hash)
  - validator_pubkeys: PubKey[MAX_VALS]
  - validator_stakes: uint64[MAX_VALS]
  - merkle_proofs: MerkleProof[MAX_VALS] (prove validators are in commitment tree)

Constraints:
  1. verdict_hash == SHA256(step_verdicts)
  2. For each validator i:
     a. VerifySignature(validator_signatures[i], verdict_hash, validator_pubkeys[i]) == true
     b. MerkleVerify(validator_pubkeys[i], merkle_proofs[i], validator_root) == true
  3. count(valid_signatures) >= min_validators
  4. sum(validator_stakes where valid) >= STAKE_THRESHOLD
```

### 12.2 — Implementation Choice

```python
# We implement the ZK layer using one of:

# Option A: Circom + SnarkJS (JavaScript-friendly, well-documented)
# Pros: Large community, easy to deploy verifier on EVM
# Cons: Trusted setup, limited circuit flexibility

# Option B: Halo2 (Rust, recursive-friendly)
# Pros: No trusted setup, recursive composition native, high performance
# Cons: Smaller community, steeper learning curve

# Option C: SP1 / Risc0 (zkVM approach)
# Pros: Write verification logic in Rust, compiled to ZK circuit automatically
# Cons: Larger proof size, slower proving

# RECOMMENDED: Halo2 for recursive proofs + SP1 for complex verification logic
# This matches Xythum's existing ZK infrastructure
```

### 12.3 — Circuit Parameters

```python
# Circuit constraints budget
MAX_REASONING_STEPS = 32         # Max steps per verification
MAX_VALIDATORS = 16              # Max validators per certificate
MERKLE_DEPTH = 10                # Supports up to 1024 validators in tree
HASH_FUNCTION = "Poseidon"       # ZK-friendly hash (not SHA-256 in circuit)

# Proof generation estimates
PROVING_TIME_SECONDS = 10-30     # On modern CPU
PROOF_SIZE_BYTES = 256           # Groth16 constant size
VERIFICATION_TIME_MS = 2-5       # On-chain verification
VERIFICATION_GAS = ~250_000      # EVM gas cost
```

---

## 13. Recursive Proof Composition

### 13.1 — Why Recursive

A 10-step reasoning chain doesn't need 10 separate proofs. With recursive composition:

```
Step 1 proof: "Step 1 is verified"
Step 2 proof: "Step 2 is verified AND I've verified the proof that Step 1 is verified"
Step 3 proof: "Step 3 is verified AND I've verified the proof that Steps 1-2 are verified"
...
Step N proof: "Step N is verified AND I've verified the proof that Steps 1-(N-1) are verified"
```

The final proof is a SINGLE constant-size proof that the ENTIRE chain is valid.

### 13.2 — Recursive Circuit

```
RecursiveVerificationCircuit:

Public Inputs:
  - accumulated_verdict_hash: Field    (running hash of all step verdicts)
  - step_count: uint32                 (how many steps verified so far)
  - chain_commitment: Field            (commitment to the full chain)

Private Inputs:
  - previous_proof: Proof              (proof for steps 1..N-1)
  - current_step_verdict: bool         (did step N pass?)
  - current_step_formal: Field         (hash of formal representation)
  - verification_output: Field         (hash of verifier output)

Constraints:
  1. Verify(previous_proof, previous_public_inputs) == true
  2. accumulated_verdict_hash == Poseidon(
       previous_accumulated_hash, current_step_verdict, current_step_formal
     )
  3. step_count == previous_step_count + 1
```

### 13.3 — Implementation

```python
class RecursiveProver:
    """
    Compose step-level verification proofs into a single recursive proof.
    Uses Halo2 IVC (Incrementally Verifiable Computation).
    """

    def __init__(self, params_path: str):
        self.params = load_params(params_path)

    async def prove_chain(
        self,
        step_verdicts: list[StepVerdict],
        validator_commitments: list[ValidatorCommitment],
    ) -> VerificationCertificate:
        """
        Build recursive proof from step verdicts.

        Process:
        1. Base case: prove step 1
        2. For each subsequent step: fold in step N proof with previous accumulator
        3. Final: wrap in certificate with validator attestations
        """

        # Base case
        accumulator = await self._prove_base(step_verdicts[0])

        # Recursive folding
        for verdict in step_verdicts[1:]:
            accumulator = await self._fold_step(accumulator, verdict)

        # Add validator attestations
        final_proof = await self._finalize(
            accumulator, validator_commitments
        )

        # Construct certificate
        return VerificationCertificate(
            certificate_id=self._compute_id(final_proof),
            zk_proof=final_proof.to_bytes(),
            total_steps=len(step_verdicts),
            verified_steps=sum(1 for v in step_verdicts if v.verified),
            overall_verdict=self._compute_verdict(step_verdicts),
            ...
        )
```

---

## 14. On-Chain Certificate Registry

### 14.1 — Smart Contract (Solidity)

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./Verifier.sol";  // Auto-generated Groth16/Halo2 verifier

contract ReasonForgeCertificateRegistry {

    struct Certificate {
        bytes32 taskHash;          // Hash of the verified reasoning
        bytes32 verdictHash;       // Hash of verification verdict
        uint8 domain;              // 0=math, 1=code, 2=logic
        uint8 proofLevel;          // 0=formal, 1=standard, 2=quick
        uint16 totalSteps;
        uint16 verifiedSteps;
        uint8 verdict;             // 0=VERIFIED, 1=PARTIAL, 2=FAILED
        uint64 timestamp;
        uint8 validatorCount;
        address registrant;        // Who registered (validator's address)
    }

    // State
    mapping(bytes32 => Certificate) public certificates;   // certId → cert
    mapping(bytes32 => bool) public proofVerified;         // certId → was ZK proof valid
    Verifier public immutable verifier;                     // ZK proof verifier

    uint256 public totalCertificates;
    uint256 public totalVerified;

    // Events
    event CertificateRegistered(
        bytes32 indexed certificateId,
        bytes32 indexed taskHash,
        uint8 verdict,
        uint16 verifiedSteps,
        uint16 totalSteps
    );

    constructor(address _verifier) {
        verifier = Verifier(_verifier);
    }

    /**
     * @notice Register a new verification certificate with ZK proof
     * @param certId Unique certificate identifier
     * @param cert Certificate metadata
     * @param proof ZK-SNARK proof bytes
     * @param publicInputs Public inputs to the ZK circuit
     */
    function registerCertificate(
        bytes32 certId,
        Certificate calldata cert,
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) external {
        require(certificates[certId].timestamp == 0, "Certificate exists");

        // Verify ZK proof on-chain
        bool valid = verifier.verify(proof, publicInputs);
        require(valid, "Invalid ZK proof");

        // Store
        certificates[certId] = cert;
        proofVerified[certId] = true;
        totalCertificates++;

        if (cert.verdict == 0) {  // VERIFIED
            totalVerified++;
        }

        emit CertificateRegistered(
            certId, cert.taskHash, cert.verdict,
            cert.verifiedSteps, cert.totalSteps
        );
    }

    /**
     * @notice Check if reasoning has a valid verification certificate
     * @param taskHash Hash of the reasoning chain to check
     * @return True if a VERIFIED certificate exists
     */
    function isVerified(bytes32 taskHash) external view returns (bool) {
        // Search for certificate by task hash
        // In production: use a taskHash → certId mapping
        ...
    }

    /**
     * @notice Anyone can verify a certificate's ZK proof
     * @param certId Certificate to verify
     * @return True if the ZK proof is valid
     */
    function verifyCertificate(bytes32 certId) external view returns (bool) {
        return proofVerified[certId];
    }
}
```

### 14.2 — Deployment Targets

```
Primary:   EVM (Ethereum L2 — Arbitrum or Base for low gas costs)
Secondary: Bittensor EVM (when available)
Future:    Solana, Cosmos IBC
```

---

## 15. Certificate Verification Contract

The Verifier.sol is auto-generated from the ZK circuit. For Groth16:

```solidity
// Auto-generated by snarkjs or halo2 export
contract Verifier {
    // Verification key embedded as constants
    uint256 constant ALPHA_X = ...;
    uint256 constant ALPHA_Y = ...;
    // ... (elliptic curve points)

    function verify(
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) external view returns (bool) {
        // Pairing check
        // Gas cost: ~250,000
        ...
    }
}
```

---

# PART IV — BITTENSOR INTEGRATION (REVISED)

---

## 16. New Synapse Protocol

### 16.1 — TranslationTask Synapse (Validator → Miner)

```python
class TranslationTask(bt.Synapse):
    """
    Validator sends NL reasoning chain to miner for formal translation.
    This REPLACES the old ReasoningTask synapse.
    """

    # Immutable (validator sets)
    task_id: str
    original_query: str
    reasoning_chain: list[dict]      # [{step_id, content, claimed_conclusion}]
    domain: str                      # "mathematics" | "code" | "logic"
    difficulty: int
    proof_level: str                 # "formal" | "standard" | "quick"
    timeout_seconds: int = 300

    # Mutable (miner fills)
    translations: Optional[list[dict]] = None   # StepTranslation objects as dicts
    compilation_status: Optional[str] = None     # "COMPILED" | "PARTIAL" | "FAILED"
    full_proof_artifact: Optional[str] = None    # Base64 of complete proof file
    translation_time_ms: Optional[int] = None
    submission_hash: Optional[str] = None

    required_hash_fields: list[str] = [
        "task_id", "original_query", "domain", "difficulty", "proof_level"
    ]

    def deserialize(self) -> dict:
        return {
            "translations": self.translations or [],
            "compilation_status": self.compilation_status,
            "full_proof_artifact": self.full_proof_artifact,
            "translation_time_ms": self.translation_time_ms,
            "submission_hash": self.submission_hash,
        }
```

### 16.2 — VerificationResult Synapse (Validator → Miner, informational)

```python
class VerificationResult(bt.Synapse):
    """Validator notifies miner of their verification results."""

    epoch_id: int
    miner_uid: int
    tasks_translated: int
    steps_compiled: int              # How many of their translations compiled
    steps_total: int
    compilation_rate: float          # steps_compiled / steps_total
    epoch_score: float
    rank: int
    tao_earned: float

    required_hash_fields: list[str] = ["epoch_id", "miner_uid"]
```

---

## 17. Miner Role: Translator

### 17.1 — Translator Miner Architecture

```python
class TranslatorMiner(BaseNeuron):
    """
    Miners in the Proof Layer DON'T solve problems.
    They TRANSLATE natural language reasoning into formal proofs.

    This requires:
    1. Understanding the reasoning (comprehension)
    2. Knowing the target formal language (Lean4/Python/SMT-LIB)
    3. Producing compilable/executable output (precision)
    """

    def __init__(self, config):
        super().__init__(config)

        self.translator = TranslationEngine(
            backend=self.config.miner.backend,
            model=self.config.miner.model,
            domains=self.config.miner.domains,
        )

        # Local compilation check (optional but improves score)
        self.local_lean = LocalLeanChecker() if "mathematics" in self.config.miner.domains else None
        self.local_python = LocalPythonChecker() if "code" in self.config.miner.domains else None
        self.local_z3 = LocalZ3Checker() if "logic" in self.config.miner.domains else None

    async def handle_translation_task(self, synapse: TranslationTask) -> TranslationTask:
        """
        Core handler:
        1. Parse the reasoning chain
        2. For each step, generate formal translation
        3. Optionally verify locally before submitting
        4. Return translations
        """
        start = time.time_ns()

        try:
            translations = []
            for step in synapse.reasoning_chain:
                # Generate formal translation using LLM
                formal = await self.translator.translate_step(
                    step=step,
                    domain=synapse.domain,
                    proof_level=synapse.proof_level,
                    previous_steps=[t for t in translations],  # Context
                    original_query=synapse.original_query,
                )

                # Optional: local compilation check
                if self.local_lean and synapse.domain == "mathematics":
                    formal.compilation_check = await self.local_lean.quick_check(
                        formal.formal_representation
                    )

                translations.append(formal)

            # Fill Synapse
            synapse.translations = [asdict(t) for t in translations]
            synapse.compilation_status = self._assess_compilation(translations)
            synapse.full_proof_artifact = self._build_full_artifact(translations, synapse.domain)
            synapse.translation_time_ms = int((time.time_ns() - start) / 1_000_000)
            synapse.submission_hash = self._compute_hash(synapse)

        except Exception as e:
            bt.logging.error(f"Translation failed: {e}")
            synapse.translations = []
            synapse.compilation_status = "FAILED"

        return synapse
```

### 17.2 — Translation Engine

```python
class TranslationEngine:
    """
    Uses LLM to translate NL reasoning steps into formal representations.
    The system prompt is critical — it determines translation quality.
    """

    MATH_SYSTEM_PROMPT = """You are a Lean 4 formalization expert.
Your task: translate a natural language mathematical reasoning step into Lean 4 code.

Rules:
1. Each step becomes a `theorem` or `lemma` with explicit type signature
2. Use `by` tactic blocks. Prefer: ring, linarith, omega, simp, norm_num, exact
3. Declare all assumptions as hypotheses in the type signature
4. Reference previous steps by importing their theorems
5. If a step is an assumption/given, use `axiom` or `variable`
6. MUST compile independently. Test mentally before submitting.
7. Include type annotations for all variables
8. Add comments mapping back to the natural language

Output format:
```lean
-- NL: "{original natural language step}"
-- Dependencies: Step {N}, Step {M}

import ReasonForge.Step{N}
import ReasonForge.Step{M}

theorem step_{current_id}
  (h1 : {type from step N})
  (h2 : {type from step M})
  : {conclusion type} := by
  {tactic proof}
```

Do NOT use sorry. If you cannot prove it, say so explicitly."""

    CODE_SYSTEM_PROMPT = """You are a code verification expert.
Your task: translate a natural language code reasoning step into executable Python with tests.

Rules:
1. Each step produces: implementation + unit tests + property tests
2. Use hypothesis library for property-based testing
3. Include type hints on all functions
4. Tests must be runnable with pytest
5. If the step claims O(n) complexity, include a statistical timing test
6. Cover edge cases: empty input, single element, large input, negative numbers
7. Each test function must have a clear docstring explaining what it verifies

Output format:
```python
# NL: "{original natural language step}"
# Verifies: {what this step claims}

def {function_name}({params}: {types}) -> {return_type}:
    \"\"\"Implementation of step {N}.\"\"\"
    ...

def test_{function_name}_basic():
    \"\"\"Verify basic correctness.\"\"\"
    assert ...

def test_{function_name}_edge_cases():
    \"\"\"Verify edge case handling.\"\"\"
    assert ...

@given(...)
def test_{function_name}_properties():
    \"\"\"Verify claimed properties hold for all inputs.\"\"\"
    assert ...
```"""

    LOGIC_SYSTEM_PROMPT = """You are a formal logic expert.
Your task: translate a natural language logical reasoning step into SMT-LIB format.

Rules:
1. Declare all sorts, functions, and predicates
2. Assert all premises explicitly
3. To prove a conclusion: assert its NEGATION and check for UNSAT
4. Use quantifiers (forall, exists) where appropriate
5. Keep sorts minimal — only declare what's needed
6. Include comments mapping back to natural language

Output format:
```smt2
; NL: "{original natural language step}"
; Proves: {conclusion} follows from {premises}

(set-logic ALL)

; Declarations
(declare-sort ...)
(declare-fun ...)

; Premises (from previous steps)
(assert ...)

; Negated conclusion (proof by refutation)
(assert (not ...))

(check-sat)
; Expected: unsat (conclusion is valid)
```"""

    async def translate_step(
        self,
        step: dict,
        domain: str,
        proof_level: str,
        previous_steps: list,
        original_query: str,
    ) -> StepTranslation:
        """Use LLM to translate one reasoning step."""

        # Select system prompt based on domain
        system_prompt = {
            "mathematics": self.MATH_SYSTEM_PROMPT,
            "code": self.CODE_SYSTEM_PROMPT,
            "logic": self.LOGIC_SYSTEM_PROMPT,
        }[domain]

        # Build context from previous steps
        context = self._build_context(previous_steps, original_query)

        # Call LLM
        response = await self.backend.generate_structured(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
Original problem: {original_query}

Previous verified steps:
{context}

Current step to translate:
Step {step['step_id']}: {step['content']}
Claimed conclusion: {step.get('claimed_conclusion', 'N/A')}

Translate this step into {domain} formal representation.
"""},
            ],
            schema=StepTranslationSchema,
            timeout=60,
        )

        return StepTranslation(
            step_id=step["step_id"],
            original_content=step["content"],
            formal_representation=response["formal_representation"],
            dependencies=response.get("dependencies", []),
            translation_confidence=response.get("confidence", 0.5),
            compilation_check=False,  # Will be set by local checker
        )
```

---

## 18. Validator Role: Verifier

### 18.1 — Revised Validator Epoch Loop

```python
class ProofLayerValidator(BaseNeuron):
    """
    Validators in the Proof Layer:
    1. Receive verification requests (from API or generated)
    2. Dispatch to multiple translators (miners)
    3. Run MECHANICAL verification on returned translations
    4. Compare: multiple independent translations should verify the SAME steps
    5. Generate verification verdict
    6. Produce ZK certificate
    7. Set on-chain weights based on translation quality
    """

    def run_epoch(self):
        self.epoch_id += 1

        # Phase A: Get verification requests
        requests = self.task_manager.get_epoch_tasks()

        # Phase B: For each request, run verification pipeline
        results = []
        for req in requests:
            result = asyncio.run(self.verify_reasoning(req))
            results.append(result)

        # Phase C: Score miners based on translation quality
        self.score_translators(results)

        # Phase D: Set weights
        self.set_weights()

        # Phase E: Generate certificates for verified chains
        for result in results:
            if result.verdict.overall in ("VERIFIED", "PARTIAL"):
                cert = asyncio.run(self.generate_certificate(result))
                self.register_certificate(cert)

        # Phase F: Persist & notify
        self.save_state()
        asyncio.run(self.notify_miners())

    async def verify_reasoning(self, request: VerificationRequest) -> VerificationResult:
        """
        Full verification pipeline for one reasoning chain.
        """
        # 1. Select translators (miners)
        miner_uids = self.select_translators(request.domain, n=5)

        # 2. Send translation task to each
        synapse = TranslationTask(
            task_id=request.task_id,
            original_query=request.original_query,
            reasoning_chain=request.reasoning_chain,
            domain=request.domain,
            difficulty=request.difficulty,
            proof_level=request.proof_level,
        )

        axons = [self.metagraph.axons[uid] for uid in miner_uids]
        responses = await self.dendrite(
            axons=axons, synapse=synapse, timeout=request.timeout
        )

        # 3. For each miner's translation, run mechanical verification
        miner_verdicts = {}
        for uid, response in zip(miner_uids, responses):
            if not response.translations:
                miner_verdicts[uid] = None
                continue

            translations = [StepTranslation(**t) for t in response.translations]

            # Run the appropriate verifier
            if request.domain == "mathematics":
                verdict = await self.lean4_verifier.verify_chain(
                    request.task_id, translations, request.original_query
                )
            elif request.domain == "code":
                verdict = await self.code_verifier.verify_chain(
                    request.task_id, translations, request.original_query
                )
            elif request.domain == "logic":
                verdict = await self.fol_verifier.verify_chain(
                    request.task_id, translations
                )

            miner_verdicts[uid] = verdict

        # 4. Cross-validate: compare verdicts across miners
        # Multiple independent translations should agree on which steps are valid
        consensus_verdict = self.cross_validate(miner_verdicts)

        # 5. Score each miner based on their translation quality
        miner_scores = {}
        for uid, verdict in miner_verdicts.items():
            if verdict is None:
                miner_scores[uid] = 0.0
            else:
                miner_scores[uid] = self.score_translation(verdict, consensus_verdict)

        return VerificationResult(
            request=request,
            verdict=consensus_verdict,
            miner_verdicts=miner_verdicts,
            miner_scores=miner_scores,
        )

    def cross_validate(self, miner_verdicts: dict) -> VerificationVerdict:
        """
        Compare N independent translations to build consensus.

        If 3 out of 5 miners produce translations that verify step K,
        step K is considered verified. This handles:
        - Translation errors (one miner mistranslates)
        - Model limitations (one miner's LLM can't formalize a step)
        - Adversarial miners (one miner submits garbage)
        """
        valid_verdicts = {uid: v for uid, v in miner_verdicts.items() if v is not None}
        if not valid_verdicts:
            return VerificationVerdict(overall="FAILED", ...)

        # For each step, count how many miners got it to verify
        step_counts = defaultdict(int)
        step_total = defaultdict(int)

        for uid, verdict in valid_verdicts.items():
            for sv in verdict.step_verdicts:
                step_total[sv.step_id] += 1
                if sv.verified:
                    step_counts[sv.step_id] += 1

        # Step is verified if majority of miners got it to compile
        threshold = len(valid_verdicts) / 2
        consensus_steps = []
        for step_id in sorted(step_total.keys()):
            verified = step_counts[step_id] > threshold
            consensus_steps.append(StepVerdict(
                step_id=step_id,
                verified=verified,
                error_message=None if verified else f"Only {step_counts[step_id]}/{step_total[step_id]} translations compiled",
            ))

        all_verified = all(s.verified for s in consensus_steps)
        any_verified = any(s.verified for s in consensus_steps)

        return VerificationVerdict(
            overall="VERIFIED" if all_verified else ("PARTIAL" if any_verified else "FAILED"),
            step_verdicts=consensus_steps,
            total_steps=len(consensus_steps),
            verified_steps=sum(1 for s in consensus_steps if s.verified),
            failure_points=[s for s in consensus_steps if not s.verified],
        )
```

---

## 19. Revised Incentive Mechanism

### 19.1 — New Scoring Dimensions

Replace the old CMS (Quality, Accuracy, Novelty, Efficiency) with dimensions
that measure TRANSLATION quality:

```python
# ── New Composite Translation Score (CTS) ──
# Replaces CMS (Eq. 2)

W_COMPILATION = 0.45      # Did the translation compile/execute?
W_CORRECTNESS = 0.30      # Did verification pass? (binary per step)
W_COMPLETENESS = 0.15     # What fraction of steps were translated?
W_EFFICIENCY = 0.10       # Translation time relative to timeout

# CTS(m, t) = 0.45·Compilation + 0.30·Correctness + 0.15·Completeness + 0.10·Efficiency

class TranslationScorer:
    """Score miners based on their formal translations."""

    @staticmethod
    def compute_cts(
        translations: list[StepTranslation],
        verdict: VerificationVerdict,
        total_steps: int,
        time_ms: int,
        timeout_ms: int,
    ) -> float:
        """Composite Translation Score — replaces CMS."""

        if not translations:
            return 0.0

        # Compilation: what fraction of translations compiled?
        compiled = sum(1 for t in translations if t.compilation_check) / len(translations)

        # Correctness: what fraction of verified steps match consensus?
        if verdict and verdict.step_verdicts:
            correct = sum(1 for sv in verdict.step_verdicts if sv.verified) / len(verdict.step_verdicts)
        else:
            correct = 0.0

        # Completeness: did the miner translate all steps?
        completeness = len(translations) / max(1, total_steps)

        # Efficiency: faster is better (but not suspiciously fast)
        time_ratio = time_ms / max(1, timeout_ms)
        if time_ratio < 0.02:    # Suspiciously fast (< 2% of timeout)
            efficiency = 0.1
        elif time_ratio > 1.0:   # Timed out
            efficiency = 0.0
        else:
            efficiency = 1.0 - (time_ratio * 0.4)

        return (
            W_COMPILATION * compiled +
            W_CORRECTNESS * correct +
            W_COMPLETENESS * min(1.0, completeness) +
            W_EFFICIENCY * efficiency
        )
```

### 19.2 — Equations That Stay the Same

These work identically — just swap CMS for CTS:

```
Eq. 3 (S_epoch):  S_epoch(m) = avg(CTS(m,t) · D(t)) · trap_penalty — SAME formula, CTS replaces CMS
Eq. 4 (PEB):      PEB(m) = α · (1/rank) · √streak               — UNCHANGED
Eq. 5 (Emission):  R(m) = E_miner · [S·(1+PEB)] / Σ[S·(1+PEB)]  — UNCHANGED
Eq. 9 (Trap):     trap_penalty = avg_trap_score / θ if below       — UNCHANGED
Eq. 10 (Slash):    slash(v) = γ · stake · (θ - VAS)²              — UNCHANGED
```

### 19.3 — Revised Trap Problems

Traps are now reasoning chains with KNOWN formal translations:

```python
# Trap for mathematics domain
trap_task = {
    "reasoning_chain": [
        {"step_id": 1, "content": "Let n be an arbitrary integer"},
        {"step_id": 2, "content": "Since n² ≥ 0 for all integers, we have n² + 1 > 0"},
        {"step_id": 3, "content": "Therefore n² + 1 ≠ 0, so 1/(n²+1) is well-defined"},
    ],
    "ground_truth_translations": {
        1: "variable (n : ℤ)",
        2: "theorem step2 (n : ℤ) : n^2 + 1 > 0 := by positivity",
        3: "theorem step3 (n : ℤ) : n^2 + 1 ≠ 0 := by linarith [sq_nonneg n]",
    },
    "ground_truth_verdict": "VERIFIED",  # All 3 steps should verify
}
```

---

## 20. Weight Computation

Weights sent to chain are the same formula as before, just driven by CTS instead of CMS:

```python
def compute_weights(self, miner_states: dict[int, MinerState], n: int):
    """
    Map CTS-based epoch scores → on-chain weight vector.
    Yuma Consensus on-chain then determines actual TAO emissions.
    """
    weights = torch.zeros(n)
    for uid, state in miner_states.items():
        weights[uid] = state.s_epoch * (1.0 + state.peb)

    # Normalize
    if weights.sum() > 0:
        weights = weights / weights.sum()

    return weights
```

---

# PART V — ENTERPRISE API PRODUCT

---

## 21. Verification-as-a-Service API

### 21.1 — Endpoints

```
POST   /v1/verify                    — Submit reasoning chain for verification
GET    /v1/verify/{task_id}          — Poll for verification result
GET    /v1/certificates/{cert_id}    — Get certificate details
POST   /v1/certificates/{cert_id}/verify — Verify certificate ZK proof (off-chain)
GET    /v1/stats                     — Network statistics
WS     /v1/stream                    — Real-time verification updates
```

### 21.2 — Request/Response

```python
# Request
class VerifyRequest(BaseModel):
    reasoning_chain: list[ReasoningStep]   # The AI's reasoning to verify
    domain: Literal["mathematics", "code", "logic"]
    original_query: str                     # What was asked
    claimed_answer: str                     # What the AI concluded
    proof_level: Literal["formal", "standard", "quick"] = "standard"
    callback_url: Optional[str] = None     # Webhook on completion
    generate_certificate: bool = True      # Generate ZK certificate?

class ReasoningStep(BaseModel):
    step_id: int
    content: str                           # Natural language reasoning
    claimed_conclusion: Optional[str] = None

# Response
class VerifyResponse(BaseModel):
    task_id: str
    status: str                            # "queued" | "translating" | "verifying" | "complete"
    verdict: Optional[str] = None          # "VERIFIED" | "PARTIAL" | "FAILED"
    steps_verified: Optional[int] = None
    steps_total: Optional[int] = None
    failure_points: Optional[list[FailureDetail]] = None
    certificate_id: Optional[str] = None
    certificate_url: Optional[str] = None
    verification_time_ms: Optional[int] = None
    cost_credits: Optional[float] = None   # Credits consumed

class FailureDetail(BaseModel):
    step_id: int
    original_content: str
    error: str
    suggested_fix: Optional[str] = None
    cascade_impact: list[int]              # Steps invalidated by this failure
```

### 21.3 — Pricing Model

```python
PRICING = {
    "formal": {
        "per_step": 0.50,          # $0.50 per reasoning step
        "base_fee": 2.00,          # $2.00 base
        "certificate_fee": 1.00,   # $1.00 for ZK certificate
    },
    "standard": {
        "per_step": 0.10,
        "base_fee": 0.50,
        "certificate_fee": 0.50,
    },
    "quick": {
        "per_step": 0.02,
        "base_fee": 0.10,
        "certificate_fee": 0.25,
    },
}

# Example: 10-step mathematical proof, formal level
# Cost = $2.00 + (10 × $0.50) + $1.00 = $8.00

# Free tier: 50 quick verifications per month
# Pro tier: $99/mo, 500 standard verifications included
# Enterprise tier: Custom pricing, SLA, dedicated validators
```

---

## 22. SDK

### 22.1 — Python SDK

```python
from reasonforge import ReasonForge

rf = ReasonForge(api_key="rf_...")

# Verify a reasoning chain
result = rf.verify(
    reasoning_chain=[
        {"step_id": 1, "content": "Let x = 3 and y = 4"},
        {"step_id": 2, "content": "By Pythagorean theorem, x² + y² = z²"},
        {"step_id": 3, "content": "So z² = 9 + 16 = 25"},
        {"step_id": 4, "content": "Therefore z = 5"},
    ],
    domain="mathematics",
    original_query="Find the hypotenuse of a right triangle with legs 3 and 4",
    claimed_answer="z = 5",
    proof_level="formal",
)

print(result.verdict)           # "VERIFIED"
print(result.certificate_url)   # https://verify.reasonforge.ai/0xabc...

# Check if a step failed
if result.verdict == "PARTIAL":
    for failure in result.failure_points:
        print(f"Step {failure.step_id} failed: {failure.error}")
        print(f"  Last valid step: {failure.step_id - 1}")
```

### 22.2 — TypeScript SDK

```typescript
import { ReasonForge } from '@reasonforge/sdk';

const rf = new ReasonForge({ apiKey: 'rf_...' });

const result = await rf.verify({
  reasoningChain: [
    { stepId: 1, content: 'Initialize empty hash map' },
    { stepId: 2, content: 'Iterate through array, count frequencies' },
    { stepId: 3, content: 'Return the key with maximum count' },
  ],
  domain: 'code',
  originalQuery: 'Find the most frequent element in an array',
  claimedAnswer: 'Use hash map for O(n) solution',
  proofLevel: 'standard',
});

console.log(result.verdict); // "VERIFIED"
```

### 22.3 — Model Provider Integration (OpenAI wrapper example)

```python
import openai
from reasonforge import ReasonForge

rf = ReasonForge(api_key="rf_...")
client = openai.OpenAI()

# Step 1: Get reasoning from OpenAI
response = client.chat.completions.create(
    model="o1-preview",
    messages=[{"role": "user", "content": "Prove that √2 is irrational"}],
)

# Step 2: Parse reasoning steps (o1 returns them in thinking)
steps = rf.parse_reasoning_chain(response)  # Auto-extract from model output

# Step 3: Verify
verification = rf.verify(
    reasoning_chain=steps,
    domain="mathematics",
    original_query="Prove that √2 is irrational",
    claimed_answer=response.choices[0].message.content,
    proof_level="formal",
)

# Step 4: Attach certificate to your application's response
if verification.verdict == "VERIFIED":
    print(f"✓ Reasoning formally verified: {verification.certificate_url}")
else:
    print(f"⚠ Verification failed at step {verification.failure_points[0].step_id}")
```

---

## 23. Model Provider Integrations

Build first-party integrations for the top model providers:

```python
# reasonforge/integrations/openai.py
class OpenAIIntegration:
    """Auto-extract reasoning chains from o1/o3 thinking tokens."""
    def parse(self, response) -> list[ReasoningStep]: ...

# reasonforge/integrations/anthropic.py
class AnthropicIntegration:
    """Auto-extract reasoning from Claude's extended thinking."""
    def parse(self, response) -> list[ReasoningStep]: ...

# reasonforge/integrations/deepseek.py
class DeepSeekIntegration:
    """Auto-extract reasoning from DeepSeek R1 traces."""
    def parse(self, response) -> list[ReasoningStep]: ...

# reasonforge/integrations/langchain.py
class LangChainIntegration:
    """Middleware that auto-verifies LangChain agent reasoning chains."""
    def as_callback(self) -> BaseCallbackHandler: ...
```

---

## 24. Compliance Report Generator

For enterprise customers who need audit-ready documentation:

```python
class ComplianceReportGenerator:
    """
    Generate PDF compliance reports from verification results.
    Suitable for regulatory submission (EU AI Act, FDA, SEC).
    """

    def generate(self, verification_result, template="eu_ai_act") -> bytes:
        """
        Report includes:
        - Original query and reasoning chain
        - Per-step verification verdict with formal proofs
        - Failure analysis (if any)
        - Certificate reference (on-chain tx hash)
        - Validator participation details
        - Methodology description (suitable for auditor)
        - Timestamp and chain-of-custody proof
        """
        ...
```

---

# PART VI — BUILD ORDER & MILESTONES

---

## 25. Directory Structure

```
reasonforge/
├── [All MVP + Production files from previous plans]
│
├── reasonforge/
│   ├── [Existing modules]
│   │
│   ├── translation/                     # [NEW] NL-to-Formal pipeline
│   │   ├── __init__.py
│   │   ├── engine.py                    # TranslationEngine (LLM-powered)
│   │   ├── prompts.py                   # Domain-specific system prompts
│   │   ├── parsers.py                   # Parse LLM output into StepTranslation
│   │   └── types.py                     # TranslationRequest, TranslationResult, StepTranslation
│   │
│   ├── verification/                    # [EXPANDED] Now the core product
│   │   ├── __init__.py
│   │   ├── lean4_verifier.py            # Full Lean 4 verification pipeline
│   │   ├── code_verifier.py             # Code execution + property testing
│   │   ├── fol_verifier.py              # SMT-LIB / Z3 verification
│   │   ├── cross_validator.py           # Multi-miner consensus on verdicts
│   │   ├── verdict.py                   # VerificationVerdict, StepVerdict, FailureReport
│   │   └── process_supervisor.py        # Dependency graph, failure cascade
│   │
│   ├── certificates/                    # [NEW] ZK proof layer
│   │   ├── __init__.py
│   │   ├── schema.py                    # VerificationCertificate dataclass
│   │   ├── prover.py                    # ZK proof generation (Halo2/Circom)
│   │   ├── recursive.py                 # Recursive proof composition
│   │   ├── registry.py                  # On-chain certificate registration
│   │   └── verifier_client.py           # Off-chain certificate verification
│   │
│   ├── contracts/                       # [NEW] Solidity contracts
│   │   ├── CertificateRegistry.sol
│   │   ├── Verifier.sol                 # Auto-generated from ZK circuit
│   │   ├── deploy.py                    # Deployment scripts
│   │   └── abi/                         # Compiled ABIs
│   │
│   ├── sdk/                             # [NEW] Client SDKs
│   │   ├── python/
│   │   │   ├── reasonforge/__init__.py
│   │   │   ├── reasonforge/client.py
│   │   │   ├── reasonforge/types.py
│   │   │   └── setup.py
│   │   └── typescript/
│   │       ├── src/index.ts
│   │       ├── src/client.ts
│   │       ├── src/types.ts
│   │       └── package.json
│   │
│   └── integrations/                    # [NEW] Model provider integrations
│       ├── __init__.py
│       ├── openai.py
│       ├── anthropic.py
│       ├── deepseek.py
│       └── langchain.py
│
├── circuits/                            # [NEW] ZK circuits
│   ├── verification_circuit/
│   │   ├── src/main.rs                  # Halo2 circuit (or circom)
│   │   ├── Cargo.toml
│   │   └── params/                      # Proving/verification keys
│   └── recursive_circuit/
│       ├── src/main.rs
│       ├── Cargo.toml
│       └── params/
│
├── lean_templates/                      # [NEW] Lean 4 project templates
│   ├── lakefile.lean
│   ├── lean-toolchain
│   └── ReasonForge/
│       └── Template.lean
│
├── benchmarks/                          # [REVISED] Only 3 domains
│   ├── mathematics/
│   ├── code/
│   └── logic/
│
└── docs/                                # [EXPANDED]
    ├── [Previous docs]
    ├── PROOF_LAYER.md                   # This document
    ├── ZK_ARCHITECTURE.md               # ZK circuit documentation
    ├── CERTIFICATE_SPEC.md              # Certificate format specification
    └── INTEGRATION_GUIDE.md             # Model provider integration guide
```

---

## 26. Phase-by-Phase Build Order

```
PHASE 1 — DOMAIN NARROWING (Week 1)
  Step 1:   Remove scientific, strategic, causal, ethical domains from types.py
  Step 2:   Update task_generator.py for 3 domains only
  Step 3:   Update all tests for 3-domain model
  Step 4:   Run: pytest — verify nothing broke

PHASE 2 — TRANSLATION TYPES (Week 1)
  Step 5:   Write reasonforge/translation/types.py
  Step 6:   Write reasonforge/verification/verdict.py (VerificationVerdict, StepVerdict)
  Step 7:   Write tests for new types
  Step 8:   Run: pytest

PHASE 3 — LEAN 4 VERIFICATION (Weeks 2-3)
  Step 9:   Write lean_templates/ (project scaffold)
  Step 10:  Write reasonforge/verification/lean4_verifier.py
  Step 11:  Write 20 test cases with known Lean 4 proofs
  Step 12:  Write tests/test_lean4.py
  Step 13:  Run: pytest tests/test_lean4.py -v
  Step 14:  Benchmark: latency for 5-step / 10-step / 20-step proofs

PHASE 4 — CODE VERIFICATION (Week 3)
  Step 15:  Write reasonforge/verification/code_verifier.py
  Step 16:  Write docker/Dockerfile.sandbox (hardened)
  Step 17:  Write 20 test cases with known-correct code translations
  Step 18:  Write tests/test_code_verifier.py
  Step 19:  Run: pytest tests/test_code_verifier.py -v

PHASE 5 — FOL VERIFICATION (Week 4)
  Step 20:  Install Z3 in validator environment
  Step 21:  Write reasonforge/verification/fol_verifier.py
  Step 22:  Write 20 test cases with known-valid SMT-LIB formulas
  Step 23:  Write tests/test_fol_verifier.py
  Step 24:  Run: pytest tests/test_fol_verifier.py -v

PHASE 6 — TRANSLATION ENGINE (Weeks 4-5)
  Step 25:  Write reasonforge/translation/prompts.py (3 domain system prompts)
  Step 26:  Write reasonforge/translation/parsers.py
  Step 27:  Write reasonforge/translation/engine.py
  Step 28:  Write tests — generate translations for 50 benchmark problems
  Step 29:  Measure: compilation rate, correctness rate per domain
  Step 30:  Iterate on prompts until compilation rate > 70%

PHASE 7 — PROCESS SUPERVISION (Week 5)
  Step 31:  Write reasonforge/verification/process_supervisor.py (dependency DAG)
  Step 32:  Write reasonforge/verification/cross_validator.py (multi-miner consensus)
  Step 33:  Write tests for failure cascade, cross-validation
  Step 34:  Run: pytest

PHASE 8 — NEW SCORING MECHANISM (Week 6)
  Step 35:  Replace CMS with CTS in engine.py (new Eq. 2)
  Step 36:  Update simulator.py for translator miner profiles
  Step 37:  Update all formula tests
  Step 38:  Run CLI simulation with new scoring: verify elite translators score highest
  Step 39:  Run: pytest — all tests pass with new scoring

PHASE 9 — REVISED BITTENSOR NEURONS (Weeks 6-7)
  Step 40:  Write new protocol.py (TranslationTask, VerificationResult synapses)
  Step 41:  Write neurons/miner.py as TranslatorMiner
  Step 42:  Write neurons/validator.py as ProofLayerValidator
  Step 43:  Write tests/test_integration_local.py for new pipeline
  Step 44:  Test on localnet: validator → miner → translation → verification → weights

PHASE 10 — ZK CERTIFICATES (Weeks 8-10)
  Step 45:  Design ZK circuit (choose Halo2 or Circom)
  Step 46:  Write circuits/verification_circuit/
  Step 47:  Write circuits/recursive_circuit/
  Step 48:  Generate proving + verification keys
  Step 49:  Write reasonforge/certificates/prover.py
  Step 50:  Write reasonforge/certificates/recursive.py
  Step 51:  Write reasonforge/certificates/schema.py
  Step 52:  Write tests — generate certificate for known verification
  Step 53:  Benchmark: proving time, proof size, verification time

PHASE 11 — ON-CHAIN REGISTRY (Weeks 10-11)
  Step 54:  Write contracts/CertificateRegistry.sol
  Step 55:  Generate contracts/Verifier.sol from circuit
  Step 56:  Write contracts/deploy.py
  Step 57:  Deploy to testnet (Arbitrum Sepolia or Base Sepolia)
  Step 58:  Write reasonforge/certificates/registry.py (Python ↔ contract)
  Step 59:  Write reasonforge/certificates/verifier_client.py
  Step 60:  End-to-end test: verify → prove → register → verify on-chain

PHASE 12 — ENTERPRISE API (Weeks 11-12)
  Step 61:  Rewrite gateway/app.py for verification API
  Step 62:  Write gateway/schemas.py (VerifyRequest, VerifyResponse)
  Step 63:  Add webhook callbacks for async verification
  Step 64:  Add pricing/billing logic
  Step 65:  Write tests/test_api.py
  Step 66:  Load test: 100 concurrent verification requests

PHASE 13 — SDKs (Week 12)
  Step 67:  Write sdk/python/ (pip installable)
  Step 68:  Write sdk/typescript/ (npm installable)
  Step 69:  Write integration examples for OpenAI, Anthropic, LangChain
  Step 70:  Write sdk tests

PHASE 14 — COMPLIANCE & DOCS (Week 13)
  Step 71:  Write ComplianceReportGenerator (PDF output)
  Step 72:  Write docs/PROOF_LAYER.md
  Step 73:  Write docs/ZK_ARCHITECTURE.md
  Step 74:  Write docs/CERTIFICATE_SPEC.md
  Step 75:  Write docs/INTEGRATION_GUIDE.md
  Step 76:  Update README.md for proof layer positioning

PHASE 15 — FINAL INTEGRATION (Weeks 14-16)
  Step 77:  Full end-to-end test: API → subnet → verify → certificate → on-chain
  Step 78:  Run 100-epoch simulation with new scoring
  Step 79:  Security audit: adversarial translations, sandbox escape, ZK soundness
  Step 80:  Performance optimization: parallelize verification, cache embeddings
  Step 81:  Deploy validator + miner on Bittensor testnet
  Step 82:  Run for 48 hours, monitor stability
  Step 83:  Fix issues, re-deploy
  Step 84:  Write launch blog post
  Step 85:  Tag v1.0.0
```

---

## 27. Success Criteria

### Must-Have (v1.0)
- [ ] Lean 4 verifier: >80% compilation rate on benchmark math problems
- [ ] Code verifier: >90% test pass rate on benchmark code problems
- [ ] FOL verifier: >85% correct SAT/UNSAT on benchmark logic problems
- [ ] Translation engine: >70% of translations compile on first attempt
- [ ] Cross-validation: 3/5 miner agreement on step verdicts
- [ ] ZK certificate: generated in <30 seconds
- [ ] On-chain verification: <250k gas on EVM
- [ ] API: <60 second latency for 10-step standard verification
- [ ] Localnet: full epoch cycle works end-to-end
- [ ] All tests pass, no regressions from MVP

### Should-Have (v1.1)
- [ ] Recursive proofs: single proof for N-step chain
- [ ] Python SDK on PyPI
- [ ] TypeScript SDK on NPM
- [ ] OpenAI integration: auto-extract from o1 thinking tokens
- [ ] Compliance report generator: PDF output
- [ ] Grafana dashboards: verification success rates, miner leaderboards
- [ ] Docker deployment: one-command validator/miner setup

### Nice-to-Have (v1.2+)
- [ ] Anthropic integration: extended thinking extraction
- [ ] LangChain callback middleware
- [ ] Cross-subnet API (other Bittensor subnets query ReasonForge)
- [ ] Multi-chain certificate registry (Ethereum + Arbitrum + Base)
- [ ] Formal verification of the ZK circuit itself (meta-verification)

---

## 28. Dependency Map

```
types.py ──────────────────────────────────────────────────────────┐
                                                                    │
translation/types.py ──┬── translation/engine.py                    │
                       │         │                                  │
                       │         ▼                                  │
                       │   translation/prompts.py                   │
                       │   translation/parsers.py                   │
                       │                                            │
verification/verdict.py┤                                            │
                       │                                            │
                       ├── verification/lean4_verifier.py           │
                       ├── verification/code_verifier.py            │
                       ├── verification/fol_verifier.py             │
                       │         │                                  │
                       │         ▼                                  │
                       ├── verification/cross_validator.py          │
                       ├── verification/process_supervisor.py       │
                       │         │                                  │
                       │         ▼                                  │
engine.py ─────────────┼── scoring (CTS replaces CMS)              │
                       │         │                                  │
                       │         ▼                                  │
                       ├── certificates/prover.py ◄── circuits/     │
                       ├── certificates/recursive.py                │
                       ├── certificates/schema.py                   │
                       │         │                                  │
                       │         ▼                                  │
                       ├── certificates/registry.py ◄── contracts/  │
                       │         │                                  │
                       │         ▼                                  │
protocol.py ───────────┼── neurons/miner.py (TranslatorMiner)      │
                       ├── neurons/validator.py (ProofLayerValidator)
                       │         │                                  │
                       │         ▼                                  │
                       └── gateway/app.py ──► sdk/python/           │
                                             sdk/typescript/        │
                                             integrations/          │
```

---

*End of Proof Layer build plan. This transforms ReasonForge from "another AI scoring subnet" into "the trust infrastructure for all AI reasoning." The moat is the intersection of formal verification + Bittensor incentives + ZK proofs — a combination that requires exactly the skillset you have.*
