# ReasonForge — Production Subnet Build Plan

> **Purpose**: Upgrade the existing MVP (simulation-only codebase) into a production-ready Bittensor subnet deployable to testnet and mainnet. This plan assumes the MVP is already built and working — all 13 whitepaper formulas implemented, simulator passing tests, CLI running multi-epoch simulations.
>
> **Prerequisite**: The MVP codebase from `PLAN.md` must be complete and passing all tests before starting this plan.
>
> **Target**: Bittensor SDK v10 compatible subnet with real miners, validators, on-chain weight setting, formal verification, sandboxed code execution, embedding-based plagiarism detection, persistent state, API gateway, monitoring, Docker deployment, and CI/CD.

---

## Table of Contents

1. [Architecture Delta: MVP → Production](#1-architecture-delta)
2. [Directory Structure](#2-directory-structure)
3. [Phase 1: Bittensor Protocol Layer](#phase-1)
4. [Phase 2: Real Miner Neuron](#phase-2)
5. [Phase 3: Real Validator Neuron](#phase-3)
6. [Phase 4: Formal Verification & Sandboxed Execution](#phase-4)
7. [Phase 5: Embedding-Based Plagiarism Detection](#phase-5)
8. [Phase 6: Task Sourcing & Benchmark Database](#phase-6)
9. [Phase 7: Persistent State & Recovery](#phase-7)
10. [Phase 8: API Gateway & External Access](#phase-8)
11. [Phase 9: Monitoring & Observability](#phase-9)
12. [Phase 10: Security Hardening](#phase-10)
13. [Phase 11: Docker & Deployment](#phase-11)
14. [Phase 12: CI/CD & Testing Infrastructure](#phase-12)
15. [Phase 13: Documentation & SDK](#phase-13)
16. [Subnet Hyperparameters](#subnet-hyperparameters)
17. [Environment Variables](#environment-variables)
18. [Build Order](#build-order)
19. [Success Criteria](#success-criteria)

---

## 1. Architecture Delta: MVP → Production <a name="1-architecture-delta"></a>

### What the MVP has (keep everything):
```
reasonforge/
├── types.py           → KEEP: All dataclasses, constants, enums
├── engine.py          → KEEP: All 13 whitepaper formulas (stateless, tested)
├── simulator.py       → KEEP: Used for offline testing + benchmarking
├── plagiarism.py      → REPLACE: Upgrade from jaccard to embedding cosine similarity
├── task_generator.py  → EXPAND: Add real benchmark DB + API task ingestion
├── run.py             → KEEP: CLI simulation still useful for development
├── __init__.py        → KEEP
tests/                 → EXPAND: Add integration tests for real neurons
api/server.py          → EXPAND: Add auth, rate limiting, task submission
dashboard/App.jsx      → KEEP: Add live network stats panel
```

### What production adds:
```
NEW: reasonforge/protocol.py          — Bittensor Synapse definitions (wire protocol)
NEW: reasonforge/base/                — Base neuron class (wallet, subtensor, metagraph)
NEW: neurons/miner.py                 — Real miner neuron (Axon server, LLM reasoning)
NEW: neurons/validator.py             — Real validator neuron (Dendrite client, scoring, weight setting)
NEW: reasonforge/verification/        — Lean4 proof checker, code sandbox, fact checker
NEW: reasonforge/embeddings/          — Sentence-transformer embedding + cosine similarity
NEW: reasonforge/state/               — SQLite persistence, checkpoint/recovery
NEW: reasonforge/gateway/             — External API gateway (auth, billing, rate limits)
NEW: reasonforge/monitoring/          — Prometheus metrics, structured logging
NEW: reasonforge/security/            — Input sanitization, DoS protection, rate limiting
NEW: scripts/                         — Deployment, wallet setup, registration
NEW: docker/                          — Dockerfiles for miner, validator, gateway
NEW: .github/workflows/               — CI/CD pipelines
```

### Key Architectural Decisions:
1. **MVP engine.py is the single source of truth** — production neurons call the same `ScoringEngine` methods. No formula reimplementation.
2. **Validators own the scoring loop** — miners just solve tasks and return reasoning chains.
3. **Validators set on-chain weights** — computed from S_epoch scores, submitted via `subtensor.set_weights()`.
4. **State is local-first** — SQLite for persistence, no external DB dependency.
5. **Miners are model-agnostic** — any LLM backend (local, API, agent framework) plugs in via a standard interface.

---

## 2. Directory Structure <a name="2-directory-structure"></a>

```
reasonforge/
├── README.md
├── PLAN.md                              # MVP plan (reference)
├── PLAN_PRODUCTION.md                   # This file
├── pyproject.toml
├── requirements.txt                     # Core deps
├── requirements-miner.txt              # Miner-specific deps (torch, transformers)
├── requirements-validator.txt          # Validator-specific deps (lean4, docker)
├── requirements-dev.txt                # Dev deps (pytest, mypy, ruff)
├── .env.example                        # Environment variable template
├── .gitignore
│
├── reasonforge/                         # Core package (MVP + production extensions)
│   ├── __init__.py
│   ├── types.py                         # [MVP] Constants, dataclasses, enums
│   ├── engine.py                        # [MVP] All 13 whitepaper formulas
│   ├── simulator.py                     # [MVP] Epoch simulator (offline testing)
│   ├── task_generator.py                # [MVP→EXPANDED] + benchmark DB + API ingestion
│   ├── run.py                           # [MVP] CLI runner
│   │
│   ├── protocol.py                      # [NEW] Bittensor Synapse definitions
│   │
│   ├── base/                            # [NEW] Base neuron infrastructure
│   │   ├── __init__.py
│   │   ├── neuron.py                    # BaseNeuron: wallet, subtensor, metagraph, registration
│   │   └── config.py                    # CLI argument parsing, config management
│   │
│   ├── miner/                           # [NEW] Miner-side modules
│   │   ├── __init__.py
│   │   ├── reasoning.py                 # ReasoningEngine interface + implementations
│   │   ├── backends/                    # LLM backend adapters
│   │   │   ├── __init__.py
│   │   │   ├── base.py                  # Abstract LLMBackend class
│   │   │   ├── openai_backend.py        # OpenAI/compatible API backend
│   │   │   ├── anthropic_backend.py     # Anthropic API backend
│   │   │   ├── local_backend.py         # Local transformers/vLLM backend
│   │   │   └── agent_backend.py         # LangGraph/CrewAI agent backend
│   │   ├── domain_router.py             # Route tasks to domain-specialized prompts
│   │   └── proof_generator.py           # Generate formal proof fragments
│   │
│   ├── validator/                       # [NEW] Validator-side modules
│   │   ├── __init__.py
│   │   ├── scoring.py                   # Orchestrates full scoring pipeline
│   │   ├── objective_scorer.py          # Domain-specific automated checks
│   │   ├── consensus.py                 # Stake-weighted trimmed median
│   │   ├── weight_setter.py             # On-chain weight computation + submission
│   │   ├── task_manager.py              # Task queue, dispatch, assignment
│   │   └── trap_manager.py              # Trap problem injection + tracking
│   │
│   ├── verification/                    # [NEW] Formal verification backends
│   │   ├── __init__.py
│   │   ├── lean4_checker.py             # Lean 4 proof verification
│   │   ├── code_sandbox.py              # Docker-isolated code execution
│   │   ├── math_checker.py              # SymPy numerical/symbolic verification
│   │   └── fact_checker.py              # Citation and factual claim verification
│   │
│   ├── embeddings/                      # [NEW] Embedding-based similarity
│   │   ├── __init__.py
│   │   └── similarity.py               # Sentence-transformer cosine similarity
│   │
│   ├── state/                           # [NEW] Persistence layer
│   │   ├── __init__.py
│   │   ├── database.py                  # SQLite schema + CRUD operations
│   │   ├── checkpoint.py                # State serialization/recovery
│   │   └── migrations.py               # Schema versioning
│   │
│   ├── gateway/                         # [NEW] External API gateway
│   │   ├── __init__.py
│   │   ├── app.py                       # FastAPI app (public-facing)
│   │   ├── auth.py                      # API key management, JWT
│   │   ├── billing.py                   # Usage tracking, quotas
│   │   ├── rate_limiter.py              # Token-bucket rate limiting
│   │   └── schemas.py                   # Request/response Pydantic models
│   │
│   ├── monitoring/                      # [NEW] Observability
│   │   ├── __init__.py
│   │   ├── metrics.py                   # Prometheus counters/histograms/gauges
│   │   ├── logger.py                    # Structured JSON logging
│   │   └── health.py                    # Health check endpoints
│   │
│   └── security/                        # [NEW] Security utilities
│       ├── __init__.py
│       ├── sanitizer.py                 # Input validation, injection prevention
│       ├── rate_guard.py                # Per-UID rate limiting for validators
│       └── anomaly.py                   # Anomaly detection in miner behavior
│
├── neurons/                             # [NEW] Neuron entry points
│   ├── miner.py                         # Miner neuron entry point
│   └── validator.py                     # Validator neuron entry point
│
├── benchmarks/                          # [NEW] Benchmark task database
│   ├── README.md
│   ├── mathematics/                     # Domain-specific benchmark sets
│   │   ├── algebra.json
│   │   ├── calculus.json
│   │   ├── number_theory.json
│   │   └── combinatorics.json
│   ├── code/
│   │   ├── algorithms.json
│   │   ├── systems.json
│   │   └── debugging.json
│   ├── scientific/
│   │   ├── physics.json
│   │   ├── chemistry.json
│   │   └── biology.json
│   ├── strategic/
│   │   ├── game_theory.json
│   │   ├── optimization.json
│   │   └── planning.json
│   ├── causal/
│   │   ├── inference.json
│   │   └── counterfactual.json
│   └── ethical/
│       ├── dilemmas.json
│       └── policy_analysis.json
│
├── api/                                 # [MVP→EXPANDED] Internal simulation API
│   ├── __init__.py
│   └── server.py
│
├── dashboard/                           # [MVP→EXPANDED] React dashboard
│   └── App.jsx
│
├── tests/                               # [MVP→EXPANDED] Full test suite
│   ├── test_engine.py                   # [MVP] Formula unit tests
│   ├── test_simulator.py               # [MVP] Simulation integration tests
│   ├── test_types.py                    # [MVP] Type/constant tests
│   ├── test_protocol.py                # [NEW] Synapse serialization tests
│   ├── test_scoring.py                 # [NEW] Validator scoring pipeline
│   ├── test_verification.py            # [NEW] Lean4, sandbox, math checker
│   ├── test_embeddings.py              # [NEW] Similarity detection tests
│   ├── test_state.py                   # [NEW] Persistence + recovery
│   ├── test_gateway.py                 # [NEW] API gateway tests
│   ├── test_security.py               # [NEW] Input sanitization tests
│   ├── test_integration_local.py       # [NEW] Full miner↔validator on localnet
│   └── conftest.py                     # Shared fixtures
│
├── scripts/                             # [NEW] Deployment & operations
│   ├── setup_wallets.sh                 # Create owner/miner/validator wallets
│   ├── register_subnet.sh              # Register subnet on testnet/mainnet
│   ├── register_neurons.sh             # Register miner/validator UIDs
│   ├── stake.sh                        # Stake TAO to validator
│   ├── run_localnet.sh                 # Start local subtensor for development
│   ├── benchmark_import.py             # Import benchmark tasks into DB
│   ├── generate_traps.py               # Generate trap problems with ground truth
│   └── health_check.py                 # Production health verification
│
├── docker/                              # [NEW] Container definitions
│   ├── Dockerfile.miner                 # Miner container
│   ├── Dockerfile.validator             # Validator container
│   ├── Dockerfile.gateway               # API gateway container
│   ├── Dockerfile.sandbox               # Isolated code execution container
│   ├── docker-compose.yml               # Full stack (miner + validator + gateway + monitoring)
│   ├── docker-compose.localnet.yml      # Local development stack
│   └── docker-compose.monitoring.yml    # Prometheus + Grafana stack
│
├── monitoring/                          # [NEW] Monitoring configs
│   ├── prometheus.yml                   # Prometheus scrape config
│   ├── grafana/
│   │   └── dashboards/
│   │       ├── subnet_overview.json     # Grafana dashboard: subnet metrics
│   │       └── miner_performance.json   # Grafana dashboard: per-miner stats
│   └── alerts/
│       └── rules.yml                    # Alerting rules
│
├── docs/                                # [MVP→EXPANDED] Documentation
│   ├── ARCHITECTURE.md                  # System architecture
│   ├── PROTOCOL.md                      # Wire protocol specification
│   ├── MINER_GUIDE.md                   # How to run a miner
│   ├── VALIDATOR_GUIDE.md               # How to run a validator
│   ├── API_REFERENCE.md                 # External API docs
│   ├── DEPLOYMENT.md                    # Production deployment guide
│   ├── SECURITY.md                      # Security model & threat analysis
│   └── BENCHMARKS.md                    # Benchmark format & contribution guide
│
├── .github/                             # [NEW] CI/CD
│   └── workflows/
│       ├── test.yml                     # Run tests on PR
│       ├── lint.yml                     # Ruff + mypy
│       ├── build-docker.yml             # Build & push containers
│       └── release.yml                  # Tag-based release
│
└── min_compute.yml                      # [NEW] Bittensor minimum compute requirements
```

---

## Phase 1: Bittensor Protocol Layer <a name="phase-1"></a>

### 1.1 — protocol.py (Wire Protocol)

Define all Synapse subclasses for validator↔miner communication. Every Synapse inherits from `bt.Synapse`.

```python
import bittensor as bt
from typing import Optional, List
from pydantic import Field

class ReasoningTask(bt.Synapse):
    """Validator → Miner: Here is a reasoning task to solve."""

    # ── Immutable fields (set by validator, read by miner) ──
    task_id: str                                    # UUID
    problem: str                                    # Natural language problem statement
    domain: str                                     # "mathematics"|"code"|"scientific"|"strategic"|"causal"|"ethical"
    difficulty: int = Field(ge=1, le=10)            # Difficulty level
    timeout_seconds: int = 300                      # Max time to solve
    context: Optional[str] = None                   # Additional context/data
    constraints: Optional[str] = None               # Specific constraints

    # ── Mutable fields (filled by miner, read back by validator) ──
    reasoning_steps: Optional[List[dict]] = None    # List of {step_id, reasoning, evidence, confidence}
    final_answer: Optional[str] = None              # Final answer text
    proof_status: Optional[str] = None              # "VERIFIED"|"FAILED"|None
    proof_artifact: Optional[str] = None            # Base64 encoded proof file (Lean4, Coq)
    code_artifact: Optional[str] = None             # Code solution if applicable
    time_taken_ms: Optional[int] = None             # Self-reported solve time
    submission_hash: Optional[str] = None           # SHA-256 of steps+answer for integrity

    # Required by Bittensor
    required_hash_fields: List[str] = ["task_id", "problem", "domain", "difficulty"]

    def deserialize(self) -> dict:
        """Deserialize response into scoreable format."""
        return {
            "task_id": self.task_id,
            "steps": self.reasoning_steps or [],
            "final_answer": self.final_answer,
            "proof_status": self.proof_status,
            "proof_artifact": self.proof_artifact,
            "code_artifact": self.code_artifact,
            "time_taken_ms": self.time_taken_ms,
            "submission_hash": self.submission_hash,
        }


class HealthCheck(bt.Synapse):
    """Validator → Miner: Are you alive and what are your capabilities?"""

    # Mutable
    status: Optional[str] = None                    # "ready"|"busy"|"warming_up"
    supported_domains: Optional[List[str]] = None   # Which domains this miner supports
    model_info: Optional[str] = None                # Model identifier (optional, for transparency)
    version: Optional[str] = None                   # Miner software version

    required_hash_fields: List[str] = []

    def deserialize(self) -> dict:
        return {
            "status": self.status,
            "supported_domains": self.supported_domains,
            "model_info": self.model_info,
            "version": self.version,
        }


class TaskResult(bt.Synapse):
    """Validator → Miner: Here are your scores for a batch of tasks (informational)."""

    # Immutable
    epoch_id: int
    miner_uid: int
    scores: Optional[List[dict]] = None             # [{task_id, cms, rank}]
    s_epoch: Optional[float] = None
    rank: Optional[int] = None
    total_tao: Optional[float] = None

    required_hash_fields: List[str] = ["epoch_id", "miner_uid"]

    def deserialize(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "scores": self.scores,
            "s_epoch": self.s_epoch,
            "rank": self.rank,
        }
```

**Rules:**
- `required_hash_fields` contains only the fields the validator sets (immutable). Miner cannot tamper with these.
- Mutable fields use `Optional[...] = None` — the miner fills them in.
- Synapses are Pydantic models under the hood — they auto-serialize/deserialize for transit.
- Keep Synapse payloads under 1MB. For large proof artifacts, use base64 encoding with compression.
- The `submission_hash` field lets validators verify integrity: `SHA256(json.dumps(steps) + final_answer)`.

### 1.2 — base/neuron.py (Base Neuron Class)

All neurons (miner and validator) inherit from this base:

```python
class BaseNeuron:
    """Shared infrastructure for miners and validators."""

    neuron_type: str  # "miner" or "validator"

    def __init__(self, config=None):
        # 1. Parse CLI args / config
        self.config = config or self.get_config()

        # 2. Initialize Bittensor objects
        self.wallet = bt.Wallet(config=self.config)
        self.subtensor = bt.Subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)

        # 3. Check registration
        self.uid = self.get_uid()
        if self.uid is None:
            bt.logging.error("Neuron not registered. Run: btcli register")
            exit(1)

        # 4. Initialize state persistence
        self.state_db = StateDatabase(
            db_path=f"state/{self.neuron_type}_{self.uid}.db"
        )

        # 5. Initialize metrics
        self.metrics = MetricsCollector(neuron_type=self.neuron_type, uid=self.uid)

        # 6. Load previous state if exists
        self.load_state()

    @staticmethod
    def get_config() -> bt.Config:
        parser = argparse.ArgumentParser()
        parser.add_argument("--netuid", type=int, required=True)
        parser.add_argument("--subtensor.network", type=str, default="finney")
        parser.add_argument("--subtensor.chain_endpoint", type=str, default=None)
        parser.add_argument("--logging.debug", action="store_true")
        bt.Wallet.add_args(parser)
        bt.Subtensor.add_args(parser)
        bt.logging.add_args(parser)
        return bt.Config(parser)

    def get_uid(self) -> Optional[int]:
        """Find our UID in the metagraph."""
        hotkey = self.wallet.hotkey.ss58_address
        if hotkey in self.metagraph.hotkeys:
            return self.metagraph.hotkeys.index(hotkey)
        return None

    def sync(self):
        """Re-sync metagraph from chain."""
        self.metagraph.sync(subtensor=self.subtensor)

    def should_sync_metagraph(self) -> bool:
        """Sync every 5 blocks (60 seconds)."""
        current_block = self.subtensor.get_current_block()
        return (current_block - self.last_sync_block) >= 5

    def save_state(self):
        """Persist neuron state to SQLite."""
        self.state_db.save_checkpoint(self.get_state_dict())

    def load_state(self):
        """Restore from last checkpoint."""
        state = self.state_db.load_latest_checkpoint()
        if state:
            self.restore_state_dict(state)

    @abstractmethod
    def get_state_dict(self) -> dict: ...

    @abstractmethod
    def restore_state_dict(self, state: dict): ...

    @abstractmethod
    def run(self): ...
```

### 1.3 — base/config.py

Extend CLI argument parsing for miner-specific and validator-specific flags:

```python
class MinerConfig:
    """Additional args for miner neurons."""
    @staticmethod
    def add_args(parser):
        parser.add_argument("--miner.backend", type=str, default="openai",
                            choices=["openai", "anthropic", "local", "agent"])
        parser.add_argument("--miner.model", type=str, default="gpt-4o")
        parser.add_argument("--miner.api_key_env", type=str, default="OPENAI_API_KEY")
        parser.add_argument("--miner.max_concurrent", type=int, default=4)
        parser.add_argument("--miner.port", type=int, default=8091)
        parser.add_argument("--miner.domains", type=str, nargs="+",
                            default=["mathematics", "code", "scientific", "strategic", "causal", "ethical"])

class ValidatorConfig:
    """Additional args for validator neurons."""
    @staticmethod
    def add_args(parser):
        parser.add_argument("--validator.epoch_length", type=int, default=360,
                            help="Blocks per epoch (360 = ~72 min)")
        parser.add_argument("--validator.tasks_per_epoch", type=int, default=12)
        parser.add_argument("--validator.trap_rate", type=float, default=0.15)
        parser.add_argument("--validator.timeout", type=int, default=300)
        parser.add_argument("--validator.sample_size", type=int, default=16,
                            help="Number of miners to query per task")
        parser.add_argument("--validator.port", type=int, default=8092)
        parser.add_argument("--validator.sandbox_enabled", action="store_true")
        parser.add_argument("--validator.lean4_enabled", action="store_true")
        parser.add_argument("--validator.embedding_model", type=str,
                            default="all-MiniLM-L6-v2")
```

---

## Phase 2: Real Miner Neuron <a name="phase-2"></a>

### 2.1 — neurons/miner.py (Entry Point)

The miner neuron:
1. Registers an **Axon** server
2. Attaches handler functions for each Synapse type
3. Serves continuously, responding to validator queries

```python
class ReasonForgeMiner(BaseNeuron):
    neuron_type = "miner"

    def __init__(self, config=None):
        super().__init__(config)

        # Initialize LLM backend based on config
        self.reasoning_engine = ReasoningEngine(
            backend=self.config.miner.backend,
            model=self.config.miner.model,
            domains=self.config.miner.domains,
        )

        # Create Axon server
        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        # Attach handlers
        self.axon.attach(
            forward_fn=self.handle_reasoning_task,
            blacklist_fn=self.blacklist_reasoning_task,
            priority_fn=self.priority_reasoning_task,
        ).attach(
            forward_fn=self.handle_health_check,
        ).attach(
            forward_fn=self.handle_task_result,
        )

    async def handle_reasoning_task(self, synapse: ReasoningTask) -> ReasoningTask:
        """Core handler: receive task, produce reasoning chain, return."""
        start_time = time.time_ns()

        try:
            # Route to domain-specific prompt template
            prompt = self.reasoning_engine.domain_router.build_prompt(synapse)

            # Execute multi-step reasoning
            result = await self.reasoning_engine.solve(
                problem=synapse.problem,
                domain=synapse.domain,
                difficulty=synapse.difficulty,
                context=synapse.context,
                constraints=synapse.constraints,
                timeout=synapse.timeout_seconds,
            )

            # Fill mutable Synapse fields
            synapse.reasoning_steps = [
                {
                    "step_id": i,
                    "reasoning": step.reasoning,
                    "evidence": step.evidence,
                    "confidence": step.confidence,
                    "formal_proof_fragment": step.formal_proof_fragment,
                }
                for i, step in enumerate(result.steps)
            ]
            synapse.final_answer = result.final_answer
            synapse.proof_status = result.proof_status
            synapse.proof_artifact = result.proof_artifact
            synapse.code_artifact = result.code_artifact
            synapse.time_taken_ms = int((time.time_ns() - start_time) / 1_000_000)
            synapse.submission_hash = self._compute_hash(synapse)

        except Exception as e:
            bt.logging.error(f"Task {synapse.task_id} failed: {e}")
            synapse.final_answer = f"ERROR: {str(e)}"
            synapse.reasoning_steps = []

        return synapse

    def blacklist_reasoning_task(self, synapse: ReasoningTask) -> tuple[bool, str]:
        """Reject requests from non-validators or unregistered neurons."""
        caller_hotkey = synapse.dendrite.hotkey
        if caller_hotkey not in self.metagraph.hotkeys:
            return True, "Unregistered hotkey"
        caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
        if not self.metagraph.validator_permit[caller_uid]:
            return True, "No validator permit"
        return False, ""

    def priority_reasoning_task(self, synapse: ReasoningTask) -> float:
        """Higher-stake validators get priority."""
        caller_hotkey = synapse.dendrite.hotkey
        caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
        return float(self.metagraph.S[caller_uid])

    def run(self):
        """Main loop."""
        bt.logging.info(f"Miner starting on UID {self.uid}")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        while True:
            if self.should_sync_metagraph():
                self.sync()
            time.sleep(12)  # One block
```

### 2.2 — miner/reasoning.py (Reasoning Engine)

Abstract interface + orchestrator that routes to the correct backend:

```python
class ReasoningEngine:
    def __init__(self, backend: str, model: str, domains: list[str]):
        self.backend = self._create_backend(backend, model)
        self.domain_router = DomainRouter(domains)

    async def solve(self, problem, domain, difficulty, context, constraints, timeout) -> ReasoningResult:
        """
        Execute multi-step reasoning:
        1. Build domain-specific system prompt
        2. Request chain-of-thought from LLM
        3. Parse structured reasoning steps
        4. Attempt formal proof generation (math/code domains)
        5. Return structured result
        """
        ...

@dataclass
class ReasoningResult:
    steps: list[ReasoningStep]
    final_answer: str
    proof_status: Optional[str]
    proof_artifact: Optional[str]
    code_artifact: Optional[str]
```

### 2.3 — miner/backends/base.py (LLM Backend Interface)

```python
class LLMBackend(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict], temperature: float,
                       max_tokens: int, timeout: int) -> str: ...

    @abstractmethod
    async def generate_structured(self, messages: list[dict],
                                   schema: dict, timeout: int) -> dict: ...
```

**Implementations required:**
- `openai_backend.py` — OpenAI/compatible (GPT-4o, DeepSeek, local vLLM)
- `anthropic_backend.py` — Anthropic (Claude Sonnet/Opus)
- `local_backend.py` — HuggingFace transformers, direct GPU inference
- `agent_backend.py` — LangGraph/CrewAI multi-agent reasoning

### 2.4 — miner/domain_router.py

Maps domains to specialized system prompts and output parsers:

```python
DOMAIN_PROMPTS = {
    Domain.MATHEMATICS: """You are a mathematical reasoning engine. For each step:
    1. State your approach
    2. Show formal work
    3. Verify the step
    If possible, express proofs in Lean 4 syntax.
    Output: structured JSON with steps array.""",

    Domain.CODE: """You are a code reasoning engine. For each step:
    1. Analyze requirements
    2. Design solution approach
    3. Implement with test cases
    Output includes executable code artifact.""",

    # ... (all 6 domains)
}
```

---

## Phase 3: Real Validator Neuron <a name="phase-3"></a>

### 3.1 — neurons/validator.py (Entry Point)

The validator neuron runs the **main epoch loop**:

```python
class ReasonForgeValidator(BaseNeuron):
    neuron_type = "validator"

    def __init__(self, config=None):
        super().__init__(config)

        # Initialize components
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.task_manager = TaskManager(config=self.config)
        self.trap_manager = TrapManager(trap_rate=self.config.validator.trap_rate)
        self.scorer = ValidatorScorer(config=self.config)
        self.weight_setter = WeightSetter(subtensor=self.subtensor, config=self.config)
        self.similarity_detector = SimilarityDetector(
            model_name=self.config.validator.embedding_model
        )

        # State tracking (per-epoch)
        self.miner_states: dict[int, MinerState] = {}    # uid → MinerState
        self.epoch_id: int = 0
        self.scores: torch.FloatTensor = torch.zeros(256) # Weight vector

    def run(self):
        """Main validator loop."""
        bt.logging.info(f"Validator starting on UID {self.uid}")

        while True:
            try:
                # 1. Sync metagraph
                self.sync()

                # 2. Check if epoch boundary
                current_block = self.subtensor.get_current_block()
                if self.is_epoch_boundary(current_block):
                    self.run_epoch()

                # 3. Sleep for one block
                time.sleep(12)

            except Exception as e:
                bt.logging.error(f"Validator loop error: {e}")
                traceback.print_exc()
                time.sleep(12)

    def run_epoch(self):
        """Execute one complete scoring epoch."""
        self.epoch_id += 1
        bt.logging.info(f"=== EPOCH {self.epoch_id} ===")

        # Phase A: Generate tasks for this epoch
        tasks = self.task_manager.generate_epoch_tasks(
            count=self.config.validator.tasks_per_epoch,
            trap_rate=self.config.validator.trap_rate,
        )

        # Phase B: For each task, query miners and score
        all_task_results = []
        for task in tasks:
            task_result = asyncio.run(self.process_task(task))
            all_task_results.append(task_result)

        # Phase C: Compute epoch scores using MVP engine
        self.compute_epoch_scores(all_task_results)

        # Phase D: Compute and set on-chain weights
        self.set_weights()

        # Phase E: Persist state
        self.save_state()

        # Phase F: Send score notifications to miners (informational)
        asyncio.run(self.notify_miners())

    async def process_task(self, task: Task) -> TaskProcessingResult:
        """Query miners, collect responses, score them."""

        # 1. Select miners to query (sample or all)
        miner_uids = self.get_queryable_miners()

        # 2. Build Synapse
        synapse = ReasoningTask(
            task_id=task.task_id,
            problem=task.problem,
            domain=task.domain.value,
            difficulty=task.difficulty,
            timeout_seconds=self.config.validator.timeout,
        )

        # 3. Query miners via dendrite
        axons = [self.metagraph.axons[uid] for uid in miner_uids]
        responses: List[ReasoningTask] = await self.dendrite(
            axons=axons,
            synapse=synapse,
            timeout=self.config.validator.timeout,
        )

        # 4. Score each response
        scored_results = []
        for uid, response in zip(miner_uids, responses):
            # 4a. Check for timeout/failure
            if response.final_answer is None:
                scored_results.append((uid, DimensionScores(0, 0, 0, 0)))
                continue

            # 4b. Verify submission hash integrity
            expected_hash = self._compute_hash(response)
            if response.submission_hash != expected_hash:
                bt.logging.warning(f"UID {uid}: hash mismatch, penalizing")
                scored_results.append((uid, DimensionScores(0, 0, 0, 0)))
                continue

            # 4c. Run plagiarism check against other responses
            similarity = self.similarity_detector.check_against_batch(
                response, [r for r in responses if r != response]
            )
            plagiarism_penalty = SIMILARITY_PENALTY if similarity > SIMILARITY_THRESHOLD else 1.0

            # 4d. Compute objective score (automated checks)
            o_score = await self.scorer.compute_objective_score(task, response)

            # 4e. Compute dimension scores
            dim_scores = await self.scorer.compute_dimensions(task, response)

            # 4f. Apply plagiarism penalty
            dim_scores = DimensionScores(
                quality=dim_scores.quality * plagiarism_penalty,
                accuracy=dim_scores.accuracy * plagiarism_penalty,
                novelty=dim_scores.novelty * plagiarism_penalty,
                efficiency=dim_scores.efficiency,
            )

            scored_results.append((uid, dim_scores))

            # 4g. Track trap scores
            if task.is_trap:
                self.track_trap_score(uid, dim_scores.cms, task.ground_truth_score)

        return TaskProcessingResult(task=task, scored_results=scored_results)

    def compute_epoch_scores(self, task_results: list):
        """Aggregate per-task CMS into S_epoch using MVP engine."""
        for uid in self.get_all_miner_uids():
            miner_state = self.get_or_create_miner_state(uid)

            # Gather CMS scores and difficulty multipliers for this miner
            cms_list = []
            diff_mults = []
            for tr in task_results:
                for scored_uid, dim_scores in tr.scored_results:
                    if scored_uid == uid:
                        cms = ScoringEngine.compute_cms(dim_scores)
                        cms_list.append(cms)
                        diff_mults.append(tr.task.difficulty_multiplier)

            if not cms_list:
                continue

            # Compute trap penalty (Eq. 9)
            trap_penalty = ScoringEngine.compute_trap_penalty(miner_state.trap_scores)

            # Compute S_epoch (Eq. 3)
            miner_state.s_epoch = ScoringEngine.compute_s_epoch(
                cms_list, diff_mults, trap_penalty
            )

        # Rank miners
        ranked = sorted(
            [ms for ms in self.miner_states.values() if ms.s_epoch > 0],
            key=lambda m: m.s_epoch, reverse=True
        )
        for i, ms in enumerate(ranked):
            ms.rank = i + 1
            # Update streak
            if ms.rank <= PEB_K:
                ms.streak += 1
            else:
                ms.streak = 0
            # Compute PEB (Eq. 4)
            ms.peb = ScoringEngine.compute_peb(ms.rank, ms.streak)

    def set_weights(self):
        """Compute normalized weight vector and submit to chain."""
        weights = torch.zeros(self.metagraph.n)

        for uid, ms in self.miner_states.items():
            if uid < len(weights):
                # Weight = S_epoch * (1 + PEB) — same as emission formula denominator
                weights[uid] = ms.s_epoch * (1.0 + ms.peb)

        # Normalize to sum to 1
        total = weights.sum()
        if total > 0:
            weights = weights / total

        # Submit to chain
        success = self.subtensor.set_weights(
            netuid=self.config.netuid,
            wallet=self.wallet,
            uids=torch.arange(self.metagraph.n),
            weights=weights,
        )

        if success:
            bt.logging.info(f"Weights set successfully for epoch {self.epoch_id}")
        else:
            bt.logging.error("Failed to set weights on chain")
```

### 3.2 — validator/scoring.py (Full Scoring Pipeline)

Orchestrates the scoring pipeline using the MVP's `ScoringEngine`:

```python
class ValidatorScorer:
    """
    Wraps the MVP ScoringEngine for production use.
    Adds objective verification backends on top of the formula layer.
    """

    def __init__(self, config):
        self.engine = ScoringEngine()  # MVP engine — all formulas
        self.lean4 = Lean4Checker() if config.validator.lean4_enabled else None
        self.sandbox = CodeSandbox() if config.validator.sandbox_enabled else None
        self.math_checker = MathChecker()
        self.fact_checker = FactChecker()

    async def compute_dimensions(self, task: Task, response: ReasoningTask) -> DimensionScores:
        """
        Compute all 4 dimension scores for a miner's response.
        Maps to the Quality, Accuracy, Novelty, Efficiency dimensions in the whitepaper.
        """
        quality = self._score_quality(task, response)
        accuracy = await self._score_accuracy(task, response)
        novelty = self._score_novelty(task, response)
        efficiency = self._score_efficiency(task, response)
        return DimensionScores(quality, accuracy, novelty, efficiency)

    def _score_quality(self, task, response) -> float:
        """
        Quality (40% of CMS):
        - Step coherence: Do steps logically follow each other?
        - Completeness: Are all aspects of the problem addressed?
        - Depth: Sufficient detail per step?
        - Formal proof fragments present? (bonus for math/code)
        """
        steps = response.reasoning_steps or []
        if not steps:
            return 0.0

        # Step count vs difficulty expectation
        expected_steps = max(3, task.difficulty)
        step_ratio = min(1.0, len(steps) / expected_steps)

        # Average confidence
        avg_confidence = sum(s.get("confidence", 0) for s in steps) / len(steps)

        # Evidence presence
        evidence_ratio = sum(1 for s in steps if s.get("evidence")) / len(steps)

        # Proof fragment bonus
        proof_bonus = 0.1 if any(s.get("formal_proof_fragment") for s in steps) else 0.0

        return min(1.0, (0.3 * step_ratio) + (0.3 * avg_confidence) + (0.2 * evidence_ratio) + (0.2 + proof_bonus))

    async def _score_accuracy(self, task, response) -> float:
        """
        Accuracy (30% of CMS):
        Domain-specific automated checks → Eq. 11 objective scoring.
        """
        domain = Domain(task.domain) if isinstance(task.domain, str) else task.domain

        if domain == Domain.MATHEMATICS:
            checks = {}
            if self.lean4 and response.proof_artifact:
                checks["proof"] = await self.lean4.verify(response.proof_artifact)
            checks["numerical"] = self.math_checker.verify(
                task.problem, response.final_answer
            )
            checks["steps"] = self._verify_math_steps(response.reasoning_steps)
            weights = DOMAIN_CHECK_WEIGHTS[Domain.MATHEMATICS]

        elif domain == Domain.CODE:
            checks = {}
            if self.sandbox and response.code_artifact:
                checks["tests"] = await self.sandbox.run_tests(response.code_artifact)
                checks["static_analysis"] = await self.sandbox.lint(response.code_artifact)
            checks["formal"] = 0.5  # Default if no sandbox
            weights = DOMAIN_CHECK_WEIGHTS[Domain.CODE]

        # ... (other domains)

        return self.engine.compute_objective_score(checks, weights)

    def _score_novelty(self, task, response) -> float:
        """
        Novelty (15% of CMS):
        - Unique approach vs common solutions
        - Creative reasoning paths
        - Non-trivial insights
        """
        steps = response.reasoning_steps or []
        if not steps:
            return 0.0

        # Heuristic: longer, more varied reasoning → higher novelty
        avg_step_length = sum(len(s.get("reasoning", "")) for s in steps) / len(steps)
        length_score = min(1.0, avg_step_length / 500)

        # Unique terms ratio (rough diversity measure)
        all_words = " ".join(s.get("reasoning", "") for s in steps).split()
        diversity = len(set(all_words)) / max(1, len(all_words))

        return min(1.0, 0.5 * length_score + 0.5 * diversity)

    def _score_efficiency(self, task, response) -> float:
        """
        Efficiency (15% of CMS):
        - Solve time relative to timeout
        - Steps vs difficulty (conciseness)
        """
        time_ms = response.time_taken_ms or (task.timeout_seconds * 1000)
        timeout_ms = task.timeout_seconds * 1000

        # Faster = better, but don't reward instant (likely garbage)
        time_ratio = time_ms / timeout_ms
        if time_ratio < 0.01:  # Suspiciously fast
            time_score = 0.2
        elif time_ratio > 1.0:  # Timed out
            time_score = 0.0
        else:
            time_score = 1.0 - (time_ratio * 0.5)  # Linear penalty, max 0.5 deduction

        return min(1.0, time_score)
```

### 3.3 — validator/weight_setter.py

```python
class WeightSetter:
    """Compute and submit on-chain weights from epoch scores."""

    def __init__(self, subtensor, config):
        self.subtensor = subtensor
        self.config = config

    def compute_weights(self, miner_states: dict[int, MinerState], n: int) -> tuple:
        """
        Convert S_epoch + PEB into normalized weight vector.
        This is the core mapping from off-chain scoring → on-chain Yuma Consensus input.
        """
        uids = []
        weights = []

        for uid in range(n):
            if uid in miner_states and miner_states[uid].s_epoch > 0:
                w = miner_states[uid].s_epoch * (1.0 + miner_states[uid].peb)
                uids.append(uid)
                weights.append(w)

        if not weights:
            return torch.tensor([]), torch.tensor([])

        # Normalize
        weight_tensor = torch.FloatTensor(weights)
        weight_tensor = weight_tensor / weight_tensor.sum()

        return torch.tensor(uids), weight_tensor

    def submit(self, uids, weights) -> bool:
        """Submit weights to chain with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                success = self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=True,
                    wait_for_finalization=False,
                )
                if success:
                    return True
            except Exception as e:
                bt.logging.warning(f"Weight setting attempt {attempt+1} failed: {e}")
                time.sleep(5)
        return False
```

### 3.4 — validator/consensus.py

```python
def compute_consensus_score(
    validator_scores: list[tuple[float, float]],  # (score, stake)
    trim_delta: float = CONSENSUS_TRIM_DELTA,
) -> float:
    """
    Stake-weighted trimmed median (Eq. 12).
    Reuses ScoringEngine.compute_consensus_score from MVP.
    This wrapper adapts production validator data into the format the engine expects.
    """
    return ScoringEngine.compute_consensus_score(validator_scores, trim_delta)
```

### 3.5 — validator/trap_manager.py

```python
class TrapManager:
    """Inject trap problems with known ground-truth scores."""

    def __init__(self, trap_rate: float = TRAP_RATE):
        self.trap_rate = trap_rate
        self.trap_db = self._load_traps()

    def inject_traps(self, tasks: list[Task]) -> list[Task]:
        """Replace trap_rate fraction of tasks with traps."""
        n_traps = max(1, int(len(tasks) * self.trap_rate))
        trap_tasks = random.sample(self.trap_db, min(n_traps, len(self.trap_db)))

        # Replace last n_traps tasks
        for i in range(n_traps):
            tasks[-(i+1)] = trap_tasks[i]

        random.shuffle(tasks)
        return tasks

    def evaluate_trap_response(self, task: Task, response: ReasoningTask) -> float:
        """Compare response against ground truth. Returns score 0-1."""
        # For math traps: numerical comparison
        # For code traps: test case execution
        # For others: embedding similarity to known-correct answer
        ...
```

---

## Phase 4: Formal Verification & Sandboxed Execution <a name="phase-4"></a>

### 4.1 — verification/lean4_checker.py

```python
class Lean4Checker:
    """
    Verify Lean 4 proof artifacts submitted by miners.
    Requires: lean4 toolchain installed in validator environment.
    """

    def __init__(self, lean_path: str = "lean"):
        self.lean_path = lean_path
        self.timeout = 60  # seconds

    async def verify(self, proof_b64: str) -> float:
        """
        Decode proof artifact → write to temp .lean file → run lean4 → check exit code.
        Returns: 1.0 if proof compiles, 0.0 if it fails, 0.5 if timeout.
        """
        proof_text = base64.b64decode(proof_b64).decode("utf-8")

        with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
            f.write(proof_text)
            f.flush()

            try:
                result = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        self.lean_path, f.name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    ),
                    timeout=self.timeout,
                )
                stdout, stderr = await result.communicate()
                return 1.0 if result.returncode == 0 else 0.0
            except asyncio.TimeoutError:
                return 0.5
            finally:
                os.unlink(f.name)
```

### 4.2 — verification/code_sandbox.py

```python
class CodeSandbox:
    """
    Run miner code submissions in an isolated Docker container.
    Prevents: filesystem access, network access, fork bombs, resource exhaustion.
    """

    def __init__(self, image: str = "reasonforge-sandbox:latest"):
        self.image = image
        self.client = docker.from_env()
        self.timeout = 30
        self.memory_limit = "256m"
        self.cpu_period = 100000
        self.cpu_quota = 50000  # 50% of one core

    async def run_tests(self, code_b64: str) -> float:
        """Execute code in sandbox, run any included test cases."""
        code = base64.b64decode(code_b64).decode("utf-8")

        container = self.client.containers.run(
            self.image,
            command=["python3", "-c", code],
            detach=True,
            mem_limit=self.memory_limit,
            cpu_period=self.cpu_period,
            cpu_quota=self.cpu_quota,
            network_disabled=True,
            read_only=True,
            tmpfs={"/tmp": "size=64m"},
        )

        try:
            result = container.wait(timeout=self.timeout)
            logs = container.logs().decode("utf-8")

            if result["StatusCode"] == 0:
                # Parse test output for pass/fail counts
                return self._parse_test_results(logs)
            return 0.0
        except Exception:
            return 0.0
        finally:
            container.remove(force=True)

    async def lint(self, code_b64: str) -> float:
        """Run ruff/pylint on code, return quality score."""
        ...
```

### 4.3 — verification/math_checker.py

```python
class MathChecker:
    """Numerical and symbolic verification using SymPy."""

    def verify(self, problem: str, answer: str) -> float:
        """
        Try to:
        1. Parse the answer as a mathematical expression
        2. Evaluate numerically
        3. Compare against independent computation
        """
        try:
            # Extract numerical value from answer
            parsed = sympify(answer)
            # For known problem types, verify against computed solution
            # Returns 1.0 for correct, 0.0 for incorrect, 0.5 for unverifiable
            ...
        except:
            return 0.5  # Can't verify → neutral score
```

### 4.4 — docker/Dockerfile.sandbox

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy scipy sympy
RUN useradd -m sandbox
USER sandbox
WORKDIR /tmp
# No network, no filesystem beyond /tmp
```

---

## Phase 5: Embedding-Based Plagiarism Detection <a name="phase-5"></a>

### 5.1 — embeddings/similarity.py

Replace MVP's jaccard similarity with real embedding cosine similarity:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SimilarityDetector:
    """
    Detect plagiarism between miner submissions using sentence embeddings.
    Uses: sentence-transformers/all-MiniLM-L6-v2 (fast, 384-dim)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.history_embeddings: list[np.ndarray] = []  # Rolling buffer
        self.max_history = 5000

    def embed_submission(self, response: ReasoningTask) -> np.ndarray:
        """Encode reasoning chain into a single embedding vector."""
        steps_text = " ".join(
            s.get("reasoning", "") for s in (response.reasoning_steps or [])
        )
        full_text = f"{steps_text} {response.final_answer or ''}"
        return self.model.encode(full_text, normalize_embeddings=True)

    def check_against_batch(self, response: ReasoningTask,
                            other_responses: list[ReasoningTask]) -> float:
        """Return max cosine similarity against other responses in this batch."""
        if not other_responses:
            return 0.0

        target_emb = self.embed_submission(response)
        other_embs = np.array([self.embed_submission(r) for r in other_responses])

        # Cosine similarity (embeddings are normalized, so dot product = cosine)
        similarities = other_embs @ target_emb
        return float(np.max(similarities))

    def check_against_history(self, response: ReasoningTask) -> float:
        """Check against historical submissions (cross-epoch plagiarism)."""
        if not self.history_embeddings:
            return 0.0

        target_emb = self.embed_submission(response)
        history_matrix = np.array(self.history_embeddings[-self.max_history:])
        similarities = history_matrix @ target_emb
        return float(np.max(similarities))

    def add_to_history(self, response: ReasoningTask):
        """Store embedding for future cross-epoch checks."""
        emb = self.embed_submission(response)
        self.history_embeddings.append(emb)
        if len(self.history_embeddings) > self.max_history:
            self.history_embeddings = self.history_embeddings[-self.max_history:]
```

---

## Phase 6: Task Sourcing & Benchmark Database <a name="phase-6"></a>

### 6.1 — Benchmark JSON Format

Every benchmark file is a JSON array of task objects:

```json
[
  {
    "task_id": "math-algebra-001",
    "problem": "Prove that for all positive integers n, the sum 1 + 2 + ... + n = n(n+1)/2",
    "domain": "mathematics",
    "difficulty": 4,
    "timeout_seconds": 300,
    "ground_truth": "By mathematical induction...",
    "ground_truth_score": 0.95,
    "is_trap": false,
    "previously_unsolved": false,
    "tags": ["induction", "series", "algebra"],
    "source": "benchmark",
    "author": "ReasonForge Team",
    "created_at": "2026-01-15"
  }
]
```

### 6.2 — Benchmark Requirements

| Domain | Min Tasks | Difficulty Spread | Trap Tasks |
|--------|-----------|-------------------|------------|
| Mathematics | 100 | 1-10 evenly | 15 |
| Code | 100 | 1-10 evenly | 15 |
| Scientific | 80 | 1-10 evenly | 12 |
| Strategic | 60 | 1-10 evenly | 9 |
| Causal | 60 | 1-10 evenly | 9 |
| Ethical | 50 | 1-8 (no 9-10) | 8 |
| **Total** | **450** | | **68** |

### 6.3 — task_generator.py (Expanded)

Add to existing MVP task_generator:

```python
class TaskGenerator:
    """Production task generator with benchmark DB + synthetic + API ingestion."""

    def __init__(self, benchmark_dir: str = "benchmarks/"):
        self.benchmark_db = self._load_benchmarks(benchmark_dir)
        self.used_task_ids: set[str] = set()  # Avoid repeats within window

    def generate_epoch_tasks(self, count: int, trap_rate: float) -> list[Task]:
        """Generate a balanced set of tasks for one epoch."""
        n_traps = max(1, int(count * trap_rate))
        n_benchmark = count - n_traps

        tasks = []

        # 1. Sample benchmark tasks (balanced across domains)
        tasks += self._sample_balanced(n_benchmark)

        # 2. Add trap problems
        tasks += self._sample_traps(n_traps)

        # 3. Shuffle to hide traps
        random.shuffle(tasks)

        return tasks

    def ingest_api_task(self, request: dict) -> Task:
        """Accept an external task submission via the API gateway."""
        # Validate, assign difficulty, create Task object
        ...
```

---

## Phase 7: Persistent State & Recovery <a name="phase-7"></a>

### 7.1 — state/database.py

SQLite schema for persistent neuron state:

```python
class StateDatabase:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS miner_epochs (
        epoch_id INTEGER,
        miner_uid INTEGER,
        s_epoch REAL,
        peb REAL,
        rank INTEGER,
        streak INTEGER,
        tao_earned REAL,
        trap_penalty REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (epoch_id, miner_uid)
    );

    CREATE TABLE IF NOT EXISTS validator_epochs (
        epoch_id INTEGER,
        validator_uid INTEGER,
        vas REAL,
        reputation_multiplier REAL,
        tao_earned REAL,
        slashed REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (epoch_id, validator_uid)
    );

    CREATE TABLE IF NOT EXISTS task_results (
        task_id TEXT PRIMARY KEY,
        epoch_id INTEGER,
        domain TEXT,
        difficulty INTEGER,
        is_trap BOOLEAN,
        avg_cms REAL,
        best_miner_uid INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS submissions (
        submission_id TEXT PRIMARY KEY,
        task_id TEXT,
        miner_uid INTEGER,
        cms REAL,
        quality REAL,
        accuracy REAL,
        novelty REAL,
        efficiency REAL,
        submission_hash TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (task_id) REFERENCES task_results(task_id)
    );

    CREATE TABLE IF NOT EXISTS checkpoints (
        checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
        epoch_id INTEGER,
        state_blob TEXT,  -- JSON serialized state
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS embedding_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        epoch_id INTEGER,
        miner_uid INTEGER,
        task_id TEXT,
        embedding BLOB,  -- numpy array bytes
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_miner_epochs_uid ON miner_epochs(miner_uid);
    CREATE INDEX IF NOT EXISTS idx_submissions_miner ON submissions(miner_uid);
    CREATE INDEX IF NOT EXISTS idx_submissions_task ON submissions(task_id);
    """
```

### 7.2 — state/checkpoint.py

```python
class CheckpointManager:
    """Save and restore full neuron state for crash recovery."""

    def save(self, db: StateDatabase, epoch_id: int, state: dict):
        """Serialize state to JSON and store in checkpoints table."""
        ...

    def load_latest(self, db: StateDatabase) -> Optional[dict]:
        """Load most recent checkpoint for crash recovery."""
        ...

    def prune_old(self, db: StateDatabase, keep_last: int = 10):
        """Remove old checkpoints to save disk space."""
        ...
```

---

## Phase 8: API Gateway & External Access <a name="phase-8"></a>

### 8.1 — gateway/app.py

External-facing FastAPI application for users to submit tasks and query results:

```python
app = FastAPI(title="ReasonForge Gateway", version="0.1.0")

# Endpoints:
# POST /v1/tasks              — Submit a reasoning task (authenticated)
# GET  /v1/tasks/{task_id}    — Get task status and results
# GET  /v1/leaderboard        — Current miner rankings
# GET  /v1/stats              — Network statistics
# GET  /v1/health             — Health check
# WS   /v1/stream             — WebSocket for real-time epoch updates

@app.post("/v1/tasks", dependencies=[Depends(verify_api_key)])
async def submit_task(request: TaskSubmissionRequest):
    """Submit a reasoning task to the network."""
    # 1. Validate input
    # 2. Assign difficulty (if not provided)
    # 3. Queue for next epoch
    # 4. Return task_id for polling
    ...

@app.get("/v1/tasks/{task_id}")
async def get_task_result(task_id: str):
    """Poll for task results."""
    ...

@app.get("/v1/leaderboard")
async def get_leaderboard(domain: Optional[str] = None, limit: int = 20):
    """Get current miner rankings."""
    ...
```

### 8.2 — gateway/auth.py

```python
class APIKeyManager:
    """API key management with usage tracking and rate limits."""

    def __init__(self, db: StateDatabase):
        self.db = db

    def create_key(self, owner: str, tier: str = "free") -> str:
        """Generate new API key. Tiers: free (100 req/mo), pro (10k), enterprise (unlimited)."""
        ...

    def verify_key(self, key: str) -> Optional[APIKeyInfo]:
        """Validate key and check rate limits."""
        ...

    def track_usage(self, key: str, task_id: str):
        """Record API usage for billing."""
        ...
```

### 8.3 — gateway/schemas.py

```python
class TaskSubmissionRequest(BaseModel):
    problem: str = Field(..., min_length=10, max_length=10000)
    domain: Optional[str] = None  # Auto-detect if not provided
    difficulty: Optional[int] = Field(None, ge=1, le=10)
    timeout_seconds: Optional[int] = Field(300, ge=30, le=600)
    callback_url: Optional[str] = None  # Webhook for async notification

class TaskResultResponse(BaseModel):
    task_id: str
    status: str  # "queued"|"processing"|"completed"|"failed"
    result: Optional[dict] = None
    best_answer: Optional[str] = None
    confidence: Optional[float] = None
    reasoning_steps: Optional[list[dict]] = None
    processing_time_ms: Optional[int] = None
```

---

## Phase 9: Monitoring & Observability <a name="phase-9"></a>

### 9.1 — monitoring/metrics.py

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class MetricsCollector:
    """Prometheus metrics for subnet monitoring."""

    def __init__(self, neuron_type: str, uid: int):
        prefix = f"reasonforge_{neuron_type}"

        # Counters
        self.tasks_processed = Counter(f"{prefix}_tasks_total", "Total tasks processed", ["domain", "difficulty"])
        self.epochs_completed = Counter(f"{prefix}_epochs_total", "Epochs completed")
        self.traps_injected = Counter(f"{prefix}_traps_total", "Trap problems injected")
        self.breakthroughs = Counter(f"{prefix}_breakthroughs_total", "Breakthrough solutions")
        self.plagiarism_detected = Counter(f"{prefix}_plagiarism_total", "Plagiarism detections")
        self.weight_set_failures = Counter(f"{prefix}_weight_failures_total", "Weight setting failures")

        # Histograms
        self.task_latency = Histogram(f"{prefix}_task_latency_seconds", "Task processing time", ["domain"])
        self.cms_distribution = Histogram(f"{prefix}_cms_score", "CMS score distribution", buckets=[0.1*i for i in range(11)])
        self.vas_distribution = Histogram(f"{prefix}_vas_score", "VAS score distribution", buckets=[0.1*i for i in range(11)])

        # Gauges
        self.current_epoch = Gauge(f"{prefix}_current_epoch", "Current epoch number")
        self.active_miners = Gauge(f"{prefix}_active_miners", "Number of active miners")
        self.avg_cms = Gauge(f"{prefix}_avg_cms", "Average CMS this epoch")
        self.total_emission = Gauge(f"{prefix}_total_emission_tao", "Total TAO emitted")
        self.top_miner_score = Gauge(f"{prefix}_top_miner_score", "Highest S_epoch")

        # Start metrics server
        start_http_server(9090 + uid)
```

### 9.2 — monitoring/logger.py

```python
import structlog

def setup_logging(neuron_type: str, uid: int, debug: bool = False):
    """Configure structured JSON logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    logger = structlog.get_logger()
    return logger.bind(neuron_type=neuron_type, uid=uid)
```

### 9.3 — Grafana Dashboards

Create two dashboard JSON files:

**subnet_overview.json** — Panels:
- Total tasks processed (counter)
- Average CMS per epoch (timeseries)
- Miner count (gauge)
- Emission distribution (pie chart)
- Trap detection rate (timeseries)
- Weight setting success rate (timeseries)

**miner_performance.json** — Panels:
- Per-miner S_epoch over time (multi-line)
- CMS dimension breakdown (stacked bar)
- PEB distribution (bar)
- Streak lengths (table)
- Plagiarism events (annotations)

---

## Phase 10: Security Hardening <a name="phase-10"></a>

### 10.1 — security/sanitizer.py

```python
class InputSanitizer:
    """Validate and sanitize all inputs from miners and external API."""

    MAX_STEP_LENGTH = 10000      # chars per reasoning step
    MAX_STEPS = 50               # max steps per submission
    MAX_ANSWER_LENGTH = 50000    # chars
    MAX_PROOF_SIZE = 1_000_000   # bytes (1MB)
    MAX_CODE_SIZE = 500_000      # bytes (500KB)

    @staticmethod
    def sanitize_submission(response: ReasoningTask) -> ReasoningTask:
        """Validate all miner-provided fields."""
        # 1. Truncate oversized fields
        if response.reasoning_steps and len(response.reasoning_steps) > InputSanitizer.MAX_STEPS:
            response.reasoning_steps = response.reasoning_steps[:InputSanitizer.MAX_STEPS]

        # 2. Strip potential injection in reasoning text
        if response.reasoning_steps:
            for step in response.reasoning_steps:
                step["reasoning"] = step.get("reasoning", "")[:InputSanitizer.MAX_STEP_LENGTH]

        # 3. Validate proof artifact size
        if response.proof_artifact:
            decoded = base64.b64decode(response.proof_artifact)
            if len(decoded) > InputSanitizer.MAX_PROOF_SIZE:
                response.proof_artifact = None

        # 4. Validate code artifact size
        if response.code_artifact:
            decoded = base64.b64decode(response.code_artifact)
            if len(decoded) > InputSanitizer.MAX_CODE_SIZE:
                response.code_artifact = None

        return response
```

### 10.2 — security/rate_guard.py

```python
class RateGuard:
    """Per-UID rate limiting to prevent DoS."""

    def __init__(self, max_requests_per_minute: int = 10):
        self.limits: dict[int, list[float]] = defaultdict(list)
        self.max_rpm = max_requests_per_minute

    def check(self, uid: int) -> bool:
        """Returns True if request is allowed."""
        now = time.time()
        self.limits[uid] = [t for t in self.limits[uid] if now - t < 60]
        if len(self.limits[uid]) >= self.max_rpm:
            return False
        self.limits[uid].append(now)
        return True
```

### 10.3 — security/anomaly.py

```python
class AnomalyDetector:
    """Detect suspicious miner behavior patterns."""

    def check_timing_anomaly(self, time_ms: int, difficulty: int) -> bool:
        """Flag if solve time is unrealistically fast for difficulty."""
        min_expected = difficulty * 500  # ms
        return time_ms < min_expected

    def check_score_manipulation(self, cms_history: list[float]) -> bool:
        """Flag if CMS scores are suspiciously consistent (gaming)."""
        if len(cms_history) < 5:
            return False
        variance = statistics.variance(cms_history)
        return variance < 0.001  # Nearly identical scores = suspicious

    def check_collusion(self, submissions: list[ReasoningTask]) -> list[tuple[int, int, float]]:
        """Detect colluding miners with near-identical submissions."""
        # Compare all pairs, flag if similarity > threshold
        ...
```

---

## Phase 11: Docker & Deployment <a name="phase-11"></a>

### 11.1 — docker/Dockerfile.miner

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt requirements-miner.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-miner.txt

# App code
COPY reasonforge/ reasonforge/
COPY neurons/miner.py neurons/

# Wallet mount point
VOLUME /root/.bittensor/wallets

# Metrics port
EXPOSE 9091

# Entry
ENTRYPOINT ["python", "neurons/miner.py"]
CMD ["--netuid", "XX", "--subtensor.network", "finney"]
```

### 11.2 — docker/Dockerfile.validator

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps (includes docker-cli for sandbox)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl docker.io && rm -rf /var/lib/apt/lists/*

# Install Lean 4 (optional, for math verification)
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:stable
ENV PATH="/root/.elan/bin:$PATH"

# Python deps
COPY requirements.txt requirements-validator.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-validator.txt

# App code
COPY reasonforge/ reasonforge/
COPY neurons/validator.py neurons/
COPY benchmarks/ benchmarks/

# Volumes
VOLUME /root/.bittensor/wallets
VOLUME /app/state

# Ports: metrics + API
EXPOSE 9092 8092

ENTRYPOINT ["python", "neurons/validator.py"]
CMD ["--netuid", "XX", "--subtensor.network", "finney", "--validator.sandbox_enabled", "--validator.lean4_enabled"]
```

### 11.3 — docker-compose.yml

```yaml
version: "3.8"

services:
  validator:
    build:
      context: .
      dockerfile: docker/Dockerfile.validator
    volumes:
      - ~/.bittensor/wallets:/root/.bittensor/wallets:ro
      - validator-state:/app/state
      - /var/run/docker.sock:/var/run/docker.sock  # For sandbox containers
    environment:
      - NETUID=${NETUID}
      - SUBTENSOR_NETWORK=${SUBTENSOR_NETWORK:-finney}
    ports:
      - "9092:9092"  # Metrics
      - "8092:8092"  # API
    restart: unless-stopped

  miner:
    build:
      context: .
      dockerfile: docker/Dockerfile.miner
    volumes:
      - ~/.bittensor/wallets:/root/.bittensor/wallets:ro
    environment:
      - NETUID=${NETUID}
      - SUBTENSOR_NETWORK=${SUBTENSOR_NETWORK:-finney}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "9091:9091"
      - "8091:8091"
    restart: unless-stopped

  gateway:
    build:
      context: .
      dockerfile: docker/Dockerfile.gateway
    volumes:
      - gateway-data:/app/data
    ports:
      - "8000:8000"
    restart: unless-stopped

  sandbox:
    build:
      context: .
      dockerfile: docker/Dockerfile.sandbox
    # Not a service — built as an image for validator to spawn containers from

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}

volumes:
  validator-state:
  gateway-data:
  prometheus-data:
  grafana-data:
```

### 11.4 — scripts/run_localnet.sh

```bash
#!/bin/bash
# Start a local subtensor for development & testing
# Requires: subtensor repo cloned and built

set -e

echo "Starting local subtensor..."
cd ~/subtensor
./scripts/localnet.sh &

sleep 10

echo "Creating wallets..."
btcli wallet new_coldkey --wallet.name owner --no_password
btcli wallet new_coldkey --wallet.name validator --no_password
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
btcli wallet new_coldkey --wallet.name miner --no_password
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

echo "Funding wallets..."
btcli wallet faucet --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli wallet faucet --wallet.name validator --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli wallet faucet --wallet.name miner --subtensor.chain_endpoint ws://127.0.0.1:9946

echo "Creating subnet..."
btcli subnets create --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946

echo "Registering neurons..."
btcli subnets register --wallet.name miner --wallet.hotkey default --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli subnets register --wallet.name validator --wallet.hotkey default --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946

echo "Staking to validator..."
btcli stake add --wallet.name validator --wallet.hotkey default --amount 100 --subtensor.chain_endpoint ws://127.0.0.1:9946

echo "Registering validator on root subnet..."
btcli root register --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli root boost --netuid 1 --increase 1 --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946

echo "✅ Localnet ready. NETUID=1"
echo "Run miner:     python neurons/miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name miner --wallet.hotkey default"
echo "Run validator:  python neurons/validator.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name validator --wallet.hotkey default"
```

---

## Phase 12: CI/CD & Testing Infrastructure <a name="phase-12"></a>

### 12.1 — .github/workflows/test.yml

```yaml
name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --tb=short -x

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install ruff mypy
      - run: ruff check reasonforge/ neurons/
      - run: mypy reasonforge/ --ignore-missing-imports

  integration:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[all]"
      - run: pytest tests/test_integration_local.py -v --timeout=120
```

### 12.2 — Test Categories

| Test File | What It Tests | Depends On |
|-----------|---------------|------------|
| `test_engine.py` | All 13 formulas (MVP) | Nothing |
| `test_types.py` | Constants, dataclasses (MVP) | Nothing |
| `test_simulator.py` | Epoch simulation (MVP) | engine, types |
| `test_protocol.py` | Synapse serialization, hash verification | protocol.py |
| `test_scoring.py` | Dimension scoring, quality/accuracy/novelty/efficiency | scorer, engine |
| `test_verification.py` | Lean4 checker, code sandbox, math checker | verification/ |
| `test_embeddings.py` | Cosine similarity, plagiarism detection | embeddings/ |
| `test_state.py` | SQLite CRUD, checkpoint save/load, migration | state/ |
| `test_gateway.py` | API endpoints, auth, rate limiting | gateway/ |
| `test_security.py` | Input sanitization, rate guard, anomaly detection | security/ |
| `test_integration_local.py` | Full miner↔validator exchange on localnet | Everything |

### 12.3 — test_integration_local.py (Key Test)

```python
class TestLocalIntegration:
    """
    End-to-end test: validator sends task → miner solves → validator scores → weights set.
    Runs without a real blockchain — mocks subtensor.
    """

    def test_full_epoch_cycle(self):
        """One complete epoch with mocked blockchain."""
        # 1. Create mock subtensor + metagraph
        # 2. Initialize validator and miner
        # 3. Validator generates tasks
        # 4. Validator queries miner via localhost axon
        # 5. Miner responds with reasoning chain
        # 6. Validator scores responses
        # 7. Validator computes weights
        # 8. Verify: weights are normalized, best miner gets highest weight
        # 9. Verify: emission conservation
        # 10. Verify: state persisted to SQLite

    def test_trap_detection_live(self):
        """Trap problem correctly penalizes a deliberately bad miner."""
        ...

    def test_plagiarism_detection_live(self):
        """Two miners submitting identical answers get flagged."""
        ...

    def test_crash_recovery(self):
        """Kill validator mid-epoch, restart, verify state restored."""
        ...
```

---

## Phase 13: Documentation & SDK <a name="phase-13"></a>

### Required Documents

| Document | Audience | Content |
|----------|----------|---------|
| `README.md` | Everyone | Project overview, quick start, architecture summary |
| `docs/ARCHITECTURE.md` | Developers | System design, data flow, component interactions |
| `docs/PROTOCOL.md` | Subnet devs | Wire protocol (Synapse types), message flow, serialization |
| `docs/MINER_GUIDE.md` | Miners | How to set up and run a miner, LLM backend selection, GPU requirements, earnings optimization |
| `docs/VALIDATOR_GUIDE.md` | Validators | How to run a validator, stake requirements, Lean4 setup, monitoring |
| `docs/API_REFERENCE.md` | API consumers | Endpoint specs, auth, rate limits, examples |
| `docs/DEPLOYMENT.md` | Operators | Docker deployment, environment vars, scaling, backup |
| `docs/SECURITY.md` | Security reviewers | Threat model, anti-adversarial mechanisms, input validation |
| `docs/BENCHMARKS.md` | Contributors | How to add benchmark tasks, format spec, review process |

### MINER_GUIDE.md — Key Sections

```markdown
## Requirements
- Python 3.10+
- GPU recommended (for local LLM backend)
- Bittensor wallet with registered hotkey
- Registration cost: ~0.1 TAO

## Quick Start
pip install -e ".[miner]"
python neurons/miner.py \
  --netuid <YOUR_NETUID> \
  --subtensor.network finney \
  --wallet.name my_miner \
  --wallet.hotkey default \
  --miner.backend openai \
  --miner.model gpt-4o

## Backend Options
| Backend | Flag | Requirements | Performance |
|---------|------|-------------|-------------|
| OpenAI | `--miner.backend openai` | API key | High (o1-level) |
| Anthropic | `--miner.backend anthropic` | API key | High |
| Local | `--miner.backend local` | GPU + model | Variable |
| Agent | `--miner.backend agent` | LangGraph + LLM | Highest potential |

## Earning Optimization
1. Specialize in high-difficulty tasks (2× multiplier at difficulty 10)
2. Include formal proofs when possible (quality bonus)
3. Maintain consistency for PEB streak bonuses
4. Avoid plagiarism (0.5× penalty if detected)
```

---

## 16. Subnet Hyperparameters <a name="subnet-hyperparameters"></a>

Set via `btcli subnets hyperparameters` or in subnet registration:

```yaml
# min_compute.yml — Minimum compute requirements
min_compute:
  miner:
    cpu: 4
    ram_gb: 16
    storage_gb: 50
    gpu: optional  # Depends on LLM backend
    bandwidth_mbps: 100
  validator:
    cpu: 8
    ram_gb: 32
    storage_gb: 100
    gpu: recommended  # For embedding model
    bandwidth_mbps: 200

# Subnet hyperparameters
subnet:
  tempo: 360                    # Blocks per epoch (~72 minutes)
  immunity_period: 7200         # New neuron protection (~24 hours)
  max_miners: 192
  max_validators: 64
  min_validator_stake: 1000     # TAO
  weights_rate_limit: 100       # Blocks between weight updates
  weights_version_key: 1
  adjustment_alpha: 0.7
  difficulty: 10000000          # POW registration difficulty
  registration_cost: 0.1        # TAO burn registration
```

---

## 17. Environment Variables <a name="environment-variables"></a>

```bash
# .env.example

# ── Bittensor ──
NETUID=XX                              # Your subnet UID (assigned after registration)
SUBTENSOR_NETWORK=finney               # finney | test | local
SUBTENSOR_CHAIN_ENDPOINT=              # Custom endpoint (optional)

# ── Wallet ──
WALLET_NAME=my_wallet
WALLET_HOTKEY=default

# ── Miner ──
MINER_BACKEND=openai                   # openai | anthropic | local | agent
MINER_MODEL=gpt-4o                     # Model identifier
OPENAI_API_KEY=sk-...                  # If using OpenAI backend
ANTHROPIC_API_KEY=sk-ant-...           # If using Anthropic backend
MINER_PORT=8091
MINER_MAX_CONCURRENT=4

# ── Validator ──
VALIDATOR_PORT=8092
VALIDATOR_EPOCH_LENGTH=360
VALIDATOR_TASKS_PER_EPOCH=12
VALIDATOR_TRAP_RATE=0.15
VALIDATOR_TIMEOUT=300
VALIDATOR_SANDBOX_ENABLED=true
VALIDATOR_LEAN4_ENABLED=true
VALIDATOR_EMBEDDING_MODEL=all-MiniLM-L6-v2

# ── Gateway ──
GATEWAY_PORT=8000
GATEWAY_API_KEY_SECRET=your-secret-key

# ── Monitoring ──
PROMETHEUS_PORT=9090
GRAFANA_PASSWORD=admin

# ── State ──
STATE_DB_PATH=state/reasonforge.db
```

---

## 18. Build Order <a name="build-order"></a>

**Prerequisites:** MVP codebase complete and all tests passing.

```
PHASE 1 — Protocol Layer (Days 1-2)
  Step 1:   Write reasonforge/protocol.py (all 3 Synapse classes)
  Step 2:   Write reasonforge/base/config.py (CLI args)
  Step 3:   Write reasonforge/base/neuron.py (BaseNeuron)
  Step 4:   Write tests/test_protocol.py — verify Synapse serialization
  Step 5:   Run tests: pytest tests/test_protocol.py -v

PHASE 2 — Miner Neuron (Days 3-5)
  Step 6:   Write reasonforge/miner/backends/base.py (LLMBackend ABC)
  Step 7:   Write reasonforge/miner/backends/openai_backend.py
  Step 8:   Write reasonforge/miner/backends/anthropic_backend.py
  Step 9:   Write reasonforge/miner/domain_router.py (6 domain prompts)
  Step 10:  Write reasonforge/miner/reasoning.py (ReasoningEngine)
  Step 11:  Write neurons/miner.py (full entry point with Axon)

PHASE 3 — Validator Neuron (Days 6-9)
  Step 12:  Write reasonforge/validator/task_manager.py
  Step 13:  Write reasonforge/validator/trap_manager.py
  Step 14:  Write reasonforge/validator/objective_scorer.py
  Step 15:  Write reasonforge/validator/consensus.py (wraps MVP engine)
  Step 16:  Write reasonforge/validator/scoring.py (orchestrator)
  Step 17:  Write reasonforge/validator/weight_setter.py
  Step 18:  Write neurons/validator.py (full entry point with epoch loop)
  Step 19:  Write tests/test_scoring.py — verify scoring pipeline
  Step 20:  Run: pytest tests/test_scoring.py -v

PHASE 4 — Verification Backends (Days 10-12)
  Step 21:  Write reasonforge/verification/math_checker.py (SymPy)
  Step 22:  Write reasonforge/verification/code_sandbox.py (Docker)
  Step 23:  Write reasonforge/verification/lean4_checker.py
  Step 24:  Write reasonforge/verification/fact_checker.py
  Step 25:  Write docker/Dockerfile.sandbox
  Step 26:  Write tests/test_verification.py
  Step 27:  Run: pytest tests/test_verification.py -v

PHASE 5 — Plagiarism Detection (Day 13)
  Step 28:  Write reasonforge/embeddings/similarity.py
  Step 29:  Write tests/test_embeddings.py
  Step 30:  Run: pytest tests/test_embeddings.py -v

PHASE 6 — Benchmark Database (Days 14-16)
  Step 31:  Create benchmark JSON files (450+ tasks across 6 domains)
  Step 32:  Expand reasonforge/task_generator.py (DB loading, balanced sampling)
  Step 33:  Write scripts/benchmark_import.py
  Step 34:  Write scripts/generate_traps.py (68 trap problems)

PHASE 7 — Persistence (Days 17-18)
  Step 35:  Write reasonforge/state/database.py (SQLite schema + CRUD)
  Step 36:  Write reasonforge/state/checkpoint.py
  Step 37:  Write reasonforge/state/migrations.py
  Step 38:  Write tests/test_state.py
  Step 39:  Run: pytest tests/test_state.py -v

PHASE 8 — API Gateway (Days 19-20)
  Step 40:  Write reasonforge/gateway/schemas.py
  Step 41:  Write reasonforge/gateway/auth.py
  Step 42:  Write reasonforge/gateway/rate_limiter.py
  Step 43:  Write reasonforge/gateway/billing.py
  Step 44:  Write reasonforge/gateway/app.py
  Step 45:  Write tests/test_gateway.py
  Step 46:  Run: pytest tests/test_gateway.py -v

PHASE 9 — Monitoring (Days 21-22)
  Step 47:  Write reasonforge/monitoring/metrics.py (Prometheus)
  Step 48:  Write reasonforge/monitoring/logger.py (structlog)
  Step 49:  Write reasonforge/monitoring/health.py
  Step 50:  Create monitoring/prometheus.yml
  Step 51:  Create monitoring/grafana/dashboards/*.json

PHASE 10 — Security (Day 23)
  Step 52:  Write reasonforge/security/sanitizer.py
  Step 53:  Write reasonforge/security/rate_guard.py
  Step 54:  Write reasonforge/security/anomaly.py
  Step 55:  Write tests/test_security.py
  Step 56:  Run: pytest tests/test_security.py -v

PHASE 11 — Docker & Deployment (Days 24-25)
  Step 57:  Write docker/Dockerfile.miner
  Step 58:  Write docker/Dockerfile.validator
  Step 59:  Write docker/Dockerfile.gateway
  Step 60:  Write docker/docker-compose.yml
  Step 61:  Write docker/docker-compose.localnet.yml
  Step 62:  Write docker/docker-compose.monitoring.yml
  Step 63:  Write scripts/setup_wallets.sh
  Step 64:  Write scripts/register_subnet.sh
  Step 65:  Write scripts/register_neurons.sh
  Step 66:  Write scripts/run_localnet.sh

PHASE 12 — CI/CD (Day 26)
  Step 67:  Write .github/workflows/test.yml
  Step 68:  Write .github/workflows/lint.yml
  Step 69:  Write .github/workflows/build-docker.yml

PHASE 13 — Documentation (Days 27-28)
  Step 70:  Update README.md (production sections)
  Step 71:  Write docs/ARCHITECTURE.md
  Step 72:  Write docs/PROTOCOL.md
  Step 73:  Write docs/MINER_GUIDE.md
  Step 74:  Write docs/VALIDATOR_GUIDE.md
  Step 75:  Write docs/API_REFERENCE.md
  Step 76:  Write docs/DEPLOYMENT.md
  Step 77:  Write docs/SECURITY.md
  Step 78:  Write docs/BENCHMARKS.md
  Step 79:  Write min_compute.yml

INTEGRATION & VERIFICATION (Days 29-30)
  Step 80:  Write tests/test_integration_local.py
  Step 81:  Run full test suite: pytest tests/ -v
  Step 82:  Run: docker-compose -f docker/docker-compose.localnet.yml up
  Step 83:  Verify miner↔validator exchange on localnet
  Step 84:  Verify weights are set on local chain
  Step 85:  Verify state persistence across restart
  Step 86:  Verify metrics appear in Prometheus/Grafana
  Step 87:  Verify API gateway responds correctly
  Step 88:  Run security audit: input fuzzing, oversized payloads, invalid hashes
  Step 89:  Final: ruff check + mypy on entire codebase
  Step 90:  Tag v0.1.0 release
```

---

## 19. Success Criteria <a name="success-criteria"></a>

### Unit Tests (must all pass)
- [ ] All 13 MVP formula tests still pass
- [ ] Synapse serialization roundtrip works
- [ ] Scoring pipeline produces correct dimension scores
- [ ] Lean4 checker verifies valid proof, rejects invalid
- [ ] Code sandbox executes safely, returns test results
- [ ] Embedding similarity detects plagiarism > 0.95 threshold
- [ ] SQLite state saves and loads correctly
- [ ] API gateway auth rejects invalid keys
- [ ] Input sanitizer truncates oversized submissions
- [ ] Anomaly detector flags suspiciously fast responses

### Integration Tests (must all pass)
- [ ] Miner Axon starts and serves ReasoningTask Synapse
- [ ] Validator queries miner and receives valid response
- [ ] Full epoch cycle completes: tasks → scoring → weights
- [ ] Weights are normalized and submitted to chain
- [ ] Emission conservation holds (within rounding)
- [ ] Trap problems correctly penalize low-quality miners
- [ ] Plagiarism detection works across submissions in same batch
- [ ] State persists across validator restart
- [ ] Multiple miners compete, best scores highest weight

### Localnet Tests
- [ ] `run_localnet.sh` completes without errors
- [ ] Miner and validator register UIDs on local chain
- [ ] Validator sets weights after first epoch
- [ ] Emissions flow to miner after subnet tempo
- [ ] `btcli subnets metagraph --netuid 1` shows correct data

### Docker Tests
- [ ] All 4 Docker images build successfully
- [ ] `docker-compose up` starts full stack
- [ ] Sandbox container runs isolated code safely
- [ ] Prometheus scrapes metrics from both neurons
- [ ] Grafana dashboards render correctly

### Security Tests
- [ ] Oversized submissions are truncated, not crashed
- [ ] Invalid submission hashes are detected and penalized
- [ ] Rate limiting prevents DoS from single UID
- [ ] Miner blacklists non-validator callers
- [ ] Code sandbox prevents filesystem/network access

### Documentation
- [ ] README has working quick-start commands
- [ ] Miner guide covers all 4 backends
- [ ] Validator guide covers Lean4 + sandbox setup
- [ ] API reference documents all endpoints with examples
- [ ] Deployment guide covers Docker + bare metal

---

## Appendix A: Dependency Versions

```
# Core
bittensor>=10.0.1
torch>=2.0.0
numpy>=1.24.0
pydantic>=2.0.0
structlog>=23.0.0

# Miner
openai>=1.0.0
anthropic>=0.20.0
transformers>=4.35.0       # For local backend
vllm>=0.3.0                # Optional: fast local inference
langchain>=0.1.0            # For agent backend
langgraph>=0.0.10           # For agent backend

# Validator
sentence-transformers>=2.2.0
sympy>=1.12
docker>=7.0.0               # For code sandbox
prometheus-client>=0.19.0

# Gateway
fastapi>=0.100.0
uvicorn>=0.23.0
python-jose>=3.3.0          # JWT
passlib>=1.7.4              # Password hashing

# Dev
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-timeout>=2.1.0
ruff>=0.1.0
mypy>=1.6.0
```

---

## Appendix B: Key Differences from MVP

| Aspect | MVP | Production |
|--------|-----|------------|
| Miners | Statistical profiles (probabilistic) | Real LLMs via API/local inference |
| Validators | Simulated noise/bias profiles | Real scoring pipeline with verification |
| Network | None — in-memory simulation | Bittensor Axon/Dendrite over TCP |
| Scoring | All in ScoringEngine (kept!) | ScoringEngine + verification backends |
| Weights | Simulated emission distribution | On-chain via `subtensor.set_weights()` |
| State | In-memory, lost on exit | SQLite + checkpoint recovery |
| Plagiarism | Jaccard similarity | Sentence-transformer cosine similarity |
| Security | None needed | Full input sanitization, rate limiting |
| Monitoring | CLI print statements | Prometheus + Grafana + structured logging |
| Deployment | `python -m reasonforge.run` | Docker Compose + systemd |
| Testing | Unit tests only | Unit + integration + localnet + security |

---

*End of production build plan. This document assumes the MVP from PLAN.md is complete and transforms it into a deployable Bittensor subnet.*
