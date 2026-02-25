<div align="center">

# ReasonForge

**The Decentralized Marketplace for Verifiable Intelligence**
*A Bittensor Subnet Proposal -- Subnet Ideathon Round I*

[Live Dashboard](https://reasonforge-app.vercel.app) | [GitHub Source](https://github.com/unspecifiedcoder/ReasonForge)

</div>

---

## What is ReasonForge?

ReasonForge is a Bittensor subnet proposal for a decentralized marketplace for verifiable multi-step reasoning. Unlike existing AI subnets that reward raw generation, ReasonForge incentivizes **structured, auditable reasoning chains** across mathematics, code, science, strategy, causal inference, and ethics.

The protocol implements 13 mathematically rigorous formulas covering scoring, emissions, trap detection, plagiarism prevention, and validator accountability. Every answer comes with a traceable reasoning chain that can be independently verified, making it suitable for high-stakes deployment in legal, financial, and scientific markets.

---

## Quick Start

### Run the Simulator
```bash
pip install -e .
python -m reasonforge.run --epochs 10 --emission 100 --output results.json --seed 42
```

### Launch the API
```bash
pip install fastapi uvicorn
uvicorn api.server:app --reload --port 8000
```

### Run the Dashboard
```bash
npm install
npm run dev
```

### Run Tests
```bash
pytest tests/ -v
```

---

## Architecture

The MVP consists of 7 components:

| Component | Description |
|-----------|-------------|
| **Core Engine** | All 13 whitepaper formulas as stateless Python functions |
| **Epoch Simulator** | Full lifecycle simulation with 12 miners and 6 validators |
| **Interactive Dashboard** | React/TypeScript visualization with embedded JS simulation engine |
| **CLI Runner** | Terminal-based multi-epoch simulation with rich output |
| **Test Suite** | 67 unit and integration tests validating every formula |
| **API Server** | FastAPI REST endpoint serving simulation data |
| **Plagiarism Detector** | Jaccard similarity-based duplicate submission detection |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed technical documentation.

---

## Mechanism Design — 13 Formulas

| # | Formula | Description |
|---|---------|-------------|
| Eq.1 | `E = 0.90*E_miner + 0.10*E_validator` | Emission split between miners and validators |
| Eq.2 | `CMS = 0.40*Q + 0.30*A + 0.15*N + 0.15*Eff` | Composite Miner Score across 4 dimensions |
| Eq.3 | `S_epoch = mean(CMS * D(t)) * trap_penalty` | Difficulty-weighted epoch score |
| Eq.4 | `PEB = 0.20 * (1/rank) * sqrt(min(streak, 10))` | Persistent Excellence Bonus for top miners |
| Eq.5 | `R(m) = E_miner * [S*(1+PEB)] / sum[S*(1+PEB)]` | Final miner reward (emission-conserving) |
| Eq.6 | `CMS_bt = CMS * 2.0` | Breakthrough multiplier for unsolved tasks |
| Eq.7 | `VAS = 1 - mean(|v_score - consensus|)` | Validator Accuracy Score |
| Eq.8 | `R_v = E_val * [VAS*stake*rep] / sum[...]` | Validator reward (stake + reputation weighted) |
| Eq.9 | `penalty = max(0, avg_trap / 0.30)` | Trap penalty for poor performance on known tasks |
| Eq.10 | `slash = 0.05 * stake * (0.60 - VAS_7d)^2` | Quadratic slashing for inaccurate validators |
| Eq.11 | `O_score = sum(w_k * check_k)` | Domain-specific objective score |
| Eq.12 | `C_score = TrimmedMedian(stake-weighted)` | Consensus score via trimmed median |
| Eq.13 | `Final = 0.60*O + 0.40*C` | Blended final score |

---

## Protocol Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `W_QUALITY` | 0.40 | CMS quality weight |
| `W_ACCURACY` | 0.30 | CMS accuracy weight |
| `W_NOVELTY` | 0.15 | CMS novelty weight |
| `W_EFFICIENCY` | 0.15 | CMS efficiency weight |
| `EMISSION_MINER_SHARE` | 0.90 | 90% of emissions to miners |
| `EMISSION_VALIDATOR_SHARE` | 0.10 | 10% of emissions to validators |
| `PEB_ALPHA` | 0.20 | PEB coefficient |
| `PEB_K` | 10 | Top-K miners eligible for PEB |
| `BREAKTHROUGH_MULTIPLIER` | 2.0 | Multiplier for solving unsolved tasks |
| `TRAP_RATE` | 0.15 | 15% of tasks are traps |
| `VAS_SLASH_THRESHOLD` | 0.60 | VAS below this triggers slashing |
| `SIMILARITY_THRESHOLD` | 0.95 | Plagiarism detection threshold |

---

## Project Structure

```
reasonforge/
├── README.md
├── PLAN.md                     # Build specification
├── pyproject.toml              # Python project config
├── requirements.txt
├── .gitignore
├── app.tsx                     # Interactive dashboard (React/TypeScript)
├── main.tsx                    # React entry point
├── index.html / index.css      # Frontend assets
├── package.json                # Node.js dependencies
│
├── reasonforge/                # Core Python package
│   ├── __init__.py
│   ├── types.py                # Data structures + protocol constants
│   ├── engine.py               # Scoring engine (13 formulas)
│   ├── simulator.py            # Epoch simulator
│   ├── plagiarism.py           # Similarity detection
│   ├── task_generator.py       # Task templates
│   └── run.py                  # CLI entry point
│
├── api/                        # FastAPI server
│   └── server.py               # REST API
│
├── dashboard/                  # Standalone dashboard component
│   └── App.jsx
│
├── tests/                      # Test suite (67 tests)
│   ├── test_engine.py
│   ├── test_simulator.py
│   └── test_types.py
│
├── simulation/                 # Output directory
│   └── .gitkeep
│
└── docs/
    └── ARCHITECTURE.md
```

---

## Miner & Validator Profiles

### 12 Simulated Miners

| ID | Name | Tier | Base Quality | Base Accuracy |
|----|------|------|-------------|---------------|
| m-001 | DeepReason-v3 | Elite | 0.88 | 0.90 |
| m-002 | LogicForge-7B | Elite | 0.88 | 0.90 |
| m-003 | ProofMaster | Strong | 0.78 | 0.80 |
| m-004 | ReasonSwarm | Strong | 0.78 | 0.80 |
| m-005 | CausalNet | Strong | 0.78 | 0.80 |
| m-006 | ThinkChain | Mid | 0.65 | 0.68 |
| m-007 | InferBot | Mid | 0.65 | 0.68 |
| m-008 | NovaMind | Mid | 0.65 | 0.68 |
| m-009 | BasicReasoner | Weak | 0.45 | 0.50 |
| m-010 | CheapInference | Weak | 0.45 | 0.50 |
| m-011 | SpamBot-X | Adversarial | 0.20 | 0.15 |
| m-012 | CopyCat-3 | Adversarial | 0.20 | 0.15 |

### 6 Simulated Validators

| ID | Name | Stake | Profile | Noise | Bias |
|----|------|-------|---------|-------|------|
| v-001 | TruthGuard | 5000 | Honest | 0.03 | 0.0 |
| v-002 | AccuScore | 3000 | Honest | 0.03 | 0.0 |
| v-003 | FairCheck | 4000 | Good | 0.06 | 0.0 |
| v-004 | QuickVal | 2000 | Good | 0.06 | 0.0 |
| v-005 | LazyNode | 1500 | Lazy | 0.15 | -0.10 |
| v-006 | BadActor | 1000 | Malicious | 0.25 | +0.20 |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/simulate` | Run N epochs, return full results |
| GET | `/api/simulate/stream` | SSE stream, one epoch at a time |
| GET | `/api/constants` | All protocol constants |
| GET | `/api/formulas` | Formula descriptions as JSON |

---

## License

MIT

---

<div align="center">
  <p>Built by <strong>RAVI SHANKAR BEJINI</strong></p>
  <p><i>The Decentralized Marketplace for Verifiable Intelligence</i></p>
</div>
