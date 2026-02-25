# ReasonForge — Complete MVP Build Plan

> **Purpose**: This document is a self-contained specification for building the ReasonForge MVP from scratch. Hand this to Claude Code / OpenCode in a new empty folder.
> **Estimated build time**: ~45 minutes of agent work.
> **No external APIs needed** — everything runs locally.

---

## 1. Project Overview

**ReasonForge** is a Bittensor subnet proposal for a decentralized marketplace for verifiable multi-step reasoning. This MVP implements:

1. **Core Mechanism Engine** (Python) — All 13 whitepaper formulas as working code
2. **Epoch Simulator** (Python) — Full lifecycle simulation with 12 miners (elite→adversarial) and 6 validators (honest→malicious)
3. **Interactive Dashboard** (React/JSX single file) — Real-time visualization of epoch simulations
4. **CLI Runner** — Terminal-based multi-epoch simulation with rich output
5. **Test Suite** — Unit tests validating every formula against the whitepaper
6. **API Server** (FastAPI) — REST endpoint serving simulation data to the dashboard
7. **Documentation** — README, architecture docs

---

## 2. Directory Structure

```
reasonforge/
├── README.md                          # Project overview + setup instructions
├── pyproject.toml                     # Python project config (use hatchling)
├── requirements.txt                   # Python deps
├── .gitignore
│
├── reasonforge/                       # Core Python package
│   ├── __init__.py                    # Package init, exports
│   ├── types.py                       # All data structures + protocol constants
│   ├── engine.py                      # Scoring engine (all 13 formulas)
│   ├── simulator.py                   # Epoch simulator (task gen → scoring → rewards)
│   ├── plagiarism.py                  # Similarity detection module
│   ├── task_generator.py              # Task templates + synthetic generation
│   └── run.py                         # CLI entry point
│
├── api/                               # FastAPI server
│   ├── __init__.py
│   └── server.py                      # REST API serving simulation data
│
├── dashboard/                         # React dashboard
│   └── App.jsx                        # Single-file interactive dashboard
│
├── tests/                             # Test suite
│   ├── test_engine.py                 # Formula validation tests
│   ├── test_simulator.py              # Integration tests
│   └── test_types.py                  # Type/constant tests
│
├── simulation/                        # Output directory
│   └── .gitkeep
│
└── docs/                              # Documentation
    └── ARCHITECTURE.md                # Technical architecture doc
```

---

## 3. Protocol Constants (must be exact across all files)

```python
# CMS Weights (Eq. 2)
W_QUALITY = 0.40
W_ACCURACY = 0.30
W_NOVELTY = 0.15
W_EFFICIENCY = 0.15

# Emission Split (Eq. 1)
EMISSION_MINER_SHARE = 0.90
EMISSION_VALIDATOR_SHARE = 0.10

# PEB Parameters (Eq. 4)
PEB_ALPHA = 0.20
PEB_K = 10                    # Top-K miners eligible
PEB_STREAK_CAP = 10           # Max streak value

# Breakthrough (Eq. 6)
BREAKTHROUGH_MULTIPLIER = 2.0
BREAKTHROUGH_THRESHOLD = 0.8  # Min CMS to qualify

# Trap Problems (Eq. 9)
TRAP_RATE = 0.15              # 15% of tasks are traps
TRAP_THRESHOLD = 0.30         # θ_trap

# Similarity Detection
SIMILARITY_THRESHOLD = 0.95
SIMILARITY_PENALTY = 0.50

# Validator Slashing (Eq. 10)
VAS_SLASH_THRESHOLD = 0.60    # θ_slash
VAS_SLASH_GAMMA = 0.05        # γ

# Validator Reputation
VAS_REP_THRESHOLD = 0.80
VAS_REP_MAX_MULTIPLIER = 1.50

# Operational
TASKS_PER_EPOCH = 12
VALIDATORS_PER_TASK = 3
MICRO_ROUND_SECONDS = 300     # 5 minutes
EPOCH_HOURS = 24

# Difficulty Multiplier Map (difficulty 1-10 → D(t))
DIFFICULTY_MULTIPLIER = {
    1: 1.0, 2: 1.0, 3: 1.25, 4: 1.25,
    5: 1.5, 6: 1.5, 7: 1.75, 8: 1.75,
    9: 2.0, 10: 2.0,
}

# Objective/Consensus Split (Eq. 13)
OBJECTIVE_WEIGHT = 0.60
CONSENSUS_WEIGHT = 0.40
CONSENSUS_TRIM_DELTA = 0.10   # Trim top/bottom 10%
```

---

## 4. Mathematical Formulas (implement exactly)

### Eq. 1 — Emission Split
```
E_total = E_miner + E_validator = 0.90·E + 0.10·E
```

### Eq. 2 — Composite Miner Score (CMS)
```
CMS(m,t) = 0.40·Q(m,t) + 0.30·A(m,t) + 0.15·N(m,t) + 0.15·Eff(m,t)
```
All dimension scores ∈ [0, 1].

### Eq. 3 — Epoch Score
```
S_epoch(m) = (1/|T_m|) · Σ_t [CMS(m,t) · D(t)] · trap_penalty(m)
```
Where D(t) is the difficulty multiplier from the map above.

### Eq. 4 — Persistent Excellence Bonus
```
PEB(m) = α · (1/rank(m)) · √(min(streak(m), 10))
```
Only for miners with rank ≤ K=10.

### Eq. 5 — Final Miner Reward
```
R(m) = E_miner · [S_epoch(m) · (1 + PEB(m))] / Σ_j [S_epoch(j) · (1 + PEB(j))]
```
Must conserve total emission (sum of all R(m) == E_miner).

### Eq. 6 — Breakthrough Multiplier
```
CMS_breakthrough(m,t) = CMS(m,t) · 2.0
```
Only if task is previously_unsolved AND CMS > 0.8.

### Eq. 7 — Validator Accuracy Score (VAS)
```
VAS(v) = 1 - (1/|T_v|) · Σ_t |score_v(m,t) - score_consensus(m,t)|
```

### Eq. 8 — Validator Reward
```
R_v(v) = E_validator · [VAS(v) · stake(v) · rep_mult(v)] / Σ_k [VAS(k) · stake(k) · rep_mult(k)]
```

### Eq. 9 — Trap Penalty
```
trap_penalty(m) = max(0, avg_trap_score(m) / θ_trap)    if avg < θ_trap
                = 1.0                                      if avg ≥ θ_trap
```

### Eq. 10 — Validator Slashing
```
slash(v) = γ · stake(v) · max(0, θ_slash - VAS_7d_avg(v))²
```

### Eq. 11 — Objective Score
```
O_score(m,t) = Σ_k ω_k · check_k(submission)
```
Domain-specific weights (see Section 7 for check weight tables).

### Eq. 12 — Consensus Score (Stake-weighted trimmed median)
```
C_score(m,t) = TrimmedMedian_δ({score_v · stake_v / Σstake : v ∈ V_assigned})
```
Trim δ = 10% from top and bottom when |V| ≥ 5.

### Eq. 13 — Final Score
```
FinalScore(m,t) = 0.60 · O_score + 0.40 · C_score
```

### Reputation Multiplier (validator)
```
rep_mult(v) = 1.0 + 0.5 · max(0, VAS_30d_avg - 0.80) / 0.20
```
Capped at 1.5.

---

## 5. Data Types (types.py)

### Enums
```python
class Domain(str, Enum):
    MATHEMATICS = "mathematics"
    CODE = "code"
    SCIENTIFIC = "scientific"
    STRATEGIC = "strategic"
    CAUSAL = "causal"
    ETHICAL = "ethical"

class TaskSource(str, Enum):
    USER_API = "user_api"
    BENCHMARK = "benchmark"
    SYNTHETIC = "synthetic"
    TRAP = "trap"
```

### Core Dataclasses
Use `@dataclass` for all. Include `field(default_factory=...)` for mutable defaults.

- **Task**: task_id (uuid), problem (str), domain, difficulty (1-10), timeout_seconds, source, is_trap (bool), ground_truth_score (optional float), previously_unsolved (bool). Property: `difficulty_multiplier`.

- **ReasoningStep**: step_id (int), reasoning (str), evidence (str), confidence (float 0-1), formal_proof_fragment (optional str).

- **MinerSubmission**: task_id, miner_id, steps (list[ReasoningStep]), final_answer, proof_status ("VERIFIED"/"FAILED"/None), time_ms, submission_hash, submitted_at. Method: `compute_hash()` using SHA-256.

- **DimensionScores**: quality, accuracy, novelty, efficiency — all float [0,1]. Property: `cms` computing Eq. 2.

- **ValidatorScore**: validator_id, objective_score, consensus_score, final_score, dimension_scores.

- **MinerState**: miner_id, name, epoch_scores (list[float]), epoch_tasks (list[str]), trap_scores (list[float]), s_epoch, peb, rank, streak, total_tao_earned, epoch_tao, task_count, breakthroughs. Properties: `trap_score_avg`, `trap_penalty`.

- **ValidatorState**: validator_id, name, stake, vas_history (list[float]), current_vas, reputation_multiplier, total_tao_earned, epoch_tao, slashed_amount, evaluations_count. Properties: `vas_7d_avg`, `vas_30d_avg`. Methods: `compute_reputation_multiplier()`, `compute_slash()`.

- **EpochResult**: epoch_id, total_emission, miner_pool, validator_pool, miner_results, validator_results, tasks_processed, traps_injected, breakthroughs, avg_cms, timestamp.

---

## 6. Scoring Engine (engine.py)

Implement a `ScoringEngine` class with **all static methods**. Each method maps to one formula:

| Method | Formula | Input | Output |
|--------|---------|-------|--------|
| `compute_cms(scores)` | Eq.2 | DimensionScores | float |
| `compute_s_epoch(cms_list, diff_mults, trap_penalty)` | Eq.3 | lists + float | float |
| `compute_peb(rank, streak)` | Eq.4 | int, int | float |
| `distribute_miner_emissions(miners, pool)` | Eq.5 | list[MinerState], float | list[float] |
| `apply_breakthrough(cms, is_breakthrough)` | Eq.6 | float, bool | float |
| `compute_vas(v_scores, consensus_scores)` | Eq.7 | list[float], list[float] | float |
| `distribute_validator_emissions(validators, pool)` | Eq.8 | list[ValidatorState], float | list[float] |
| `compute_trap_penalty(trap_scores)` | Eq.9 | list[float] | float |
| `compute_slash(stake, vas_7d_avg)` | Eq.10 | float, float | float |
| `compute_objective_score(checks, weights)` | Eq.11 | dict, dict | float |
| `compute_consensus_score(val_scores_stakes, trim_delta)` | Eq.12 | list[tuple], float | float |
| `compute_final_score(o_score, c_score)` | Eq.13 | float, float | float |

---

## 7. Simulator (simulator.py)

### MinerProfile class
Simulated miners with capability profiles. Constructor takes `(miner_id, name, tier)`.

**Tiers** (base scores + variance):
```
elite:       q=0.88, a=0.90, n=0.80, e=0.85, var=0.06
strong:      q=0.78, a=0.80, n=0.70, e=0.75, var=0.08
mid:         q=0.65, a=0.68, n=0.55, e=0.65, var=0.10
weak:        q=0.45, a=0.50, n=0.40, e=0.55, var=0.12
adversarial: q=0.20, a=0.15, n=0.10, e=0.30, var=0.15
```

Each miner also gets random per-domain bonuses ∈ [-0.05, 0.10].

**solve_task(task)** method:
- Computes score per dimension: `base + domain_bonus - difficulty_penalty + gaussian(0, variance)`
- difficulty_penalty = `(difficulty - 5) * 0.015`
- Clamp all scores to [0, 1]
- Returns `(DimensionScores, MinerSubmission)`

### ValidatorProfile class
Simulated validators with accuracy profiles. Constructor: `(validator_id, name, stake, accuracy)`.

**Accuracy profiles**:
```
honest:    noise=0.03, bias=0.0
good:      noise=0.06, bias=0.0
lazy:      noise=0.15, bias=-0.10
malicious: noise=0.25, bias=+0.20
```

**evaluate(true_score)** method:
- Returns `clamp(true_score + bias + gaussian(0, noise), 0, 1)`

### EpochSimulator class
Main simulation runner. Constructor: `(miners, validators, miner_states?, validator_states?, epoch_id, total_emission)`.

**generate_tasks(count=12)**:
- Creates `count` tasks
- `int(count * 0.15)` are traps (with ground_truth_score)
- Rest are randomly sampled from templates across all 6 domains
- Each task gets random difficulty 2-9
- 5% chance of `previously_unsolved = True`
- Shuffle and return

**run_epoch()** — THE MAIN LOOP:

```
1. Generate tasks
2. Reset epoch accumulators for all miners/validators
3. For each task:
   a. Each miner solves the task → (DimensionScores, Submission)
   b. Compute O_score (simulated from true quality dimensions)
   c. Assign 3 random validators
   d. Each validator evaluates → individual score
   e. Compute C_score via consensus (stake-weighted trimmed median)
   f. FinalScore = 0.60·O + 0.40·C
   g. Map FinalScore back to dimension scores
   h. Compute CMS
   i. Apply breakthrough multiplier if applicable
   j. Track VAS deviations for each validator
   k. Store CMS in miner's epoch_scores
   l. If trap task, store in trap_scores
4. For each miner:
   a. Compute S_epoch (difficulty-weighted average × trap_penalty)
5. Rank miners by S_epoch descending
6. Compute PEB for top-K
7. Distribute miner emissions (Eq. 5)
8. Finalize validator VAS (average of recent deviations)
9. Compute validator reputation multipliers
10. Compute slashing for underperformers
11. Distribute validator emissions (Eq. 8)
12. Return EpochResult
```

**to_json(result)** — Serialize to JSON-safe dict with all fields rounded.

### Task Templates
Create 5+ task templates per domain (see whitepaper Section 2.1 for examples). Include realistic problem descriptions.

### Default Roster

**12 Miners:**
```
m-001  DeepReason-v3     elite
m-002  LogicForge-7B     elite
m-003  ProofMaster       strong
m-004  ReasonSwarm       strong
m-005  CausalNet         strong
m-006  ThinkChain        mid
m-007  InferBot          mid
m-008  NovaMind          mid
m-009  BasicReasoner     weak
m-010  CheapInference    weak
m-011  SpamBot-X         adversarial
m-012  CopyCat-3         adversarial
```

**6 Validators:**
```
v-001  TruthGuard  5000  honest
v-002  AccuScore   3000  honest
v-003  FairCheck   4000  good
v-004  QuickVal    2000  good
v-005  LazyNode    1500  lazy
v-006  BadActor    1000  malicious
```

---

## 8. Plagiarism Detection (plagiarism.py)

Implement a `PlagiarismDetector` class:

- Maintains a rolling buffer of submission hashes/embeddings (last 30 epochs)
- `check(submission, history) → float` returns max similarity score
- For MVP, use a simplified approach: compute jaccard similarity of step reasoning text, or hash-based comparison
- If similarity > 0.95, flag and apply 0.5× penalty
- In production this would use embedding cosine similarity

---

## 9. CLI Runner (run.py)

Entry point: `python -m reasonforge.run [options]`

**Arguments:**
```
--epochs N        Number of epochs (default: 5)
--emission FLOAT  TAO per epoch (default: 100.0)
--output FILE     Save JSON results to file
--seed INT        Random seed for reproducibility
--verbose         Show per-task details
```

**Output format:**
- ASCII art banner ("REASONFORGE")
- Configuration summary
- Per-epoch: task stats, miner leaderboard table (Rank, Name, S_epoch, PEB, Streak, TAO, Total), validator summary table (Name, Stake, VAS, Rep×, TAO, Slashed)
- Final standings after all epochs
- Key observations (who got slashed, who maintained streaks, adversarial detection)

Use visual markers:
- ★ for top-3 miners
- ⚠ for trap penalty
- ✓/△/✗ for validator VAS health

---

## 10. API Server (api/server.py)

FastAPI server with these endpoints:

```
GET  /api/health              → {"status": "ok", "version": "0.1.0"}
POST /api/simulate            → Run N epochs, return full results JSON
  Body: {"epochs": 5, "emission": 100.0, "seed": 42}
GET  /api/simulate/stream     → SSE stream, emitting one epoch at a time (for live dashboard)
  Query: ?epochs=10&emission=100
GET  /api/constants            → Return all protocol constants
GET  /api/formulas             → Return formula descriptions as JSON
```

**Response schema for /api/simulate:**
```json
{
  "config": {"epochs": 5, "emission": 100, "miners": 12, "validators": 6},
  "epochs": [
    {
      "epoch_id": 1,
      "total_emission": 100,
      "miner_pool": 90,
      "validator_pool": 10,
      "tasks_processed": 12,
      "traps_injected": 2,
      "breakthroughs": 0,
      "avg_cms": 0.6234,
      "miners": [
        {"rank": 1, "miner_id": "m-001", "name": "DeepReason-v3", "s_epoch": 0.8921, "peb": 0.200, "streak": 1, "epoch_tao": 18.42, "total_tao": 18.42, "trap_penalty": 1.0}
      ],
      "validators": [
        {"validator_id": "v-001", "name": "TruthGuard", "stake": 5000, "vas": 0.9712, "reputation": 1.428, "epoch_tao": 4.23, "total_tao": 4.23, "slashed": 0.0}
      ]
    }
  ]
}
```

Run with: `uvicorn api.server:app --reload --port 8000`

---

## 11. Interactive Dashboard (dashboard/App.jsx)

A single React JSX file artifact that:

1. **Has a built-in simulation engine in JS** — port the core formulas so it runs client-side without needing the API. The dashboard should work standalone.

2. **Layout:**
   - Header: ReasonForge logo/title + epoch counter + "Run Epoch" / "Auto-Run" buttons
   - Left Panel (60%): Miner Leaderboard — sortable table showing all 12 miners with Rank, Name, Tier badge, S_epoch bar, PEB indicator, Streak flames, Epoch TAO, Total TAO, Trap status
   - Right Panel (40% top): Validator Health — cards showing each validator with VAS gauge, reputation multiplier, stake, slashing status
   - Right Panel (40% bottom): Epoch Stats — tasks processed, traps injected, breakthroughs, avg CMS
   - Bottom: TAO Distribution Chart — recharts BarChart showing miner rewards, with PEB highlighted in different color
   - Bottom-right: CMS Formula display — show the live equation with current values highlighted

3. **Interactivity:**
   - "Run Epoch" button: advances one epoch, animates changes
   - "Auto-Run" toggle: automatically runs epochs every 2 seconds
   - "Reset" button: resets all state
   - Click a miner row to see detailed breakdown (quality, accuracy, novelty, efficiency scores)
   - Epoch history: small line chart showing S_epoch trends over time per miner

4. **Visual Design:**
   - Dark theme: background #0D1B2A, cards #162234, accent #00BFA5, text #E0E6ED
   - Tier badges: elite=gold, strong=blue, mid=gray, weak=orange, adversarial=red
   - Use Tailwind classes for layout
   - Use recharts for charts (BarChart, LineChart, RadarChart)
   - Smooth transitions on data updates
   - VAS gauges: green >0.8, yellow >0.6, red <0.6

5. **JS Simulation Engine (embedded in the JSX):**
   Port these functions to JavaScript:
   - `computeCMS(q, a, n, e)` → Eq. 2
   - `computePEB(rank, streak)` → Eq. 4
   - `distributeEmissions(miners, pool)` → Eq. 5
   - `computeVAS(scores, consensus)` → Eq. 7
   - `computeTrapPenalty(trapScores)` → Eq. 9
   - `computeSlash(stake, vasAvg)` → Eq. 10
   - `runEpoch(minerStates, validatorStates, config)` → full epoch

   Miner profiles and validator profiles should be defined as JS objects matching the Python ones.

---

## 12. Test Suite

### test_engine.py
Test every formula method. For each test:
- State what equation is being tested
- Provide hand-calculated expected values
- Assert within 1e-10 tolerance
- Include edge cases (zero scores, max scores, boundary conditions)

**Must-have tests:**
```
test_cms_computation         — Eq.2 with known values
test_cms_boundary            — All zeros, all ones
test_peb_rank1_streak4       — 0.20 * 1.0 * √4 = 0.40
test_peb_outside_topk        — rank=11 → 0.0
test_peb_streak_cap          — streak=100 should cap at 10
test_emission_conservation   — sum(rewards) == pool
test_emission_monotonic      — higher score → higher reward
test_breakthrough_applied    — CMS × 2.0
test_breakthrough_not_applied — CMS unchanged
test_vas_perfect             — all scores match consensus → 1.0
test_vas_deviation           — known deviations
test_trap_above_threshold    — penalty = 1.0
test_trap_below_threshold    — penalty = avg/θ
test_trap_zero               — penalty = 0.0
test_slash_below_threshold   — quadratic formula
test_slash_above_threshold   — 0.0
test_final_score             — 0.60·O + 0.40·C
test_consensus_trimmed_median — verify trimming works
test_consensus_stake_weighted — higher stake influences median
```

### test_simulator.py
Integration tests:
```
test_epoch_runs_without_error   — basic smoke test
test_epoch_emission_conserved   — miner_rewards + validator_rewards ≈ total
test_adversarial_penalized      — adversarial miners earn less than elite
test_trap_detection_works       — adversarial miners get trap penalties
test_streak_increments          — top miners accumulate streaks
test_lazy_validator_lower_vas   — lazy validators have lower VAS
test_malicious_validator_slashed — malicious validators lose stake
test_multi_epoch_state_carries  — run 3 epochs, verify state continuity
test_peb_only_top_k             — only top 10 get PEB
```

### test_types.py
```
test_difficulty_multiplier_map  — all 10 levels mapped correctly
test_dimension_scores_cms       — property computes correctly
test_miner_state_trap_penalty   — property works
test_validator_state_vas_avg    — rolling average
test_task_defaults              — sensible defaults
```

---

## 13. README.md

Structure:
```markdown
# ReasonForge 🔥

> The Decentralized Marketplace for Verifiable Intelligence
> A Bittensor Subnet Proposal — Subnet Ideathon Round I

## What is ReasonForge?

[2 paragraph overview]

## Quick Start

### Run the Simulator
\`\`\`bash
pip install -e .
python -m reasonforge.run --epochs 10 --emission 100 --output results.json
\`\`\`

### Launch the API
\`\`\`bash
pip install fastapi uvicorn
uvicorn api.server:app --reload --port 8000
\`\`\`

### Run Tests
\`\`\`bash
pytest tests/ -v
\`\`\`

## Architecture

[Brief description + link to docs/ARCHITECTURE.md]

## Mechanism Design

[Table of all 13 formulas with equation numbers]

## Protocol Constants

[Table of all constants with values]

## Project Structure

[Directory tree]

## Miner & Validator Profiles

[Tables showing the simulation roster]

## License

MIT
```

---

## 14. pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "reasonforge"
version = "0.1.0"
description = "The Decentralized Marketplace for Verifiable Intelligence"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
api = ["fastapi>=0.100.0", "uvicorn>=0.23.0"]
dev = ["pytest>=7.0.0"]
all = ["fastapi>=0.100.0", "uvicorn>=0.23.0", "pytest>=7.0.0"]

[project.scripts]
reasonforge = "reasonforge.run:main"
```

---

## 15. requirements.txt

```
fastapi>=0.100.0
uvicorn>=0.23.0
pytest>=7.0.0
```

---

## 16. Build Order (follow this sequence)

```
Step 1:  Create directory structure
Step 2:  Write reasonforge/types.py (all constants + dataclasses)
Step 3:  Write reasonforge/engine.py (all 13 formula methods)
Step 4:  Write tests/test_engine.py and run it — verify ALL formulas pass
Step 5:  Write reasonforge/task_generator.py (task templates)
Step 6:  Write reasonforge/plagiarism.py
Step 7:  Write reasonforge/simulator.py (full epoch simulation)
Step 8:  Write reasonforge/run.py (CLI with rich output)
Step 9:  Write reasonforge/__init__.py
Step 10: Write tests/test_simulator.py and run it
Step 11: Write tests/test_types.py and run it
Step 12: Run full test suite: pytest tests/ -v
Step 13: Run CLI: python -m reasonforge.run --epochs 5 --output simulation/test.json
Step 14: Verify JSON output is valid and correct
Step 15: Write api/server.py (FastAPI)
Step 16: Write dashboard/App.jsx (React dashboard with embedded sim engine)
Step 17: Write pyproject.toml, requirements.txt, .gitignore
Step 18: Write README.md
Step 19: Write docs/ARCHITECTURE.md
Step 20: Final check — run all tests, run CLI, verify everything works
```

---

## 17. Key Design Decisions

1. **Stateless scoring engine** — All ScoringEngine methods are `@staticmethod`. No hidden state. Pure functions. Easy to test.

2. **State carried via dataclasses** — MinerState and ValidatorState are mutable objects passed between epochs. The simulator doesn't own state; it receives and returns it.

3. **Simulated miners, not real LLMs** — For the MVP, miners have statistical profiles that generate scores probabilistically. This demonstrates the mechanism design without needing actual AI models.

4. **Dashboard runs standalone** — The React dashboard has its own JS simulation engine. It doesn't need the Python API. This makes it demo-friendly.

5. **Every formula has a test** — If a formula is in the whitepaper, it has a unit test that validates it against hand-computed values.

6. **JSON as interchange format** — All simulation results serialize to JSON. The API returns JSON. The dashboard consumes JSON.

---

## 18. Domain-Specific Check Weights (for Eq. 11)

```python
DOMAIN_CHECK_WEIGHTS = {
    Domain.MATHEMATICS: {"proof": 0.60, "steps": 0.25, "numerical": 0.15},
    Domain.CODE: {"tests": 0.50, "static_analysis": 0.20, "formal": 0.30},
    Domain.SCIENTIFIC: {"simulation": 0.40, "statistics": 0.35, "citations": 0.25},
    Domain.STRATEGIC: {"solver": 0.50, "constraints": 0.30, "equilibrium": 0.20},
    Domain.CAUSAL: {"docalculus": 0.40, "bootstrap": 0.35, "dag": 0.25},
    Domain.ETHICAL: {"coverage": 0.30, "logic": 0.70},
}
```

---

## 19. .gitignore

```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
dist/
build/
.env
simulation/*.json
node_modules/
.vscode/
.idea/
```

---

## 20. Success Criteria

After building, verify:

- [ ] `pytest tests/ -v` — all tests pass, 0 failures
- [ ] `python -m reasonforge.run --epochs 5 --seed 42` — runs, prints leaderboard, no errors
- [ ] Elite miners consistently rank top 3
- [ ] Adversarial miners get trap penalties and earn least TAO
- [ ] Malicious validator gets slashed, honest validators earn more
- [ ] PEB accumulates for top miners over multiple epochs
- [ ] Emission conservation: sum of all miner rewards + validator rewards ≈ total emission (within rounding)
- [ ] JSON output is valid and parseable
- [ ] API server starts and `/api/simulate` returns correct response
- [ ] Dashboard JSX file is valid React that renders without errors
- [ ] README has complete setup instructions

---

*End of build plan. This document contains everything needed to implement the ReasonForge MVP from an empty directory.*
