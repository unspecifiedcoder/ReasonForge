# ReasonForge — Technical Architecture

## Overview

ReasonForge is a Bittensor subnet proposal implementing a decentralized marketplace for verifiable multi-step reasoning. This document describes the technical architecture of the MVP implementation.

## System Architecture

```
                     ┌─────────────────────────┐
                     │   Interactive Dashboard  │
                     │   (React/TypeScript)     │
                     │   Embedded JS Engine     │
                     └──────────┬──────────────┘
                                │ HTTP/SSE
                     ┌──────────▼──────────────┐
                     │   FastAPI Server         │
                     │   /api/simulate          │
                     │   /api/simulate/stream   │
                     │   /api/constants         │
                     └──────────┬──────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                   │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌────────▼────────┐
    │ Scoring Engine  │ │  Simulator   │ │ Task Generator   │
    │ (13 Formulas)   │ │  (Epoch Loop)│ │ (6 Domains)      │
    └─────────┬──────┘ └──────┬───────┘ └────────┬────────┘
              │                │                   │
              └────────────────┼───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Types & Constants  │
                    │   (Protocol Layer)   │
                    └─────────────────────┘
```

## Component Details

### 1. Protocol Layer (`reasonforge/types.py`)

The foundation of the system. Contains:

- **20+ Protocol Constants**: Exact values from the whitepaper (CMS weights, emission splits, PEB parameters, trap thresholds, slashing parameters, etc.)
- **2 Enums**: `Domain` (6 reasoning domains) and `TaskSource` (4 task origins)
- **9 Dataclasses**: `Task`, `ReasoningStep`, `MinerSubmission`, `DimensionScores`, `ValidatorScore`, `MinerState`, `ValidatorState`, `EpochResult`
- **Domain Check Weights**: Per-domain weight maps for objective scoring (Eq. 11)

Design principle: All constants are defined once and imported everywhere. No magic numbers.

### 2. Scoring Engine (`reasonforge/engine.py`)

A stateless class with 13 `@staticmethod` methods, one per whitepaper formula:

| Method | Formula | Invariants |
|--------|---------|------------|
| `compute_cms` | Eq. 2 | Output in [0, 1] for valid inputs |
| `compute_s_epoch` | Eq. 3 | Returns 0 for empty task list |
| `compute_peb` | Eq. 4 | Returns 0 for rank > K |
| `distribute_miner_emissions` | Eq. 5 | **Conservation**: sum(rewards) == pool |
| `apply_breakthrough` | Eq. 6 | Only applies if CMS > 0.8 AND unsolved |
| `compute_vas` | Eq. 7 | Returns 1.0 for perfect scoring |
| `distribute_validator_emissions` | Eq. 8 | **Conservation**: sum(rewards) == pool |
| `compute_trap_penalty` | Eq. 9 | Returns 1.0 if no traps or above threshold |
| `compute_slash` | Eq. 10 | Quadratic penalty, 0 above threshold |
| `compute_objective_score` | Eq. 11 | Weighted sum of domain checks |
| `compute_consensus_score` | Eq. 12 | Trimmed median for |V| >= 5 |
| `compute_final_score` | Eq. 13 | 0.60*O + 0.40*C |

Design principle: Pure functions. No side effects. Easy to test in isolation.

### 3. Simulator (`reasonforge/simulator.py`)

Three main classes:

#### MinerProfile
Simulated miners with statistical capability profiles across 5 tiers:
- **Elite** (base Q=0.88, A=0.90) — consistently high performance
- **Strong** (Q=0.78, A=0.80) — reliable above-average
- **Mid** (Q=0.65, A=0.68) — average performance with higher variance
- **Weak** (Q=0.45, A=0.50) — below average
- **Adversarial** (Q=0.20, A=0.15) — deliberately poor, high variance

Each miner also gets random per-domain bonuses, simulating domain specialization.

#### ValidatorProfile
Simulated validators with 4 accuracy profiles:
- **Honest** (noise=0.03, bias=0.0) — accurate scoring
- **Good** (noise=0.06, bias=0.0) — slightly noisy but unbiased
- **Lazy** (noise=0.15, bias=-0.10) — high noise, scores low
- **Malicious** (noise=0.25, bias=+0.20) — high noise, inflates scores

#### EpochSimulator
The main simulation loop (`run_epoch()`):

1. Generate 12 tasks (15% traps) across 6 domains
2. Each of 12 miners solves each task (probabilistic scores from profiles)
3. Compute objective scores per miner per task
4. Assign 3 random validators per miner-task pair
5. Validators evaluate (adding noise/bias per profile)
6. Compute consensus score (stake-weighted average)
7. Compute final score (0.60*O + 0.40*C)
8. Compute CMS, apply breakthrough multiplier if applicable
9. Compute epoch scores (difficulty-weighted average * trap penalty)
10. Rank miners, compute PEB for top-10
11. Distribute miner emissions (Eq. 5 — emission-conserving)
12. Finalize validator VAS, reputation, slashing
13. Distribute validator emissions (Eq. 8)

### 4. Task Generator (`reasonforge/task_generator.py`)

Generates synthetic reasoning tasks:
- 5-7 templates per domain (30+ total across 6 domains)
- Trap tasks with known ground truth scores
- Random difficulty assignment (2-9)
- 5% chance of `previously_unsolved` for breakthrough opportunities

### 5. Plagiarism Detector (`reasonforge/plagiarism.py`)

MVP implementation using:
- **Jaccard similarity** on 3-gram token sets from reasoning steps
- **Hash-based exact match** via SHA-256 submission hashes
- Rolling buffer of last 30 epochs
- Threshold: 0.95 similarity = plagiarism, applies 0.5x penalty

In production, this would use embedding cosine similarity.

### 6. API Server (`api/server.py`)

FastAPI server with CORS enabled for dashboard:
- `GET /api/health` — Health check
- `POST /api/simulate` — Run N epochs, return full JSON results
- `GET /api/simulate/stream` — Server-Sent Events for live dashboard
- `GET /api/constants` — All protocol constants
- `GET /api/formulas` — Formula descriptions

### 7. Dashboard (`app.tsx`)

Single-file React/TypeScript dashboard with:
- **Embedded JS simulation engine** porting all Python formulas
- **Miner leaderboard** with sortable table, tier badges, streak indicators
- **Validator health** cards with VAS gauges (green/yellow/red)
- **TAO distribution** bar chart (recharts)
- **S_epoch history** line chart tracking trends over time
- **CMS formula display** with live values
- **Miner detail panel** showing quality/accuracy/novelty/efficiency breakdown
- Controls: Run Epoch, Auto-Run (2s interval), Reset, emission input

Works completely standalone without the Python API.

## Data Flow

```
Task Generator → Tasks → Miner Profiles → Submissions → Scoring Engine
                                                              ↓
                                            Validator Profiles → Evaluations
                                                              ↓
                                                        Consensus Score
                                                              ↓
                                                        Final Score → CMS
                                                              ↓
                                                        Epoch Scores
                                                              ↓
                                                    Emission Distribution
                                                              ↓
                                                        EpochResult → JSON
```

## Testing Strategy

67 tests across 3 test files:

- **test_engine.py** (40 tests): Every formula validated against hand-calculated expected values. Tests for boundary conditions (zeros, ones), conservation invariants, monotonicity.
- **test_simulator.py** (17 tests): Integration tests for full epoch lifecycle, emission conservation, adversarial detection, streak accumulation, validator slashing.
- **test_types.py** (10 tests): Dataclass properties, constant verification, computed properties.

Key invariant: **Emission conservation** — `sum(miner_rewards) + sum(validator_rewards) == total_emission` (within floating point tolerance).

## Design Decisions

1. **Stateless engine**: All scoring methods are pure static functions. State lives in dataclasses, not in the engine.
2. **Statistical simulation**: Miners and validators are statistical profiles, not real LLMs. This demonstrates mechanism design without requiring AI inference.
3. **Dual-mode dashboard**: The React dashboard embeds its own JS simulation engine, making it fully functional without the Python backend.
4. **JSON interchange**: All components communicate via JSON. The CLI outputs JSON, the API returns JSON, the dashboard consumes JSON.
5. **Single-source constants**: Protocol constants are defined once in `types.py` and imported everywhere (Python) / defined once at the top of `app.tsx` (JavaScript).
