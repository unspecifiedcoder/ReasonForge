<div align="center">

# 🌌 ReasonForge
### The Decentralized Marketplace for Verifiable Intelligence
**A Bittensor Subnet Proposal | Subnet Ideathon Round I**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/Frontend-React-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

[**Live Dashboard**](https://reasonforge-app.vercel.app) • [**Technical Documentation**](docs/ARCHITECTURE.md) • [**GitHub Source**](https://github.com/unspecifiedcoder/ReasonForge)

---

</div>

## 📖 Overview

**ReasonForge** is a next-generation Bittensor subnet designed to incentivize **structured, auditable reasoning chains**. While existing AI subnets often prioritize raw generation speed or generic quality, ReasonForge focuses on the *integrity of the thought process*.

By leveraging a mathematically rigorous protocol, ReasonForge transforms raw compute into **Verifiable Intelligence**. Whether it's complex logical proofs, causal inference, or high-stakes financial modeling, ReasonForge ensure that every answer is backed by a transparent, verifiable chain of logic.

### 🎯 Key Pillars
- **Verifiable Chains**: Every response includes an auditable reasoning path.
- **Mathematical Rigor**: 13 unique formulas governing every aspect of the network.
- **Validator Accountability**: Quadratic slashing and reputation-weighted rewards.
- **Trap Detection**: Integrated mechanisms to catch lazy or malicious actors.

---

## 🛠️ Quick Start

Experience the power of the ReasonForge protocol locally in minutes.

### 1. 🧬 Simulation Environment
Clone the repository and install the core protocol package:
```bash
pip install -e .
python -m reasonforge.run --epochs 10 --emission 100 --output results.json --seed 42
```

### 2. ⚡ Developer API
Spin up the FastAPI server to interface with the simulation engine:
```bash
pip install fastapi uvicorn
uvicorn api.server:app --reload --port 8000
```

### 3. 🎨 Interactive Dashboard
Launch the high-fidelity visualization suite:
```bash
npm install
npm run dev
```

---

## 🏗️ System Architecture

ReasonForge is built on a modular "Separation of Concerns" architecture, ensuring scalability and ease of integration.

| Component | Tech Stack | Responsibility |
|:---|:---|:---|
| **Core Engine** | `Python` | Implementation of all 13 whitepaper formulas. |
| **Epoch Simulator** | `Python` | Full-cycle network simulation (Miners/Validators). |
| **Interactive UI** | `React/TS` | Real-time visualization of network health and rewards. |
| **Plagiarism Guard** | `Python` | Jaccard-based similarity detection for network integrity. |
| **Validation Layer** | `Pytest` | Comprehensive suite of 60+ tests for formula accuracy. |

---

## 📐 The Mathematical Core

ReasonForge is governed by 13 core formulas that ensure game-theoretic stability and fairness.

### 💎 Composite Miner Score (CMS)
$$CMS = 0.40Q + 0.30A + 0.15N + 0.15Eff$$
- **Q**: Path Quality
- **A**: Accuracy
- **N**: Novelty
- **Eff**: Efficiency

### ⚖️ Validator Accuracy (VAS)
$$VAS = 1 - \text{mean}(|v_{\text{score}} - \text{consensus}|)$$

### 🛡️ Quadratic Slashing
$$Slash = 0.05 \times \text{Stake} \times (0.60 - VAS_{7d})^2$$

---

## 👥 Network Profiles

### ⛏️ Simulated Miners
We simulate a diverse range of 12 miners to test protocol resilience:
- **Elite (m-001, m-002)**: High-performance reasoning nodes.
- **Adversarial (m-011, m-012)**: Testing plagiarism and spam detection.

### 🛡️ Simulated Validators
6 distinct validator archetypes:
- **Honest (v-001, v-002)**: High stake, high accuracy.
- **Lazy/Malicious (v-005, v-006)**: Triggering slashing and penalty mechanisms.

---

## 🛣️ Roadmap

- [x] **Phase 1**: Core Mathematical Engine & 13 Formulas.
- [x] **Phase 2**: Multi-Epoch Simulator & CLI.
- [x] **Phase 3**: FastAPI Integration & SSE Streaming.
- [wip] **Phase 4**: Mainchain Integration & Bittensor Onboarding.

---

## 📜 License & Acknowledgments

This project is licensed under the **MIT License**.

<div align="center">

**Developed with ❤️ by RAVI SHANKAR BEJINI**
*Empowering the world with Verifiable Intelligence*

[**Contact**](mailto:your-email@example.com) | [**LinkedIn**](https://linkedin.com/in/your-profile) | [**Twitter**](https://twitter.com/your-handle)

</div>
