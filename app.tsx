import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

// ─────────────────────────────────────────────────────────────
// Protocol Constants (matching Python whitepaper exactly)
// ─────────────────────────────────────────────────────────────
const W_QUALITY = 0.40;
const W_ACCURACY = 0.30;
const W_NOVELTY = 0.15;
const W_EFFICIENCY = 0.15;
const EMISSION_MINER_SHARE = 0.90;
const EMISSION_VALIDATOR_SHARE = 0.10;
const PEB_ALPHA = 0.20;
const PEB_K = 10;
const PEB_STREAK_CAP = 10;
const BREAKTHROUGH_MULTIPLIER = 2.0;
const BREAKTHROUGH_THRESHOLD = 0.8;
const TRAP_RATE = 0.15;
const TRAP_THRESHOLD = 0.30;
const VAS_SLASH_THRESHOLD = 0.60;
const VAS_SLASH_GAMMA = 0.05;
const TASKS_PER_EPOCH = 12;
const VALIDATORS_PER_TASK = 3;

const DIFFICULTY_MULTIPLIER: Record<number, number> = {
  1: 1.0, 2: 1.0, 3: 1.25, 4: 1.25,
  5: 1.5, 6: 1.5, 7: 1.75, 8: 1.75,
  9: 2.0, 10: 2.0,
};

const DOMAINS = ['mathematics', 'code', 'scientific', 'strategic', 'causal', 'ethical'] as const;

// ─────────────────────────────────────────────────────────────
// JS Simulation Engine (porting Python formulas)
// ─────────────────────────────────────────────────────────────

function computeCMS(q: number, a: number, n: number, e: number): number {
  return W_QUALITY * q + W_ACCURACY * a + W_NOVELTY * n + W_EFFICIENCY * e;
}

function computePEB(rank: number, streak: number): number {
  if (rank < 1 || rank > PEB_K) return 0;
  const capped = Math.min(streak, PEB_STREAK_CAP);
  if (capped <= 0) return 0;
  return PEB_ALPHA * (1.0 / rank) * Math.sqrt(capped);
}

function distributeEmissions(miners: MinerSt[], pool: number): number[] {
  const weighted = miners.map(m => m.sEpoch * (1.0 + m.peb));
  const total = weighted.reduce((a, b) => a + b, 0);
  if (total <= 0) return miners.map(() => pool / miners.length);
  return weighted.map(w => (w / total) * pool);
}

function computeVAS(scores: number[], consensus: number[]): number {
  if (scores.length === 0) return 1.0;
  const totalDev = scores.reduce((sum, s, i) => sum + Math.abs(s - consensus[i]), 0);
  return Math.max(0, 1.0 - totalDev / scores.length);
}

function computeTrapPenalty(trapScores: number[]): number {
  if (trapScores.length === 0) return 1.0;
  const avg = trapScores.reduce((a, b) => a + b, 0) / trapScores.length;
  if (avg >= TRAP_THRESHOLD) return 1.0;
  return Math.max(0, avg / TRAP_THRESHOLD);
}

function computeSlash(stake: number, vasAvg: number): number {
  if (vasAvg >= VAS_SLASH_THRESHOLD) return 0;
  return VAS_SLASH_GAMMA * stake * Math.pow(VAS_SLASH_THRESHOLD - vasAvg, 2);
}

function clamp(v: number, min = 0, max = 1): number {
  return Math.max(min, Math.min(max, v));
}

function gaussianRandom(rng: () => number, mean = 0, std = 1): number {
  const u1 = rng();
  const u2 = rng();
  return mean + std * Math.sqrt(-2 * Math.log(u1 || 0.001)) * Math.cos(2 * Math.PI * u2);
}

// Seeded PRNG (simple Mulberry32)
function createRng(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

interface MinerSt {
  minerId: string;
  name: string;
  tier: 'elite' | 'strong' | 'mid' | 'weak' | 'adversarial';
  sEpoch: number;
  peb: number;
  rank: number;
  streak: number;
  totalTao: number;
  epochTao: number;
  trapPenalty: number;
  breakthroughs: number;
  epochScores: number[];
  trapScores: number[];
  sEpochHistory: number[];
  quality: number;
  accuracy: number;
  novelty: number;
  efficiency: number;
}

interface ValidatorSt {
  validatorId: string;
  name: string;
  stake: number;
  profile: 'honest' | 'good' | 'lazy' | 'malicious';
  vas: number;
  repMult: number;
  totalTao: number;
  epochTao: number;
  slashed: number;
  vasHistory: number[];
}

const MINER_TIERS: Record<string, { q: number; a: number; n: number; e: number; var: number }> = {
  elite:       { q: 0.88, a: 0.90, n: 0.80, e: 0.85, var: 0.06 },
  strong:      { q: 0.78, a: 0.80, n: 0.70, e: 0.75, var: 0.08 },
  mid:         { q: 0.65, a: 0.68, n: 0.55, e: 0.65, var: 0.10 },
  weak:        { q: 0.45, a: 0.50, n: 0.40, e: 0.55, var: 0.12 },
  adversarial: { q: 0.20, a: 0.15, n: 0.10, e: 0.30, var: 0.15 },
};

const VALIDATOR_PROFILES: Record<string, { noise: number; bias: number }> = {
  honest:    { noise: 0.03, bias: 0.0 },
  good:      { noise: 0.06, bias: 0.0 },
  lazy:      { noise: 0.15, bias: -0.10 },
  malicious: { noise: 0.25, bias: +0.20 },
};

const DEFAULT_MINERS: { id: string; name: string; tier: MinerSt['tier'] }[] = [
  { id: 'm-001', name: 'DeepReason-v3', tier: 'elite' },
  { id: 'm-002', name: 'LogicForge-7B', tier: 'elite' },
  { id: 'm-003', name: 'ProofMaster', tier: 'strong' },
  { id: 'm-004', name: 'ReasonSwarm', tier: 'strong' },
  { id: 'm-005', name: 'CausalNet', tier: 'strong' },
  { id: 'm-006', name: 'ThinkChain', tier: 'mid' },
  { id: 'm-007', name: 'InferBot', tier: 'mid' },
  { id: 'm-008', name: 'NovaMind', tier: 'mid' },
  { id: 'm-009', name: 'BasicReasoner', tier: 'weak' },
  { id: 'm-010', name: 'CheapInference', tier: 'weak' },
  { id: 'm-011', name: 'SpamBot-X', tier: 'adversarial' },
  { id: 'm-012', name: 'CopyCat-3', tier: 'adversarial' },
];

const DEFAULT_VALIDATORS: { id: string; name: string; stake: number; profile: ValidatorSt['profile'] }[] = [
  { id: 'v-001', name: 'TruthGuard', stake: 5000, profile: 'honest' },
  { id: 'v-002', name: 'AccuScore', stake: 3000, profile: 'honest' },
  { id: 'v-003', name: 'FairCheck', stake: 4000, profile: 'good' },
  { id: 'v-004', name: 'QuickVal', stake: 2000, profile: 'good' },
  { id: 'v-005', name: 'LazyNode', stake: 1500, profile: 'lazy' },
  { id: 'v-006', name: 'BadActor', stake: 1000, profile: 'malicious' },
];

// ─────────────────────────────────────────────────────────────
// Full Epoch Runner (JS port of Python simulator)
// ─────────────────────────────────────────────────────────────

function initMiners(): MinerSt[] {
  return DEFAULT_MINERS.map(m => ({
    minerId: m.id,
    name: m.name,
    tier: m.tier,
    sEpoch: 0,
    peb: 0,
    rank: 0,
    streak: 0,
    totalTao: 0,
    epochTao: 0,
    trapPenalty: 1.0,
    breakthroughs: 0,
    epochScores: [],
    trapScores: [],
    sEpochHistory: [],
    quality: 0,
    accuracy: 0,
    novelty: 0,
    efficiency: 0,
  }));
}

function initValidators(): ValidatorSt[] {
  return DEFAULT_VALIDATORS.map(v => ({
    validatorId: v.id,
    name: v.name,
    stake: v.stake,
    profile: v.profile,
    vas: 1.0,
    repMult: 1.0,
    totalTao: 0,
    epochTao: 0,
    slashed: 0,
    vasHistory: [],
  }));
}

interface EpochConfig {
  totalEmission: number;
  epochId: number;
}

interface EpochStats {
  tasksProcessed: number;
  trapsInjected: number;
  breakthroughs: number;
  avgCms: number;
}

function runEpoch(
  miners: MinerSt[],
  validators: ValidatorSt[],
  config: EpochConfig,
  rng: () => number,
): EpochStats {
  const { totalEmission, epochId } = config;
  const taskCount = TASKS_PER_EPOCH;
  const trapCount = Math.max(1, Math.floor(taskCount * TRAP_RATE));

  // Reset epoch accumulators
  miners.forEach(m => {
    m.epochScores = [];
    m.trapScores = [];
    m.epochTao = 0;
    m.quality = 0;
    m.accuracy = 0;
    m.novelty = 0;
    m.efficiency = 0;
  });
  validators.forEach(v => {
    v.epochTao = 0;
    v.slashed = 0;
  });

  // Generate tasks
  const tasks: { difficulty: number; isTrap: boolean; domain: string; unsolved: boolean }[] = [];
  for (let i = 0; i < taskCount; i++) {
    const isTrap = i < trapCount;
    tasks.push({
      difficulty: Math.floor(rng() * 8) + 2, // 2-9
      isTrap,
      domain: DOMAINS[Math.floor(rng() * DOMAINS.length)],
      unsolved: rng() < 0.05,
    });
  }
  // Shuffle
  for (let i = tasks.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [tasks[i], tasks[j]] = [tasks[j], tasks[i]];
  }

  const valScoresGiven: Record<string, number[]> = {};
  const valConsensusRef: Record<string, number[]> = {};
  validators.forEach(v => {
    valScoresGiven[v.validatorId] = [];
    valConsensusRef[v.validatorId] = [];
  });

  let allCms: number[] = [];
  let totalBreakthroughs = 0;

  // Process each task
  for (const task of tasks) {
    const diffMult = DIFFICULTY_MULTIPLIER[task.difficulty] || 1.0;
    const diffPenalty = (task.difficulty - 5) * 0.015;

    for (const miner of miners) {
      const tier = MINER_TIERS[miner.tier];
      const domainBonus = (rng() - 0.3) * 0.15; // approx [-0.05, 0.10]

      const q = clamp(tier.q + domainBonus - diffPenalty + gaussianRandom(rng, 0, tier.var));
      const a = clamp(tier.a + domainBonus - diffPenalty + gaussianRandom(rng, 0, tier.var));
      const n = clamp(tier.n + domainBonus - diffPenalty + gaussianRandom(rng, 0, tier.var));
      const e = clamp(tier.e + domainBonus - diffPenalty + gaussianRandom(rng, 0, tier.var));

      // Track latest dimension scores for display
      miner.quality = q;
      miner.accuracy = a;
      miner.novelty = n;
      miner.efficiency = e;

      const oScore = computeCMS(q, a, n, e);

      // Assign validators and evaluate
      const shuffledVals = [...validators].sort(() => rng() - 0.5);
      const assigned = shuffledVals.slice(0, Math.min(VALIDATORS_PER_TASK, validators.length));

      const valScoreStakes: [number, number][] = assigned.map(v => {
        const prof = VALIDATOR_PROFILES[v.profile];
        const vScore = clamp(oScore + prof.bias + gaussianRandom(rng, 0, prof.noise));
        return [vScore, v.stake];
      });

      // Consensus score (stake-weighted average for simplicity)
      const totalStake = valScoreStakes.reduce((s, [, st]) => s + st, 0);
      const cScore = totalStake > 0
        ? valScoreStakes.reduce((s, [sc, st]) => s + sc * st, 0) / totalStake
        : oScore;

      const finalScore = 0.60 * oScore + 0.40 * cScore;

      // Compute CMS from final score
      let cms = finalScore;

      // Apply breakthrough
      if (task.unsolved && cms > BREAKTHROUGH_THRESHOLD) {
        cms *= BREAKTHROUGH_MULTIPLIER;
        totalBreakthroughs++;
        miner.breakthroughs++;
      }

      // Track VAS deviations
      assigned.forEach((v, idx) => {
        valScoresGiven[v.validatorId].push(valScoreStakes[idx][0]);
        valConsensusRef[v.validatorId].push(cScore);
      });

      miner.epochScores.push(cms * diffMult);
      allCms.push(cms);

      if (task.isTrap) {
        miner.trapScores.push(cms);
      }
    }
  }

  // Compute S_epoch
  miners.forEach(m => {
    if (m.epochScores.length > 0) {
      const avg = m.epochScores.reduce((a, b) => a + b, 0) / m.epochScores.length;
      const tp = computeTrapPenalty(m.trapScores);
      m.trapPenalty = tp;
      m.sEpoch = avg * tp;
    } else {
      m.sEpoch = 0;
    }
  });

  // Rank miners
  const sorted = [...miners].sort((a, b) => b.sEpoch - a.sEpoch);
  sorted.forEach((m, i) => {
    m.rank = i + 1;
    if (m.rank <= PEB_K && m.sEpoch > 0) {
      m.streak++;
    } else {
      m.streak = 0;
    }
    m.peb = computePEB(m.rank, m.streak);
  });

  // Store history
  miners.forEach(m => m.sEpochHistory.push(m.sEpoch));

  // Distribute miner emissions
  const minerPool = totalEmission * EMISSION_MINER_SHARE;
  const rewards = distributeEmissions(sorted, minerPool);
  sorted.forEach((m, i) => {
    m.epochTao = Math.round(rewards[i] * 1000000) / 1000000;
    m.totalTao += m.epochTao;
  });

  // Finalize validator VAS
  validators.forEach(v => {
    const given = valScoresGiven[v.validatorId];
    const ref = valConsensusRef[v.validatorId];
    v.vas = given.length > 0 ? computeVAS(given, ref) : 1.0;
    v.vasHistory.push(v.vas);

    // Reputation multiplier
    const avg30 = v.vasHistory.length > 0
      ? v.vasHistory.slice(-30).reduce((a, b) => a + b, 0) / Math.min(v.vasHistory.length, 30)
      : 1.0;
    if (avg30 <= 0.80) {
      v.repMult = 1.0;
    } else {
      v.repMult = Math.min(1.5, 1.0 + 0.5 * (avg30 - 0.80) / 0.20);
    }

    // Slashing
    const avg7 = v.vasHistory.length > 0
      ? v.vasHistory.slice(-7).reduce((a, b) => a + b, 0) / Math.min(v.vasHistory.length, 7)
      : 1.0;
    v.slashed = computeSlash(v.stake, avg7);
  });

  // Distribute validator emissions
  const valPool = totalEmission * EMISSION_VALIDATOR_SHARE;
  const valWeighted = validators.map(v => v.vas * v.stake * v.repMult);
  const valTotal = valWeighted.reduce((a, b) => a + b, 0);
  validators.forEach((v, i) => {
    v.epochTao = valTotal > 0
      ? Math.round((valWeighted[i] / valTotal) * valPool * 1000000) / 1000000
      : valPool / validators.length;
    v.totalTao += v.epochTao;
  });

  return {
    tasksProcessed: taskCount,
    trapsInjected: trapCount,
    breakthroughs: totalBreakthroughs,
    avgCms: allCms.length > 0 ? allCms.reduce((a, b) => a + b, 0) / allCms.length : 0,
  };
}

// ─────────────────────────────────────────────────────────────
// Color Helpers
// ─────────────────────────────────────────────────────────────

const TIER_COLORS: Record<string, string> = {
  elite: '#FFD700',
  strong: '#60A5FA',
  mid: '#9CA3AF',
  weak: '#FB923C',
  adversarial: '#EF4444',
};

const TIER_BG: Record<string, string> = {
  elite: 'rgba(255,215,0,0.15)',
  strong: 'rgba(96,165,250,0.15)',
  mid: 'rgba(156,163,175,0.10)',
  weak: 'rgba(251,146,60,0.15)',
  adversarial: 'rgba(239,68,68,0.15)',
};

function vasColor(vas: number): string {
  if (vas > 0.8) return '#10B981';
  if (vas > 0.6) return '#F59E0B';
  return '#EF4444';
}

function vasLabel(vas: number): string {
  if (vas > 0.8) return 'Healthy';
  if (vas > 0.6) return 'Warning';
  return 'Critical';
}

// ─────────────────────────────────────────────────────────────
// Dashboard Component
// ─────────────────────────────────────────────────────────────

export default function App() {
  const [miners, setMiners] = useState<MinerSt[]>(initMiners());
  const [validators, setValidators] = useState<ValidatorSt[]>(initValidators());
  const [epochCount, setEpochCount] = useState(0);
  const [stats, setStats] = useState<EpochStats>({ tasksProcessed: 0, trapsInjected: 0, breakthroughs: 0, avgCms: 0 });
  const [autoRun, setAutoRun] = useState(false);
  const [selectedMiner, setSelectedMiner] = useState<string | null>(null);
  const [emission, setEmission] = useState(100);
  const seedRef = useRef(42);
  const rngRef = useRef(createRng(42));

  const handleRunEpoch = useCallback(() => {
    const newEpoch = epochCount + 1;
    const newMiners = miners.map(m => ({ ...m, sEpochHistory: [...m.sEpochHistory] }));
    const newValidators = validators.map(v => ({ ...v, vasHistory: [...v.vasHistory] }));

    const epochStats = runEpoch(newMiners, newValidators, {
      totalEmission: emission,
      epochId: newEpoch,
    }, rngRef.current);

    setMiners(newMiners);
    setValidators(newValidators);
    setEpochCount(newEpoch);
    setStats(epochStats);
  }, [miners, validators, epochCount, emission]);

  const handleReset = useCallback(() => {
    seedRef.current = Math.floor(Math.random() * 100000);
    rngRef.current = createRng(seedRef.current);
    setMiners(initMiners());
    setValidators(initValidators());
    setEpochCount(0);
    setStats({ tasksProcessed: 0, trapsInjected: 0, breakthroughs: 0, avgCms: 0 });
    setSelectedMiner(null);
  }, []);

  useEffect(() => {
    if (!autoRun) return;
    const timer = setInterval(handleRunEpoch, 2000);
    return () => clearInterval(timer);
  }, [autoRun, handleRunEpoch]);

  const sortedMiners = [...miners].sort((a, b) => b.sEpoch - a.sEpoch);
  const selectedMinerData = miners.find(m => m.minerId === selectedMiner);

  // Chart data
  const barData = sortedMiners.map(m => ({
    name: m.name.length > 12 ? m.name.slice(0, 12) + '..' : m.name,
    reward: Math.round(m.epochTao * 100) / 100,
    peb: Math.round(m.peb * m.epochTao * 100) / 100,
    tier: m.tier,
  }));

  // Line chart data for epoch history
  const lineData: Record<string, unknown>[] = [];
  if (miners[0]?.sEpochHistory.length > 0) {
    const maxEpochs = miners[0].sEpochHistory.length;
    for (let i = 0; i < maxEpochs; i++) {
      const point: Record<string, unknown> = { epoch: i + 1 };
      miners.forEach(m => {
        if (i < m.sEpochHistory.length) {
          point[m.name] = Math.round(m.sEpochHistory[i] * 10000) / 10000;
        }
      });
      lineData.push(point);
    }
  }

  return (
    <div style={{ background: '#0D1B2A', minHeight: '100vh', color: '#E0E6ED', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <header style={{ background: '#162234', borderBottom: '1px solid #1E3A5F', padding: '16px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <h1 style={{ fontSize: '24px', fontWeight: 700, margin: 0 }}>
            <span style={{ color: '#00BFA5' }}>Reason</span>Forge
          </h1>
          <span style={{ fontSize: '13px', color: '#6B7280', borderLeft: '1px solid #2D3748', paddingLeft: '16px' }}>
            Decentralized Verifiable Reasoning
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '14px', color: '#9CA3AF' }}>
            Epoch: <strong style={{ color: '#00BFA5' }}>{epochCount}</strong>
          </span>
          <span style={{ fontSize: '13px', color: '#6B7280' }}>
            Emission:
          </span>
          <input
            type="number"
            value={emission}
            onChange={e => setEmission(Number(e.target.value) || 100)}
            style={{ width: '70px', background: '#0D1B2A', border: '1px solid #2D3748', color: '#E0E6ED', borderRadius: '4px', padding: '4px 8px', fontSize: '13px' }}
          />
          <span style={{ fontSize: '13px', color: '#6B7280' }}>TAO</span>
          <button onClick={handleRunEpoch} style={{ background: '#00BFA5', color: '#0D1B2A', border: 'none', borderRadius: '6px', padding: '8px 20px', fontWeight: 600, cursor: 'pointer', fontSize: '14px' }}>
            Run Epoch
          </button>
          <button
            onClick={() => setAutoRun(!autoRun)}
            style={{ background: autoRun ? '#EF4444' : '#1E3A5F', color: '#E0E6ED', border: '1px solid ' + (autoRun ? '#EF4444' : '#2D3748'), borderRadius: '6px', padding: '8px 16px', cursor: 'pointer', fontSize: '14px' }}
          >
            {autoRun ? 'Stop' : 'Auto-Run'}
          </button>
          <button onClick={handleReset} style={{ background: 'transparent', color: '#6B7280', border: '1px solid #2D3748', borderRadius: '6px', padding: '8px 16px', cursor: 'pointer', fontSize: '14px' }}>
            Reset
          </button>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 420px', gap: '16px', padding: '16px 24px', maxWidth: '1600px', margin: '0 auto' }}>
        {/* Left Panel - Miner Leaderboard */}
        <div>
          <div style={{ background: '#162234', borderRadius: '8px', border: '1px solid #1E3A5F', overflow: 'hidden' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid #1E3A5F', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ margin: 0, fontSize: '16px', fontWeight: 600 }}>Miner Leaderboard</h2>
              <span style={{ fontSize: '12px', color: '#6B7280' }}>Click a row for details</span>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ background: '#0D1B2A' }}>
                    {['#', 'Name', 'Tier', 'S_epoch', 'PEB', 'Streak', 'Epoch TAO', 'Total TAO', 'Trap'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: '#6B7280', fontWeight: 500, borderBottom: '1px solid #1E3A5F' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedMiners.map((m, i) => (
                    <tr
                      key={m.minerId}
                      onClick={() => setSelectedMiner(selectedMiner === m.minerId ? null : m.minerId)}
                      style={{
                        cursor: 'pointer',
                        background: selectedMiner === m.minerId ? 'rgba(0,191,165,0.08)' : (i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)'),
                        borderLeft: selectedMiner === m.minerId ? '3px solid #00BFA5' : '3px solid transparent',
                        transition: 'all 0.2s',
                      }}
                    >
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>
                        {m.rank <= 3 ? <span style={{ color: '#FFD700' }}>{m.rank}</span> : m.rank}
                      </td>
                      <td style={{ padding: '10px 12px', fontWeight: 500 }}>{m.name}</td>
                      <td style={{ padding: '10px 12px' }}>
                        <span style={{ background: TIER_BG[m.tier], color: TIER_COLORS[m.tier], padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase' }}>
                          {m.tier}
                        </span>
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{ width: '60px', height: '6px', background: '#1E3A5F', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min(100, (m.sEpoch / 2) * 100)}%`, height: '100%', background: TIER_COLORS[m.tier], borderRadius: '3px', transition: 'width 0.5s' }} />
                          </div>
                          <span>{m.sEpoch.toFixed(4)}</span>
                        </div>
                      </td>
                      <td style={{ padding: '10px 12px', color: m.peb > 0 ? '#00BFA5' : '#4B5563' }}>{m.peb.toFixed(4)}</td>
                      <td style={{ padding: '10px 12px' }}>
                        {m.streak > 0 && <span style={{ color: '#F59E0B' }}>{'~'.repeat(Math.min(m.streak, 5))}</span>}
                        <span style={{ marginLeft: '4px' }}>{m.streak}</span>
                      </td>
                      <td style={{ padding: '10px 12px', fontWeight: 500 }}>{m.epochTao.toFixed(2)}</td>
                      <td style={{ padding: '10px 12px', color: '#00BFA5', fontWeight: 600 }}>{m.totalTao.toFixed(2)}</td>
                      <td style={{ padding: '10px 12px' }}>
                        {m.trapPenalty < 1.0
                          ? <span style={{ color: '#EF4444', fontSize: '11px' }}>! {m.trapPenalty.toFixed(2)}</span>
                          : <span style={{ color: '#10B981', fontSize: '11px' }}>OK</span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Selected Miner Detail */}
          {selectedMinerData && (
            <div style={{ background: '#162234', borderRadius: '8px', border: '1px solid #1E3A5F', marginTop: '16px', padding: '20px' }}>
              <h3 style={{ margin: '0 0 16px', fontSize: '15px' }}>
                <span style={{ color: TIER_COLORS[selectedMinerData.tier] }}>{selectedMinerData.name}</span>
                <span style={{ color: '#6B7280', fontSize: '12px', marginLeft: '8px' }}>Dimension Breakdown (Latest Task)</span>
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                {[
                  { label: 'Quality', value: selectedMinerData.quality, weight: W_QUALITY, color: '#60A5FA' },
                  { label: 'Accuracy', value: selectedMinerData.accuracy, weight: W_ACCURACY, color: '#34D399' },
                  { label: 'Novelty', value: selectedMinerData.novelty, weight: W_NOVELTY, color: '#A78BFA' },
                  { label: 'Efficiency', value: selectedMinerData.efficiency, weight: W_EFFICIENCY, color: '#FB923C' },
                ].map(d => (
                  <div key={d.label} style={{ background: '#0D1B2A', borderRadius: '6px', padding: '12px' }}>
                    <div style={{ fontSize: '11px', color: '#6B7280', marginBottom: '4px' }}>{d.label} (w={d.weight})</div>
                    <div style={{ fontSize: '20px', fontWeight: 700, color: d.color }}>{d.value.toFixed(3)}</div>
                    <div style={{ height: '4px', background: '#1E3A5F', borderRadius: '2px', marginTop: '6px' }}>
                      <div style={{ width: `${d.value * 100}%`, height: '100%', background: d.color, borderRadius: '2px', transition: 'width 0.3s' }} />
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: '12px', fontSize: '13px', color: '#9CA3AF' }}>
                CMS = {W_QUALITY}*{selectedMinerData.quality.toFixed(3)} + {W_ACCURACY}*{selectedMinerData.accuracy.toFixed(3)} + {W_NOVELTY}*{selectedMinerData.novelty.toFixed(3)} + {W_EFFICIENCY}*{selectedMinerData.efficiency.toFixed(3)} = <strong style={{ color: '#00BFA5' }}>{computeCMS(selectedMinerData.quality, selectedMinerData.accuracy, selectedMinerData.novelty, selectedMinerData.efficiency).toFixed(4)}</strong>
              </div>
            </div>
          )}

          {/* TAO Distribution Chart */}
          {epochCount > 0 && (
            <div style={{ background: '#162234', borderRadius: '8px', border: '1px solid #1E3A5F', marginTop: '16px', padding: '20px' }}>
              <h3 style={{ margin: '0 0 16px', fontSize: '15px' }}>TAO Distribution</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={barData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E3A5F" />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#6B7280' }} angle={-30} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 11, fill: '#6B7280' }} />
                  <Tooltip
                    contentStyle={{ background: '#162234', border: '1px solid #1E3A5F', borderRadius: '6px', fontSize: '12px' }}
                    labelStyle={{ color: '#E0E6ED' }}
                  />
                  <Bar dataKey="reward" name="Epoch TAO" stackId="a">
                    {barData.map((entry, idx) => (
                      <Cell key={idx} fill={TIER_COLORS[entry.tier] || '#6B7280'} fillOpacity={0.7} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Epoch History Line Chart */}
          {lineData.length > 1 && (
            <div style={{ background: '#162234', borderRadius: '8px', border: '1px solid #1E3A5F', marginTop: '16px', padding: '20px' }}>
              <h3 style={{ margin: '0 0 16px', fontSize: '15px' }}>S_epoch Trends Over Time</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={lineData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E3A5F" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: '#6B7280' }} />
                  <YAxis tick={{ fontSize: 11, fill: '#6B7280' }} />
                  <Tooltip
                    contentStyle={{ background: '#162234', border: '1px solid #1E3A5F', borderRadius: '6px', fontSize: '11px' }}
                    labelStyle={{ color: '#E0E6ED' }}
                  />
                  {DEFAULT_MINERS.slice(0, 6).map((m, i) => (
                    <Line
                      key={m.id}
                      type="monotone"
                      dataKey={m.name}
                      stroke={TIER_COLORS[m.tier]}
                      strokeWidth={m.tier === 'elite' ? 2 : 1}
                      dot={false}
                      strokeOpacity={0.8}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', marginTop: '8px' }}>
                {DEFAULT_MINERS.slice(0, 6).map(m => (
                  <span key={m.id} style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <span style={{ width: '10px', height: '3px', background: TIER_COLORS[m.tier], display: 'inline-block', borderRadius: '2px' }} />
                    <span style={{ color: '#9CA3AF' }}>{m.name}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Panel */}
        <div>
          {/* Epoch Stats */}
          <div style={{ background: '#162234', borderRadius: '8px', border: '1px solid #1E3A5F', padding: '20px', marginBottom: '16px' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '15px' }}>Epoch Stats</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              {[
                { label: 'Tasks Processed', value: stats.tasksProcessed, color: '#60A5FA' },
                { label: 'Traps Injected', value: stats.trapsInjected, color: '#F59E0B' },
                { label: 'Breakthroughs', value: stats.breakthroughs, color: '#A78BFA' },
                { label: 'Avg CMS', value: stats.avgCms.toFixed(4), color: '#00BFA5' },
              ].map(s => (
                <div key={s.label} style={{ background: '#0D1B2A', borderRadius: '6px', padding: '12px', textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: '#6B7280', marginBottom: '4px' }}>{s.label}</div>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: s.color }}>{s.value}</div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '12px' }}>
              <div style={{ background: '#0D1B2A', borderRadius: '4px', padding: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#6B7280' }}>Miner Pool</span>
                <span style={{ color: '#00BFA5' }}>{(emission * EMISSION_MINER_SHARE).toFixed(1)} TAO</span>
              </div>
              <div style={{ background: '#0D1B2A', borderRadius: '4px', padding: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#6B7280' }}>Validator Pool</span>
                <span style={{ color: '#00BFA5' }}>{(emission * EMISSION_VALIDATOR_SHARE).toFixed(1)} TAO</span>
              </div>
            </div>
          </div>

          {/* Validator Health */}
          <div style={{ background: '#162234', borderRadius: '8px', border: '1px solid #1E3A5F', padding: '20px', marginBottom: '16px' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '15px' }}>Validator Health</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {validators.map(v => (
                <div key={v.validatorId} style={{ background: '#0D1B2A', borderRadius: '6px', padding: '12px', borderLeft: `3px solid ${vasColor(v.vas)}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <div>
                      <span style={{ fontWeight: 600, fontSize: '13px' }}>{v.name}</span>
                      <span style={{ fontSize: '11px', color: '#6B7280', marginLeft: '8px' }}>
                        Stake: {v.stake.toLocaleString()}
                      </span>
                    </div>
                    <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '4px', background: vasColor(v.vas) + '20', color: vasColor(v.vas), fontWeight: 600 }}>
                      {vasLabel(v.vas)}
                    </span>
                  </div>
                  {/* VAS Gauge */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                    <span style={{ fontSize: '11px', color: '#6B7280', width: '30px' }}>VAS</span>
                    <div style={{ flex: 1, height: '6px', background: '#1E3A5F', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ width: `${v.vas * 100}%`, height: '100%', background: vasColor(v.vas), borderRadius: '3px', transition: 'all 0.5s' }} />
                    </div>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: vasColor(v.vas), width: '50px', textAlign: 'right' }}>{v.vas.toFixed(4)}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#6B7280' }}>
                    <span>Rep: x{v.repMult.toFixed(3)}</span>
                    <span>TAO: {v.epochTao.toFixed(2)}</span>
                    <span>Total: {v.totalTao.toFixed(2)}</span>
                    {v.slashed > 0 && <span style={{ color: '#EF4444' }}>Slashed: {v.slashed.toFixed(4)}</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* CMS Formula Display */}
          <div style={{ background: '#162234', borderRadius: '8px', border: '1px solid #1E3A5F', padding: '20px' }}>
            <h3 style={{ margin: '0 0 12px', fontSize: '15px' }}>CMS Formula (Eq. 2)</h3>
            <div style={{ fontFamily: 'monospace', fontSize: '13px', background: '#0D1B2A', padding: '12px', borderRadius: '6px', lineHeight: 1.8 }}>
              <div>
                CMS(m,t) = <span style={{ color: '#60A5FA' }}>{W_QUALITY}</span>*Q +{' '}
                <span style={{ color: '#34D399' }}>{W_ACCURACY}</span>*A +{' '}
                <span style={{ color: '#A78BFA' }}>{W_NOVELTY}</span>*N +{' '}
                <span style={{ color: '#FB923C' }}>{W_EFFICIENCY}</span>*Eff
              </div>
              <div style={{ marginTop: '8px', color: '#6B7280', fontSize: '11px' }}>
                S_epoch = (1/|T|) * SUM(CMS * D(t)) * trap_penalty
              </div>
              <div style={{ color: '#6B7280', fontSize: '11px' }}>
                PEB = {PEB_ALPHA} * (1/rank) * sqrt(min(streak, {PEB_STREAK_CAP}))
              </div>
              <div style={{ color: '#6B7280', fontSize: '11px' }}>
                R(m) = E_miner * [S * (1+PEB)] / SUM[S * (1+PEB)]
              </div>
            </div>
            <div style={{ marginTop: '12px', fontSize: '11px', color: '#4B5563' }}>
              <div>Emission Split: Miners {EMISSION_MINER_SHARE * 100}% | Validators {EMISSION_VALIDATOR_SHARE * 100}%</div>
              <div>Trap Rate: {TRAP_RATE * 100}% | Breakthrough: x{BREAKTHROUGH_MULTIPLIER} (CMS {'>'} {BREAKTHROUGH_THRESHOLD})</div>
              <div>Slash: gamma={VAS_SLASH_GAMMA} * stake * (theta - VAS_7d)^2</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
