import React, { useState, useEffect, useRef } from 'react';
import {
  Brain,
  ShieldCheck,
  Cpu,
  Activity,
  Terminal,
  CheckCircle2,
  AlertCircle,
  Database,
  Zap,
  ChevronRight,
  Code,
  Variable,
  Microscope,
  Scale,
  MessageSquare,
  FileText,
  Sparkles,
  ShieldAlert,
  WifiOff,
  Globe
} from 'lucide-react';

const apiKey = ""; // The environment provides the API key at runtime

interface Task {
  problem: string;
  domain: string;
  difficulty: number;
}

interface MinerStep {
  id: number;
  logic: string;
  evidence?: string;
}

interface MinerOutput {
  steps: MinerStep[];
  final_answer: string;
}

interface ValidationData {
  scores: {
    A: number;
    C: number;
    N: number;
  };
  critique: string;
  cms: number;
}

interface DebateData {
  challenge: string;
  targeted_step_id: number;
  rebuttal: string;
  verdict: string;
}

interface ReportData {
  summary: string;
  business_impact: string;
  confidence_level: string;
}

interface LogEntry {
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'error';
}

const TASK_DOMAINS = [
  { id: 'math', name: 'Mathematics', icon: <Variable className="w-4 h-4" />, color: 'text-blue-400' },
  { id: 'code', name: 'Formal Code', icon: <Code className="w-4 h-4" />, color: 'text-green-400' },
  { id: 'science', name: 'Scientific Logic', icon: <Microscope className="w-4 h-4" />, color: 'text-purple-400' },
  { id: 'ethics', name: 'Ethical Analysis', icon: <Scale className="w-4 h-4" />, color: 'text-amber-400' }
];

const App = () => {
  const [status, setStatus] = useState<string>('idle'); // idle, generating_task, mining, validating, debating, reporting, completed
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [currentTask, setCurrentTask] = useState<Task | null>(null);
  const [minerOutput, setMinerOutput] = useState<MinerOutput | null>(null);
  const [validationData, setValidationData] = useState<ValidationData | null>(null);
  const [debateData, setDebateData] = useState<DebateData | null>(null);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [isSimulation, setIsSimulation] = useState<boolean>(!apiKey);
  const [stats, setStats] = useState<{
    totalTasks: number;
    avgCMS: number;
    taoEmitted: number;
  }>({
    totalTasks: 0,
    avgCMS: 0,
    taoEmitted: 0
  });

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const addLog = (message: string, type: 'info' | 'success' | 'error' = 'info') => {
    setLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), message, type }]);
  };

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  // --- Mock Engine for No-API Key Situations ---
  const getMockResponse = (prompt: string, systemInstruction: string): any => {
    const sys = systemInstruction.toLowerCase();
    const p = prompt.toLowerCase();

    if (sys.includes("generator")) {
      const tasks = [
        { problem: "Prove that for any prime p > 3, p² - 1 is divisible by 24.", domain: "math", difficulty: 7 },
        { problem: "Implement a thread-safe circular buffer in Rust and prove its safety against race conditions.", domain: "code", difficulty: 8 },
        { problem: "Analyze the causal effect of carbon tax implementation on regional manufacturing GDP using a Synthetic Control Method.", domain: "science", difficulty: 9 },
        { problem: "Evaluate the utilitarian vs deontological tradeoffs of an autonomous vehicle prioritizing passenger safety over multiple pedestrians in a zero-option collision scenario.", domain: "ethics", difficulty: 6 }
      ];
      return tasks[Math.floor(Math.random() * tasks.length)];
    }

    if (sys.includes("miner")) {
      return {
        steps: [
          { id: 1, logic: "Initialize formal definitions and established axioms for the given domain.", evidence: "Referencing standard library constants." },
          { id: 2, logic: "Decompose the problem into verifiable sub-clauses to prevent compounding logic errors.", evidence: "Logical modularity check passed." },
          { id: 3, logic: "Execute iterative reasoning chain, verifying each step against the formal sandbox constraints.", evidence: "Step-wise trace validated by local prover." }
        ],
        final_answer: "The solution has been formally forged and verified. All constraints are satisfied."
      };
    }

    if (sys.includes("validator")) {
      return {
        scores: { A: 0.95, C: 0.92, N: 0.88 },
        critique: "Excellent multi-step derivation. The miner demonstrated superior logical density and successfully navigated adversarial edge cases injected into the task pool."
      };
    }

    if (sys.includes("adversarial")) {
      return {
        challenge: "Step 2 assumes a static environment variable that may fluctuate under load.",
        targeted_step_id: 2,
        rebuttal: "The miner logic incorporates a dynamic mutex lock which was omitted for brevity in the summary but is active in the artifact.",
        verdict: "CHALLENGE DEFLECTED - Miner logic holds under adversarial stress."
      };
    }

    if (sys.includes("reporting")) {
      return {
        summary: "The ReasonForge protocol has verified this logic with 99.8% mathematical certainty, outperforming standard centralized LLM baselines.",
        business_impact: "Integrating this verifiable trace reduces audit overhead by 70% for compliance-heavy financial transactions.",
        confidence_level: "HIGH (SYSTEM 2)"
      };
    }

    return null;
  };

  const callGemini = async (prompt: string, systemInstruction: string, responseSchema: any) => {
    if (!apiKey) {
      await sleep(1000 + Math.random() * 1000); // Simulate network latency
      return getMockResponse(prompt, systemInstruction);
    }

    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;

    const payload = {
      contents: [{ parts: [{ text: prompt }] }],
      systemInstruction: { parts: [{ text: systemInstruction }] },
      generationConfig: {
        responseMimeType: "application/json",
        responseSchema: responseSchema
      }
    };

    let lastError: any = null;
    for (let i = 0; i < 5; i++) {
      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.error) throw new Error(data.error.message || "API Error");
        if (!data.candidates?.[0]?.content?.parts?.[0]?.text) throw new Error("Empty response from AI");

        const text = data.candidates[0].content.parts[0].text;
        const cleanJson = text.replace(/^```json\n?/, '').replace(/\n?```$/, '').trim();
        return JSON.parse(cleanJson);
      } catch (error) {
        lastError = error;
        await sleep(Math.pow(2, i) * 1000);
      }
    }

    addLog(`Network Failure: ${lastError?.message || 'Unknown error'}. Switching to Simulation Mode.`, 'error');
    setIsSimulation(true);
    return getMockResponse(prompt, systemInstruction);
  };

  const startDemo = async () => {
    if (status !== 'idle' && status !== 'completed') return;

    setStatus('generating_task');
    setLogs([]);
    setDebateData(null);
    setReportData(null);
    addLog(`Initializing ReasonForge Subnet Protocol... [${isSimulation ? 'SIMULATION' : 'LIVE'}]`);

    const taskSchema = {
      type: "OBJECT",
      properties: {
        problem: { type: "STRING" },
        domain: { type: "STRING", enum: ["math", "code", "science", "ethics"] },
        difficulty: { type: "NUMBER" }
      },
      required: ["problem", "domain", "difficulty"]
    };

    const task = await callGemini(
      `Generate a challenging reasoning task. Network Avg CMS: ${stats.avgCMS.toFixed(2)}.`,
      "You are the ReasonForge Task Generator. Create unique, difficult reasoning tasks for miners.",
      taskSchema
    );

    if (!task) {
      setStatus('idle');
      return;
    }

    setCurrentTask(task);
    addLog(`Task Dispatched: [${task.domain.toUpperCase()}] Difficulty ${task.difficulty}/10`, 'success');

    // Phase 2: Miner Execution
    setStatus('mining');
    addLog("Miner Node 0xa3f7 assigned. Starting Reasoning Engine...");

    const minerSchema = {
      type: "OBJECT",
      properties: {
        steps: {
          type: "ARRAY",
          items: {
            type: "OBJECT",
            properties: {
              id: { type: "NUMBER" },
              logic: { type: "STRING" },
              evidence: { type: "STRING" }
            },
            required: ["id", "logic"]
          }
        },
        final_answer: { type: "STRING" }
      },
      required: ["steps", "final_answer"]
    };

    const result = await callGemini(
      `Solve this problem: ${task.problem}`,
      "You are a ReasonForge Miner. Provide deep, multi-step, verifiable reasoning in a structured format.",
      minerSchema
    );

    if (!result) { setStatus('idle'); return; }

    setMinerOutput(result);
    addLog(`Miner generated ${result.steps.length} reasoning steps.`);

    // Phase 3: Validation
    setStatus('validating');
    addLog("Validator Node 0x992b analyzing reasoning trace...");

    const validatorSchema = {
      type: "OBJECT",
      properties: {
        scores: {
          type: "OBJECT",
          properties: { A: { type: "NUMBER" }, C: { type: "NUMBER" }, N: { type: "NUMBER" } },
          required: ["A", "C", "N"]
        },
        critique: { type: "STRING" }
      },
      required: ["scores", "critique"]
    };

    const validation = await callGemini(
      `Review task and miner solution: ${JSON.stringify(result)}`,
      "You are a ReasonForge Validator. Be rigorous.",
      validatorSchema
    );

    if (!validation) { setStatus('idle'); return; }

    const cms = (validation.scores.A * 0.4) + (validation.scores.C * 0.4) + (validation.scores.N * 0.2);
    setValidationData({ ...validation, cms });

    addLog(`Validation Complete. Composite Miner Score (CMS): ${cms.toFixed(4)}`, 'success');
    setStatus('completed');

    setStats(prev => ({
      totalTasks: prev.totalTasks + 1,
      avgCMS: (prev.avgCMS * prev.totalTasks + cms) / (prev.totalTasks + 1),
      taoEmitted: prev.taoEmitted + (cms * 0.5)
    }));
  };

  const startAdversarialDebate = async () => {
    if (!minerOutput) return;
    setStatus('debating');
    addLog("✨ Initializing Adversarial Debate Protocol...");

    const debate = await callGemini(
      `Challenge this trace: ${JSON.stringify(minerOutput)}.`,
      "You are simulating the ReasonForge Adversarial Debate process.",
      null
    );

    if (debate) {
      setDebateData(debate);
      addLog(`Debate concluded. Verdict generated.`, 'success');
    }
    setStatus('completed');
  };

  const generateReport = async () => {
    if (!validationData || !currentTask) return;
    setStatus('reporting');
    addLog("✨ Synthesizing Enterprise Intelligence Report...");

    const report = await callGemini(
      `Generate professional summary for Task: ${currentTask.problem}.`,
      "You are the ReasonForge Enterprise Reporting Agent.",
      null
    );

    if (report) {
      setReportData(report);
      addLog("Enterprise Report generated.", 'success');
    }
    setStatus('completed');
  };

  return (
    <div className="min-h-screen bg-[#050505] text-slate-200 font-sans p-4 md:p-8">
      <style dangerouslySetInnerHTML={{
        __html: `
        :root {
          --rf-teal: #00BFA5;
          --rf-orange: #FF6B35;
          --rf-purple: #8B5CF6;
        }
        .text-rf-teal { color: var(--rf-teal); }
        .bg-rf-teal { background-color: var(--rf-teal); }
        .border-rf-teal { border-color: var(--rf-teal); }
        .text-rf-orange { color: var(--rf-orange); }
        .bg-rf-orange { background-color: var(--rf-orange); }
        .text-rf-purple { color: var(--rf-purple); }
        .bg-rf-purple { background-color: var(--rf-purple); }
      `}} />

      {/* Header */}
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-2 uppercase">
            <Brain className="text-rf-teal w-8 h-8" />
            Reason<span className="text-rf-teal">Forge</span>
            <span className={`text-[10px] px-2 py-0.5 rounded border ml-2 normal-case tracking-normal ${isSimulation ? 'bg-amber-400/10 text-amber-400 border-amber-400/20' : 'bg-rf-teal/10 text-rf-teal border-rf-teal/20'}`}>
              {isSimulation ? 'Simulation Mode' : 'Live Network'}
            </span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">Decentralized Verifiable Reasoning Infrastructure</p>
        </div>

        <div className="flex gap-3">
          <div className="flex items-center gap-2 bg-black/40 border border-slate-800 rounded-lg px-4 py-2 text-xs">
            {isSimulation ? <WifiOff className="w-3 h-3 text-amber-400" /> : <Globe className="w-3 h-3 text-rf-teal" />}
            <span className={isSimulation ? 'text-amber-400' : 'text-rf-teal'}>
              {isSimulation ? 'API Key Not Found' : 'API Connection Active'}
            </span>
          </div>
          <button
            onClick={startDemo}
            disabled={status !== 'idle' && status !== 'completed'}
            className="flex items-center gap-2 bg-rf-teal hover:opacity-90 disabled:opacity-30 disabled:cursor-not-allowed text-black font-bold py-2.5 px-6 rounded-lg transition-all shadow-[0_0_20px_rgba(0,191,165,0.2)]"
          >
            {status === 'generating_task' || status === 'mining' || status === 'validating' ? <Activity className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4 fill-current" />}
            {status === 'idle' || status === 'completed' ? "Dispatch Task" : "Forging Logic..."}
          </button>
        </div>
      </div>

      {/* Main Layout */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* Network Metrics Sidebar */}
        <div className="lg:col-span-4 space-y-6 flex flex-col h-full">
          <div className="bg-[#0f1115] border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden group">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-5 flex items-center gap-2">
              <Activity className="w-4 h-4 text-rf-teal" /> Subnet Metrics
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-black/40 p-4 rounded-xl border border-slate-800/50">
                <div className="text-[10px] text-slate-500 uppercase font-bold mb-1">Epoch Tasks</div>
                <div className="text-2xl font-bold text-white">{stats.totalTasks}</div>
              </div>
              <div className="bg-black/40 p-4 rounded-xl border border-slate-800/50">
                <div className="text-[10px] text-slate-500 uppercase font-bold mb-1">Network CMS</div>
                <div className="text-2xl font-bold text-rf-teal">{stats.avgCMS.toFixed(3)}</div>
              </div>
              <div className="bg-black/40 p-4 rounded-xl border border-slate-800/50 col-span-2 flex justify-between items-center">
                <div>
                  <div className="text-[10px] text-slate-500 uppercase font-bold mb-1">TAO Emitted (Est.)</div>
                  <div className="text-2xl font-bold text-rf-orange">{stats.taoEmitted.toFixed(4)} <span className="text-sm font-normal opacity-50">τ</span></div>
                </div>
                <div className="bg-rf-orange/10 p-2 rounded-full">
                  <Zap className="w-5 h-5 text-rf-orange" />
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#0f1115] border border-slate-800 rounded-xl flex-grow overflow-hidden flex flex-col shadow-xl min-h-[300px]">
            <div className="p-4 border-b border-slate-800 bg-slate-900/40 flex justify-between items-center">
              <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                <Terminal className="w-3 h-3 text-rf-teal" /> Protocol Logs
              </h3>
            </div>
            <div
              ref={scrollRef}
              className="p-5 font-mono text-[11px] flex-grow overflow-y-auto space-y-2.5"
            >
              {logs.map((log, i) => (
                <div key={i} className={`flex gap-3 leading-relaxed ${log.type === 'error' ? 'text-red-400' : log.type === 'success' ? 'text-rf-teal' : 'text-slate-400'}`}>
                  <span className="opacity-30 shrink-0">[{log.timestamp}]</span>
                  <span>{log.message}</span>
                </div>
              ))}
              {status === 'idle' && <div className="text-slate-700 italic">Waiting for protocol initiation...</div>}
            </div>
          </div>
        </div>

        {/* Task Processing View */}
        <div className="lg:col-span-8 space-y-6">

          {currentTask && (
            <div className="bg-[#0f1115] border border-slate-800 rounded-xl overflow-hidden shadow-2xl animate-in fade-in slide-in-from-bottom-2">
              <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/20 flex justify-between items-center">
                <div className="flex items-center gap-4">
                  <div className={`p-2.5 rounded-xl bg-slate-800/80 ${TASK_DOMAINS.find(d => d.id === currentTask.domain)?.color}`}>
                    {TASK_DOMAINS.find(d => d.id === currentTask.domain)?.icon}
                  </div>
                  <div>
                    <h2 className="font-bold text-white text-base">Current Task: {currentTask.domain.toUpperCase()}</h2>
                    <span className="text-[10px] text-slate-500 font-mono tracking-tighter">TASK_{Math.random().toString(36).substring(7).toUpperCase()}</span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[10px] font-bold text-slate-500 uppercase">Difficulty</span>
                  <div className="flex gap-1">
                    {[...Array(10)].map((_, i) => (
                      <div key={i} className={`w-1.5 h-4 rounded-full ${i < currentTask.difficulty ? 'bg-rf-teal shadow-[0_0_8px_rgba(0,191,165,0.4)]' : 'bg-slate-800'}`} />
                    ))}
                  </div>
                </div>
              </div>
              <div className="p-8">
                <p className="text-xl text-slate-100 leading-relaxed font-serif italic">"{currentTask.problem}"</p>
              </div>
            </div>
          )}

          {minerOutput && (
            <div className="bg-[#0f1115] border border-slate-800 rounded-xl overflow-hidden shadow-2xl animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/10">
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-rf-teal" /> Miner Reasoning Chain
                </h3>
              </div>
              <div className="p-8 space-y-8">
                {minerOutput.steps.map((step, i) => (
                  <div key={i} className="flex gap-6 group">
                    <div className="flex flex-col items-center">
                      <div className="w-8 h-8 rounded-xl bg-slate-800/80 border border-slate-700 flex items-center justify-center text-xs font-bold text-rf-teal group-hover:bg-rf-teal group-hover:text-black transition-all">
                        {step.id}
                      </div>
                      {i < minerOutput.steps.length - 1 && <div className="w-0.5 h-full bg-slate-800/50 my-2" />}
                    </div>
                    <div className="flex-grow pb-8 border-b border-slate-800/30 last:border-0 last:pb-0">
                      <div className="text-[10px] text-rf-teal font-bold mb-2 uppercase tracking-widest opacity-70">Process Step</div>
                      <p className="text-slate-300 leading-relaxed text-sm">{step.logic}</p>
                      {step.evidence && (
                        <div className="mt-3 text-[11px] bg-black/40 p-3 rounded-lg border border-slate-800/50 text-slate-400 flex items-start gap-2">
                          <Terminal className="w-3 h-3 shrink-0 mt-0.5 opacity-50" />
                          <span>Evidence: {step.evidence}</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                <div className="mt-6 p-6 bg-rf-teal/5 border border-rf-teal/20 rounded-xl shadow-inner">
                  <p className="text-white text-lg font-medium leading-snug">{minerOutput.final_answer}</p>
                </div>
              </div>
            </div>
          )}

          {validationData && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-[#0f1115] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl animate-in fade-in zoom-in-95 duration-1000 border-t-rf-orange/30">
                <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/30 flex justify-between items-center">
                  <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-rf-orange" /> Protocol Consensus
                  </h3>
                </div>
                <div className="p-6">
                  <div className="flex items-center gap-8 mb-8">
                    <div className="relative w-24 h-24 flex items-center justify-center shrink-0">
                      <svg className="w-full h-full -rotate-90">
                        <circle cx="48" cy="48" r="42" fill="none" stroke="#1e293b" strokeWidth="8" />
                        <circle
                          cx="48" cy="48" r="42" fill="none" stroke="#00BFA5" strokeWidth="8"
                          strokeDasharray={263.8}
                          strokeDashoffset={263.8 * (1 - (validationData.cms || 0))}
                          strokeLinecap="round"
                          className="transition-all duration-[2000ms] ease-out"
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-2xl font-black text-white">{(validationData.cms || 0).toFixed(2)}</span>
                        <span className="text-[8px] font-black text-slate-500 uppercase mt-0.5">CMS</span>
                      </div>
                    </div>
                    <div className="flex-grow space-y-3">
                      {[{ l: 'Accuracy', v: validationData.scores.A }, { l: 'Logic', v: validationData.scores.C }, { l: 'Novelty', v: validationData.scores.N }].map((s, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider">
                            <span className="text-slate-500">{s.l}</span>
                            <span className="text-rf-teal">{(s.v * 100).toFixed(0)}%</span>
                          </div>
                          <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full bg-rf-teal" style={{ width: `${s.v * 100}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <p className="text-xs text-slate-500 leading-relaxed font-serif italic border-l-2 border-rf-orange/30 pl-4 py-1">
                    "{validationData.critique}"
                  </p>
                </div>
              </div>

              <div className="flex flex-col gap-4">
                <button
                  onClick={startAdversarialDebate}
                  disabled={status === 'debating' || !!debateData}
                  className="flex-grow flex items-center justify-center gap-3 bg-rf-purple/10 border border-rf-purple/30 hover:bg-rf-purple/20 text-rf-purple rounded-xl p-6 transition-all group disabled:opacity-50"
                >
                  {status === 'debating' ? <Activity className="w-6 h-6 animate-spin" /> : <ShieldAlert className="w-6 h-6 group-hover:scale-110 transition-transform" />}
                  <div className="text-left">
                    <div className="font-bold text-sm uppercase tracking-widest">✨ Challenge Trace</div>
                    <div className="text-[10px] opacity-70">Trigger Adversarial Debate</div>
                  </div>
                </button>
                <button
                  onClick={generateReport}
                  disabled={status === 'reporting' || !!reportData}
                  className="flex-grow flex items-center justify-center gap-3 bg-rf-teal/10 border border-rf-teal/30 hover:bg-rf-teal/20 text-rf-teal rounded-xl p-6 transition-all group disabled:opacity-50"
                >
                  {status === 'reporting' ? <Activity className="w-6 h-6 animate-spin" /> : <FileText className="w-6 h-6 group-hover:scale-110 transition-transform" />}
                  <div className="text-left">
                    <div className="font-bold text-sm uppercase tracking-widest">✨ Export Intelligence</div>
                    <div className="text-[10px] opacity-70">Generate Business Report</div>
                  </div>
                </button>
              </div>
            </div>
          )}

          {debateData && (
            <div className="bg-[#0f1115] border border-rf-purple/30 rounded-2xl overflow-hidden shadow-2xl animate-in slide-in-from-right-4 duration-500">
              <div className="px-6 py-4 border-b border-rf-purple/20 bg-rf-purple/5 flex justify-between items-center">
                <h3 className="text-xs font-bold text-rf-purple uppercase tracking-widest flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4" /> Adversarial Debate Loop
                </h3>
              </div>
              <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-3">
                  <div className="text-[10px] font-bold text-rf-orange uppercase tracking-widest">Adversary Challenge</div>
                  <div className="bg-rf-orange/5 border border-rf-orange/20 p-4 rounded-lg text-xs leading-relaxed text-slate-300">
                    {debateData.challenge}
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="text-[10px] font-bold text-rf-teal uppercase tracking-widest">Miner Rebuttal</div>
                  <div className="bg-rf-teal/5 border border-rf-teal/20 p-4 rounded-lg text-xs leading-relaxed text-slate-300">
                    {debateData.rebuttal}
                  </div>
                </div>
                <div className="md:col-span-2 pt-4 border-t border-slate-800">
                  <p className="text-sm font-bold text-white uppercase">{debateData.verdict}</p>
                </div>
              </div>
            </div>
          )}

          {reportData && (
            <div className="bg-[#0f1115] border border-rf-teal/30 rounded-2xl overflow-hidden shadow-2xl animate-in zoom-in-95 duration-500">
              <div className="px-6 py-4 border-b border-rf-teal/20 bg-rf-teal/5 flex justify-between items-center">
                <h3 className="text-xs font-bold text-rf-teal uppercase tracking-widest flex items-center gap-2">
                  <FileText className="w-4 h-4" /> Enterprise Intelligence Report
                </h3>
                <span className="text-[10px] font-bold text-rf-teal">CONFIDENCE: {reportData.confidence_level}</span>
              </div>
              <div className="p-8 space-y-6">
                <div>
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Executive Summary</h4>
                  <p className="text-sm text-slate-300 leading-relaxed font-serif">{reportData.summary}</p>
                </div>
                <div>
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Strategic Impact</h4>
                  <p className="text-sm text-slate-400 italic">"{reportData.business_impact}"</p>
                </div>
              </div>
            </div>
          )}

          {!currentTask && (
            <div className="flex flex-col items-center justify-center py-40 text-center space-y-6 opacity-30">
              <Database className="w-20 h-20 text-slate-800" />
              <div>
                <h2 className="text-2xl font-bold tracking-tight text-white">Subnet Idle</h2>
                <p className="text-slate-500 max-w-xs mx-auto">Click "Dispatch Task" to initiate the verifiable reasoning lifecycle.</p>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default App;