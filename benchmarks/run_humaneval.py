"""Benchmark the ReasonForge code verification pipeline against HumanEval.

Runs each of the 164 HumanEval canonical solutions through the
VerificationPipeline and collects per-task and aggregate metrics:
  - syntax pass rate
  - security scan results (risk levels, rule hit counts)
  - test generation counts
  - test pass/fail rates
  - overall verdict distribution
  - confidence score distribution
  - per-stage latency

Results are written to benchmarks/results/humaneval_results.json.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from human_eval.data import read_problems  # type: ignore[import-untyped]

from reasonforge.codeverify.pipeline import VerificationPipeline
from reasonforge.codeverify.security_scanner import SecurityScanner


async def run_benchmark() -> Dict[str, Any]:
    """Run all HumanEval problems through the pipeline."""
    problems = read_problems()
    pipeline = VerificationPipeline(timeout=30)
    scanner = SecurityScanner()

    results: List[Dict[str, Any]] = []
    total = len(problems)

    print(f"Running ReasonForge pipeline on {total} HumanEval problems...")
    print()

    for i, (task_id, problem) in enumerate(problems.items()):
        # Combine prompt + canonical solution to get the full function
        source_code = problem["prompt"] + problem["canonical_solution"]

        start = time.perf_counter()
        try:
            report = await pipeline.verify(source_code)
            elapsed_ms = (time.perf_counter() - start) * 1000

            result: Dict[str, Any] = {
                "task_id": task_id,
                "entry_point": problem["entry_point"],
                "code_lines": len(source_code.splitlines()),
                "syntax_valid": report.syntax_valid,
                "syntax_error": report.syntax_error,
                "security_risk_level": report.security_risk_level,
                "security_issue_count": report.security_issue_count,
                "security_issues": report.security_issues,
                "tests_generated": report.tests_generated,
                "tests_passed": report.tests_passed,
                "tests_failed": report.tests_failed,
                "test_pass_rate": report.test_pass_rate,
                "verdict": report.verdict,
                "confidence_score": report.confidence_score,
                "failure_reasons": report.failure_reasons,
                "elapsed_ms": round(elapsed_ms, 1),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            result = {
                "task_id": task_id,
                "entry_point": problem["entry_point"],
                "code_lines": len(source_code.splitlines()),
                "error": str(exc),
                "verdict": "ERROR",
                "elapsed_ms": round(elapsed_ms, 1),
            }

        results.append(result)

        # Progress output
        verdict = result.get("verdict", "ERROR")
        conf = result.get("confidence_score", 0.0)
        ms = result["elapsed_ms"]
        status = "PASS" if verdict == "PASSED" else verdict[:4]
        print(
            f"  [{i+1:3d}/{total}] {task_id:<20s} "
            f"{status:<6s} conf={conf:.2f}  {ms:.0f}ms"
        )

    # ── Aggregate statistics ──────────────────────────────────────
    print()
    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    # Verdict distribution
    verdicts = [r["verdict"] for r in results]
    verdict_counts = {v: verdicts.count(v) for v in set(verdicts)}

    # Syntax
    syntax_pass = sum(1 for r in valid_results if r["syntax_valid"])
    syntax_fail = sum(1 for r in valid_results if not r["syntax_valid"])

    # Security
    risk_levels = [r["security_risk_level"] for r in valid_results]
    risk_counts = {v: risk_levels.count(v) for v in set(risk_levels)}

    # Collect all security rule hits
    all_rule_ids: List[str] = []
    for r in valid_results:
        for issue in r.get("security_issues", []):
            all_rule_ids.append(issue.get("rule_id", "unknown"))
    rule_counts = {rid: all_rule_ids.count(rid) for rid in set(all_rule_ids)}

    # Tests
    tests_generated = [r["tests_generated"] for r in valid_results]
    tests_passed_list = [r["tests_passed"] for r in valid_results]
    tests_failed_list = [r["tests_failed"] for r in valid_results]
    pass_rates = [r["test_pass_rate"] for r in valid_results if r["tests_generated"] > 0]

    # Confidence
    confidences = [r["confidence_score"] for r in valid_results]

    # Latency
    latencies = [r["elapsed_ms"] for r in results]

    summary = {
        "dataset": "HumanEval",
        "total_problems": total,
        "pipeline_errors": len(error_results),
        "verdict_distribution": verdict_counts,
        "syntax": {
            "pass": syntax_pass,
            "fail": syntax_fail,
            "pass_rate": round(syntax_pass / max(len(valid_results), 1), 4),
        },
        "security": {
            "risk_distribution": risk_counts,
            "rule_hit_counts": dict(sorted(rule_counts.items())),
            "total_issues_found": len(all_rule_ids),
        },
        "tests": {
            "mean_generated": round(statistics.mean(tests_generated), 2)
            if tests_generated
            else 0,
            "median_generated": round(statistics.median(tests_generated), 1)
            if tests_generated
            else 0,
            "total_generated": sum(tests_generated),
            "total_passed": sum(tests_passed_list),
            "total_failed": sum(tests_failed_list),
            "mean_pass_rate": round(statistics.mean(pass_rates), 4)
            if pass_rates
            else 0,
            "median_pass_rate": round(statistics.median(pass_rates), 4)
            if pass_rates
            else 0,
        },
        "confidence": {
            "mean": round(statistics.mean(confidences), 4) if confidences else 0,
            "median": round(statistics.median(confidences), 4) if confidences else 0,
            "stdev": round(statistics.stdev(confidences), 4)
            if len(confidences) > 1
            else 0,
            "min": round(min(confidences), 4) if confidences else 0,
            "max": round(max(confidences), 4) if confidences else 0,
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else 0,
            "median": round(statistics.median(latencies), 1) if latencies else 0,
            "p95": round(
                sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1
            ),
            "min": round(min(latencies), 1) if latencies else 0,
            "max": round(max(latencies), 1) if latencies else 0,
        },
    }

    # Print summary
    print(f"Total problems:     {total}")
    print(f"Pipeline errors:    {len(error_results)}")
    print(f"Verdict distrib:    {verdict_counts}")
    print(f"Syntax pass rate:   {summary['syntax']['pass_rate']:.1%}")
    print(f"Security risk dist: {risk_counts}")
    print(f"Security rules hit: {rule_counts}")
    print(f"Mean tests/func:    {summary['tests']['mean_generated']:.1f}")
    print(f"Mean test pass rate:{summary['tests']['mean_pass_rate']:.1%}")
    print(f"Mean confidence:    {summary['confidence']['mean']:.3f}")
    print(f"Mean latency:       {summary['latency_ms']['mean']:.0f} ms")
    print(f"Median latency:     {summary['latency_ms']['median']:.0f} ms")
    print(f"P95 latency:        {summary['latency_ms']['p95']:.0f} ms")

    # Save results
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    output = {"summary": summary, "per_task": results}
    output_path = output_dir / "humaneval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    asyncio.run(run_benchmark())
