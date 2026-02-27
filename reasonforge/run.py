"""
ReasonForge - CLI Runner

Entry point: python -m reasonforge.run [options]
Runs multi-epoch simulations with rich terminal output.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from .simulator import (
    EpochSimulator,
    create_default_miners,
    create_default_validators,
)
from .types import EMISSION_MINER_SHARE, EMISSION_VALIDATOR_SHARE, PEB_K, TASKS_PER_EPOCH, TRAP_RATE

BANNER = r"""
  ____                            _____
 |  _ \ ___  __ _ ___  ___  _ __|  ___|__  _ __ __ _  ___
 | |_) / _ \/ _` / __|/ _ \| '_ \ |_ / _ \| '__/ _` |/ _ \
 |  _ <  __/ (_| \__ \ (_) | | | |  _| (_) | | | (_| |  __/
 |_| \_\___|\__,_|___/\___/|_| |_|_|  \___/|_|  \__, |\___|
                                                  |___/
   The Decentralized Marketplace for Verifiable Intelligence
"""


def format_table(headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None) -> str:
    """Format a simple ASCII table."""
    if not col_widths:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    # Header
    header_line = ""
    for i, h in enumerate(headers):
        header_line += str(h).ljust(col_widths[i])
    sep = "-" * sum(col_widths)

    lines = [header_line, sep]
    for row in rows:
        line = ""
        for i, cell in enumerate(row):
            if i < len(col_widths):
                line += str(cell).ljust(col_widths[i])
        lines.append(line)

    return "\n".join(lines)


def run_simulation(epochs: int, emission: float, output: Optional[str] = None, seed: Optional[int] = None, verbose: bool = False):
    """Run the full multi-epoch simulation."""

    print(BANNER)
    print("=" * 70)
    print("  CONFIGURATION")
    print("=" * 70)
    print(f"  Epochs:          {epochs}")
    print(f"  Emission/epoch:  {emission} TAO")
    print(f"  Miner pool:      {emission * EMISSION_MINER_SHARE} TAO ({EMISSION_MINER_SHARE*100:.0f}%)")
    print(f"  Validator pool:  {emission * EMISSION_VALIDATOR_SHARE} TAO ({EMISSION_VALIDATOR_SHARE*100:.0f}%)")
    print(f"  Tasks/epoch:     {TASKS_PER_EPOCH}")
    print(f"  Trap rate:       {TRAP_RATE*100:.0f}%")
    print(f"  Top-K PEB:       {PEB_K}")
    print(f"  Seed:            {seed if seed is not None else 'random'}")
    print("=" * 70)
    print()

    # Create default roster
    miner_profiles, miner_states = create_default_miners(seed=seed)
    validator_profiles, validator_states = create_default_validators(seed=seed)

    all_epoch_results = []

    for epoch in range(1, epochs + 1):
        epoch_seed = (seed + epoch * 1000) if seed is not None else None

        sim = EpochSimulator(
            miner_profiles=miner_profiles,
            validator_profiles=validator_profiles,
            miner_states=miner_states,
            validator_states=validator_states,
            epoch_id=epoch,
            total_emission=emission,
            seed=epoch_seed,
        )

        result = sim.run_epoch()
        all_epoch_results.append(EpochSimulator.to_json(result))

        # Print epoch header
        print(f"\n{'='*70}")
        print(f"  EPOCH {epoch}/{epochs}")
        print(f"{'='*70}")
        print(f"  Tasks: {result.tasks_processed} | Traps: {result.traps_injected} | "
              f"Breakthroughs: {result.breakthroughs} | Avg CMS: {result.avg_cms:.4f}")
        print()

        # Miner leaderboard
        miner_headers = ["Rank", "Name", "S_epoch", "PEB", "Streak", "TAO", "Total", "Status"]
        miner_rows = []
        for m in result.miner_results:
            # Status indicators
            status = ""
            if m["rank"] <= 3:
                status += " *"  # star for top-3
            if m["trap_penalty"] < 1.0:
                status += " !"  # warning for trap penalty
            if m["breakthroughs"] > 0:
                status += " B"  # breakthrough

            miner_rows.append([
                str(m["rank"]),
                m["name"],
                f"{m['s_epoch']:.4f}",
                f"{m['peb']:.4f}",
                str(m["streak"]),
                f"{m['epoch_tao']:.2f}",
                f"{m['total_tao']:.2f}",
                status.strip(),
            ])

        print("  MINER LEADERBOARD")
        print("  " + "-" * 66)
        table = format_table(miner_headers, miner_rows)
        for line in table.split("\n"):
            print(f"  {line}")

        print()

        # Validator summary
        val_headers = ["Name", "Stake", "VAS", "Rep x", "TAO", "Slashed", "Health"]
        val_rows = []
        for v in result.validator_results:
            vas = v["vas"]
            if vas > 0.8:
                health = "[OK]"
            elif vas > 0.6:
                health = "[WARN]"
            else:
                health = "[BAD]"

            val_rows.append([
                v["name"],
                str(int(v["stake"])),
                f"{v['vas']:.4f}",
                f"{v['reputation']:.3f}",
                f"{v['epoch_tao']:.2f}",
                f"{v['slashed']:.4f}",
                health,
            ])

        print("  VALIDATOR SUMMARY")
        print("  " + "-" * 66)
        table = format_table(val_headers, val_rows)
        for line in table.split("\n"):
            print(f"  {line}")

        if verbose:
            print("\n  [Verbose] Per-task CMS values:")
            for m in result.miner_results:
                print(f"    {m['name']}: S_epoch={m['s_epoch']:.4f}")

    # Final standings
    print(f"\n\n{'='*70}")
    print("  FINAL STANDINGS AFTER ALL EPOCHS")
    print(f"{'='*70}")

    # Get final miner standings
    final_miners = sorted(
        all_epoch_results[-1]["miners"],
        key=lambda m: m["total_tao"],
        reverse=True,
    )
    print("\n  TOP MINERS (by Total TAO):")
    for i, m in enumerate(final_miners[:5], 1):
        star = " *" if i <= 3 else ""
        print(f"    {i}. {m['name']:20s} Total: {m['total_tao']:8.2f} TAO  "
              f"Streak: {m['streak']:2d}  PEB: {m['peb']:.4f}{star}")

    # Final validator standings
    print("\n  VALIDATORS:")
    for v in all_epoch_results[-1]["validators"]:
        vas = v["vas"]
        if vas > 0.8:
            indicator = "[OK]"
        elif vas > 0.6:
            indicator = "[WARN]"
        else:
            indicator = "[BAD]"
        print(f"    {v['name']:12s} Stake: {int(v['stake']):5d}  VAS: {v['vas']:.4f}  "
              f"Total: {v['total_tao']:.2f} TAO  Slashed: {v['slashed']:.4f}  {indicator}")

    # Key observations
    print("\n  KEY OBSERVATIONS:")
    # Find adversarial miners
    bottom = final_miners[-2:]
    for m in bottom:
        if m["trap_penalty"] < 1.0:
            print(f"    ! {m['name']} received trap penalty ({m['trap_penalty']:.2f})")

    # Find slashed validators
    for v in all_epoch_results[-1]["validators"]:
        if v["slashed"] > 0:
            print(f"    ! {v['name']} was slashed {v['slashed']:.4f} TAO")

    # Streak leaders
    streak_leaders = [m for m in final_miners if m["streak"] >= epochs]
    if streak_leaders:
        names = ", ".join(m["name"] for m in streak_leaders)
        print(f"    * Maintained full streak: {names}")

    print(f"\n{'='*70}")

    # Save output
    if output:
        output_data = {
            "config": {
                "epochs": epochs,
                "emission": emission,
                "miners": len(miner_profiles),
                "validators": len(validator_profiles),
                "seed": seed,
            },
            "epochs": all_epoch_results,
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results saved to: {output}")

    return all_epoch_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ReasonForge - Decentralized Verifiable Reasoning Simulator"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--emission", type=float, default=100.0, help="TAO per epoch (default: 100.0)")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Show per-task details")

    args = parser.parse_args()
    run_simulation(
        epochs=args.epochs,
        emission=args.emission,
        output=args.output,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
