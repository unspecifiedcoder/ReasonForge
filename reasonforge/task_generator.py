"""
ReasonForge - Task Generator

Generates synthetic reasoning tasks across all 6 domains.
Includes trap task injection with known ground truth scores.
"""

from __future__ import annotations

import random
import uuid
from typing import List, Optional

from .types import Domain, Task, TaskSource, TRAP_RATE


# ──────────────────────────────────────────────
# Task Templates (5+ per domain)
# ──────────────────────────────────────────────

TASK_TEMPLATES = {
    Domain.MATHEMATICS: [
        "Prove that for all positive integers n, the sum 1+2+...+n = n(n+1)/2 using mathematical induction.",
        "Find all integer solutions to the Diophantine equation 3x + 5y = 23.",
        "Prove that sqrt(2) is irrational using proof by contradiction.",
        "Determine the eigenvalues and eigenvectors of the matrix [[2,1],[1,2]].",
        "Prove that every continuous function on [a,b] is Riemann integrable.",
        "Show that the sequence a_n = (1 + 1/n)^n converges to e.",
        "Prove the Cauchy-Schwarz inequality for vectors in R^n.",
    ],
    Domain.CODE: [
        "Implement a balanced binary search tree with O(log n) insertion, deletion, and search.",
        "Write a formally verified sorting algorithm and prove its correctness properties.",
        "Design a lock-free concurrent queue data structure with linearizability guarantees.",
        "Implement Dijkstra's shortest path algorithm with a Fibonacci heap priority queue.",
        "Write a type-safe parser combinator library with monadic composition.",
        "Implement a garbage collector using the tri-color marking algorithm.",
    ],
    Domain.LOGIC: [
        "All mammals are warm-blooded. Whales are mammals. Prove whales are warm-blooded using FOL.",
        "Given premises: All humans are mortal, Socrates is human. Prove Socrates is mortal via SMT.",
        "Prove that if A implies B and B implies C, then A implies C (hypothetical syllogism).",
        "Formalize and verify: No reptiles are mammals. All snakes are reptiles. Therefore no snakes are mammals.",
        "Prove by refutation: If P and (P implies Q), then Q (modus ponens) using SMT-LIB.",
        "Verify the validity of: All birds can fly. Penguins are birds. Therefore penguins can fly. (should FAIL)",
    ],
}

# Trap task templates (tasks with known ground truth / known formal translations)
TRAP_TEMPLATES = {
    Domain.MATHEMATICS: [
        ("What is 2+2?", 1.0),
        ("Is 7 a prime number? Provide a proof.", 0.95),
        ("Compute the derivative of x^3 with respect to x.", 1.0),
    ],
    Domain.CODE: [
        ("Write a function that returns the maximum of two integers.", 1.0),
        ("Implement binary search on a sorted array.", 0.95),
    ],
    Domain.LOGIC: [
        (
            "All cats are animals. Mittens is a cat. Is Mittens an animal? (Expected: VERIFIED)",
            1.0,
        ),
        (
            "If it rains then the ground is wet. It rains. Is the ground wet? (Expected: VERIFIED)",
            0.95,
        ),
    ],
}


class TaskGenerator:
    """Generates synthetic reasoning tasks with trap injection."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_tasks(self, count: int = 12) -> List[Task]:
        """
        Generate a set of tasks for an epoch.
        - int(count * TRAP_RATE) are trap tasks with ground_truth_score
        - Remaining are regular tasks across all 6 domains
        - Each task gets random difficulty 2-9
        - 5% chance of previously_unsolved = True
        """
        tasks = []
        trap_count = max(1, int(count * TRAP_RATE))
        regular_count = count - trap_count

        # Generate trap tasks
        for _ in range(trap_count):
            domain = self.rng.choice(list(Domain))
            trap_templates = TRAP_TEMPLATES.get(
                domain, TRAP_TEMPLATES[Domain.MATHEMATICS]
            )
            problem, truth = self.rng.choice(trap_templates)
            task = Task(
                task_id=str(uuid.uuid4()),
                problem=problem,
                domain=domain,
                difficulty=self.rng.randint(2, 5),  # Traps are easier
                source=TaskSource.TRAP,
                is_trap=True,
                ground_truth_score=truth,
                previously_unsolved=False,
            )
            tasks.append(task)

        # Generate regular tasks
        domains = list(Domain)
        for _ in range(regular_count):
            domain = self.rng.choice(domains)
            regular_templates = TASK_TEMPLATES[domain]
            problem = self.rng.choice(regular_templates)
            task = Task(
                task_id=str(uuid.uuid4()),
                problem=problem,
                domain=domain,
                difficulty=self.rng.randint(2, 9),
                source=TaskSource.SYNTHETIC,
                is_trap=False,
                ground_truth_score=None,
                previously_unsolved=self.rng.random() < 0.05,
            )
            tasks.append(task)

        self.rng.shuffle(tasks)
        return tasks
