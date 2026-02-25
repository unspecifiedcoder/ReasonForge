"""
ReasonForge - Task Generator

Generates synthetic reasoning tasks across all 6 domains.
Includes trap task injection with known ground truth scores.
"""

from __future__ import annotations

import random
import uuid
from typing import List

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
    Domain.SCIENTIFIC: [
        "Model the SIR epidemic dynamics for a population of 10,000 with R0=2.5 and recovery rate 0.1.",
        "Analyze the stability of the Lorenz system at its equilibrium points.",
        "Design an experiment to test whether a new drug reduces blood pressure, controlling for confounds.",
        "Simulate protein folding energy minimization using a simplified lattice model.",
        "Derive the Navier-Stokes equations for incompressible fluid flow in 2D.",
        "Model predator-prey dynamics using Lotka-Volterra equations and analyze stability.",
    ],
    Domain.STRATEGIC: [
        "Find the Nash equilibrium in a 3-player game with the given payoff matrices.",
        "Solve the traveling salesman problem for 8 cities using branch and bound.",
        "Design an optimal auction mechanism for selling spectrum licenses to 5 bidders.",
        "Formulate and solve a linear program for supply chain optimization with 3 plants and 5 warehouses.",
        "Analyze the Prisoner's Dilemma iterated 100 times with discounting and find the optimal strategy.",
    ],
    Domain.CAUSAL: [
        "Given a causal DAG X->Z->Y with confounder W->X, W->Y, identify the causal effect of X on Y.",
        "Apply the do-calculus to determine if P(Y|do(X)) is identifiable from observational data.",
        "Design a natural experiment to estimate the causal effect of education on earnings.",
        "Determine the mediating effect of Z in the path X->Z->Y using the front-door criterion.",
        "Use instrumental variables to estimate the causal effect of smoking on lung cancer.",
        "Construct a causal DAG for the relationship between exercise, diet, weight, and health outcomes.",
    ],
    Domain.ETHICAL: [
        "Analyze the trolley problem from utilitarian, deontological, and virtue ethics perspectives.",
        "Evaluate the ethical implications of autonomous vehicles making life-or-death decisions.",
        "Apply Rawls' veil of ignorance to design a fair resource allocation system for healthcare.",
        "Analyze the ethical trade-offs in deploying facial recognition technology in public spaces.",
        "Evaluate the moral status of AI systems using multiple ethical frameworks.",
        "Assess the ethical implications of gene editing in human embryos from 3+ moral perspectives.",
    ],
}

# Trap task templates (tasks with known ground truth)
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
    Domain.SCIENTIFIC: [
        ("Calculate the kinetic energy of a 2kg object moving at 3m/s.", 1.0),
        ("What is the pH of pure water at 25C?", 0.95),
    ],
    Domain.STRATEGIC: [
        ("In a zero-sum game with payoff matrix [[1,-1],[-1,1]], find the Nash equilibrium.", 0.9),
    ],
    Domain.CAUSAL: [
        ("In X->Y with no confounders, what is the adjustment set for estimating causal effect?", 0.95),
    ],
    Domain.ETHICAL: [
        ("List three major ethical frameworks used in moral philosophy.", 0.9),
    ],
}


class TaskGenerator:
    """Generates synthetic reasoning tasks with trap injection."""

    def __init__(self, seed: int = None):
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
            templates = TRAP_TEMPLATES.get(domain, TRAP_TEMPLATES[Domain.MATHEMATICS])
            problem, truth = self.rng.choice(templates)
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
            templates = TASK_TEMPLATES[domain]
            problem = self.rng.choice(templates)
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
