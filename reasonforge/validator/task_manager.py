"""
ReasonForge - Task Manager

Manages task generation, queuing, dispatch, and assignment for validators.
Loads from benchmark database, handles API-submitted tasks, and injects traps.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from pathlib import Path
from typing import Dict, List

from ..types import TRAP_RATE, Domain, Task, TaskSource

logger = logging.getLogger("reasonforge.validator.task_manager")


class TaskManager:
    """Production task manager with benchmark DB + synthetic + API ingestion."""

    def __init__(
        self,
        benchmark_dir: str = "benchmarks",
        seed: int | None = None,
    ):
        self.rng = random.Random(seed)
        self.benchmark_dir = benchmark_dir
        self.benchmark_tasks: Dict[str, List[dict]] = {}
        self.used_task_ids: set[str] = set()
        self.api_queue: List[Task] = []

        self._load_benchmarks()

    def _load_benchmarks(self) -> None:
        """Load benchmark tasks from JSON files."""
        benchmark_path = Path(self.benchmark_dir)
        if not benchmark_path.exists():
            logger.warning("Benchmark directory not found: %s", self.benchmark_dir)
            return

        for domain_dir in benchmark_path.iterdir():
            if not domain_dir.is_dir():
                continue
            domain_name = domain_dir.name
            self.benchmark_tasks[domain_name] = []

            for json_file in domain_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        tasks = json.load(f)
                    if isinstance(tasks, list):
                        self.benchmark_tasks[domain_name].extend(tasks)
                    logger.info(
                        "Loaded %d tasks from %s",
                        len(tasks) if isinstance(tasks, list) else 0,
                        json_file,
                    )
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning("Failed to load %s: %s", json_file, e)

        total = sum(len(v) for v in self.benchmark_tasks.values())
        logger.info(
            "Loaded %d total benchmark tasks across %d domains", total, len(self.benchmark_tasks)
        )

    def generate_epoch_tasks(self, count: int = 12, trap_rate: float = TRAP_RATE) -> List[Task]:
        """Generate a balanced set of tasks for one epoch."""
        n_traps = max(1, int(count * trap_rate))
        n_regular = count - n_traps

        tasks = []

        # 1. Check for API-submitted tasks first
        api_tasks: list = []
        while self.api_queue and len(api_tasks) < n_regular // 2:
            api_tasks.append(self.api_queue.pop(0))
        tasks.extend(api_tasks)

        # 2. Sample benchmark tasks (balanced across domains)
        remaining = n_regular - len(api_tasks)
        tasks.extend(self._sample_balanced(remaining))

        # 3. Add trap problems
        tasks.extend(self._sample_traps(n_traps))

        # 4. Shuffle to hide traps
        self.rng.shuffle(tasks)

        return tasks

    def _sample_balanced(self, count: int) -> List[Task]:
        """Sample tasks balanced across domains."""
        tasks = []
        domains = list(Domain)
        per_domain = max(1, count // len(domains))

        for domain in domains:
            domain_key = domain.value
            available = self.benchmark_tasks.get(domain_key, [])

            # Filter out already-used tasks
            available = [t for t in available if t.get("task_id", "") not in self.used_task_ids]

            if available:
                sampled = self.rng.sample(available, min(per_domain, len(available)))
                for task_data in sampled:
                    task = self._task_from_benchmark(task_data, domain)
                    tasks.append(task)
                    self.used_task_ids.add(task.task_id)
            else:
                # Fall back to synthetic generation
                tasks.extend(self._generate_synthetic(per_domain, domain))

        # If we still need more, add synthetic tasks
        while len(tasks) < count:
            domain = self.rng.choice(domains)
            tasks.extend(self._generate_synthetic(1, domain))

        return tasks[:count]

    def _sample_traps(self, count: int) -> List[Task]:
        """Sample trap problems with known ground truth."""
        traps = []
        for domain_key, task_list in self.benchmark_tasks.items():
            for task_data in task_list:
                if task_data.get("is_trap", False):
                    traps.append(task_data)

        if traps:
            sampled = self.rng.sample(traps, min(count, len(traps)))
            return [
                self._task_from_benchmark(t, Domain(t.get("domain", "mathematics")))
                for t in sampled
            ]

        # Fallback: generate simple trap tasks
        return self._generate_synthetic_traps(count)

    def _task_from_benchmark(self, data: dict, domain: Domain) -> Task:
        """Convert a benchmark JSON dict to a Task object."""
        return Task(
            task_id=data.get("task_id", str(uuid.uuid4())),
            problem=data.get("problem", ""),
            domain=domain,
            difficulty=data.get("difficulty", 5),
            timeout_seconds=data.get("timeout_seconds", 300),
            source=TaskSource.TRAP if data.get("is_trap") else TaskSource.BENCHMARK,
            is_trap=data.get("is_trap", False),
            ground_truth_score=data.get("ground_truth_score"),
            previously_unsolved=data.get("previously_unsolved", False),
        )

    def _generate_synthetic(self, count: int, domain: Domain) -> List[Task]:
        """Generate synthetic tasks for a domain."""
        from ..task_generator import TASK_TEMPLATES

        templates = TASK_TEMPLATES.get(domain, [])
        tasks = []
        for _ in range(count):
            if templates:
                problem = self.rng.choice(templates)
            else:
                problem = f"Solve a {domain.value} reasoning problem."
            tasks.append(
                Task(
                    task_id=str(uuid.uuid4()),
                    problem=problem,
                    domain=domain,
                    difficulty=self.rng.randint(2, 9),
                    source=TaskSource.SYNTHETIC,
                    is_trap=False,
                    previously_unsolved=self.rng.random() < 0.05,
                )
            )
        return tasks

    def _generate_synthetic_traps(self, count: int) -> List[Task]:
        """Generate synthetic trap tasks with known answers."""
        from ..task_generator import TRAP_TEMPLATES

        traps = []
        for _ in range(count):
            domain = self.rng.choice(list(Domain))
            templates = TRAP_TEMPLATES.get(domain, TRAP_TEMPLATES[Domain.MATHEMATICS])
            problem, truth = self.rng.choice(templates)
            traps.append(
                Task(
                    task_id=str(uuid.uuid4()),
                    problem=problem,
                    domain=domain,
                    difficulty=self.rng.randint(2, 5),
                    source=TaskSource.TRAP,
                    is_trap=True,
                    ground_truth_score=truth,
                )
            )
        return traps

    def submit_api_task(
        self, problem: str, domain: str | None = None, difficulty: int | None = None
    ) -> Task:
        """Accept an external task submission via the API gateway."""
        task = Task(
            task_id=str(uuid.uuid4()),
            problem=problem,
            domain=Domain(domain) if domain else self.rng.choice(list(Domain)),
            difficulty=difficulty or self.rng.randint(3, 8),
            source=TaskSource.USER_API,
        )
        self.api_queue.append(task)
        return task
