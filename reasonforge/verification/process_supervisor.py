"""Step dependency graph and topological utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class StepDependencyGraph:
    """DAG of reasoning steps and their inter-dependencies.

    ``steps`` maps each step_id to its translation object (or any payload).
    ``edges`` maps each step_id to the list of step_ids it depends on.
    """

    steps: Dict[int, Any] = field(default_factory=dict)
    edges: Dict[int, List[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_dag(self) -> bool:
        """Return *True* if the graph is a valid DAG.

        Checks:
        1. Every dependency reference points to a known step.
        2. The graph contains no cycles (via Kahn's algorithm).
        """
        all_ids = set(self.steps.keys())

        # Check referential integrity.
        for step_id, deps in self.edges.items():
            if step_id not in all_ids:
                return False
            for dep in deps:
                if dep not in all_ids:
                    return False

        # Cycle detection via Kahn's algorithm.
        # Build in-degree map treating edges[s] as "s depends on these".
        # Reverse: if s depends on d, then d -> s is an edge in the DAG.
        in_degree: Dict[int, int] = {sid: 0 for sid in all_ids}
        reverse_adj: Dict[int, List[int]] = {sid: [] for sid in all_ids}

        for step_id, deps in self.edges.items():
            in_degree[step_id] = in_degree.get(step_id, 0) + len(deps)
            for dep in deps:
                reverse_adj.setdefault(dep, []).append(step_id)

        queue: deque[int] = deque(sid for sid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for child in reverse_adj.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return visited == len(all_ids)

    # ------------------------------------------------------------------
    # Topological ordering
    # ------------------------------------------------------------------

    def get_verification_order(self) -> List[int]:
        """Return a topological sort (leaves / independent steps first).

        Uses Kahn's algorithm. Raises ``ValueError`` if the graph contains
        a cycle.
        """
        all_ids = set(self.steps.keys())

        in_degree: Dict[int, int] = {sid: 0 for sid in all_ids}
        reverse_adj: Dict[int, List[int]] = {sid: [] for sid in all_ids}

        for step_id, deps in self.edges.items():
            in_degree[step_id] = in_degree.get(step_id, 0) + len(deps)
            for dep in deps:
                reverse_adj.setdefault(dep, []).append(step_id)

        queue: deque[int] = deque(
            sorted(sid for sid, deg in in_degree.items() if deg == 0)
        )
        order: List[int] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in sorted(reverse_adj.get(node, [])):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(all_ids):
            raise ValueError("Dependency graph contains a cycle")

        return order

    # ------------------------------------------------------------------
    # Cascade analysis
    # ------------------------------------------------------------------

    def invalidation_cascade(self, failed_step: int) -> Set[int]:
        """Return every step transitively downstream of *failed_step*.

        A step *s* is downstream if it (transitively) depends on
        *failed_step*.  The *failed_step* itself is **not** included in the
        returned set.

        Uses BFS over the reverse adjacency list (dep -> dependants).
        """
        # Build forward map: dep -> list of steps that depend on it.
        dependants: Dict[int, List[int]] = {sid: [] for sid in self.steps}
        for step_id, deps in self.edges.items():
            for dep in deps:
                dependants.setdefault(dep, []).append(step_id)

        visited: Set[int] = set()
        queue: deque[int] = deque()

        for child in dependants.get(failed_step, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)

        while queue:
            node = queue.popleft()
            for child in dependants.get(node, []):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        return visited
