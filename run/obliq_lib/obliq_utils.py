"""Utility helpers shared by the ObliQ photonic solvers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def qubo_objective(Q: np.ndarray, x: Sequence[int]) -> float:
    """Evaluate the QUBO objective for a binary vector ``x``."""

    vector = np.asarray(x, dtype=float)
    return float(vector @ Q @ vector)


def find_locations(binary_list: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Return indices of ones and zeros in ``binary_list``."""

    ones = [index for index, value in enumerate(binary_list) if value == 1]
    zeros = [index for index, value in enumerate(binary_list) if value == 0]
    return ones, zeros


def string_to_list(state: Iterable[int]) -> List[int]:
    """Convert a sampler output state into a list of integers."""

    cleaned = str(state)[1:-1]
    return [int(x.strip()) for x in cleaned.split(",") if x.strip()]


def solve_qubo(Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Brute-force QUBO solve used for very small problems (testing only)."""

    size = Q.shape[0]
    best_solution: np.ndarray | None = None
    best_value = float("inf")
    for combination in product([0, 1], repeat=size):
        x = np.array(combination)
        value = qubo_objective(Q, x)
        if value < best_value:
            best_value = value
            best_solution = x
    assert best_solution is not None
    return best_solution, best_value


def _sorted_indices_desc(values: np.ndarray) -> List[int]:
    return np.argsort(values)[::-1].tolist()


def solution_guesses(avg_num: np.ndarray, graph_mode: int = 0) -> List[List[int]]:
    """Generate candidate bitstrings from average photon counts.

    ``graph_mode`` mirrors the behaviour of the original ObliQ code:
      * ``0`` – generic QUBO (prefixes of sorted averages)
      * ``1`` – grouped variables (every three indices share a group)
      * ``else`` – matrix problems requiring unique row/column selections
    """

    if graph_mode == 0:
        ordering = _sorted_indices_desc(avg_num)
        return [ordering[: i + 1] for i in range(len(ordering))]

    if graph_mode == 1:
        ordering = _sorted_indices_desc(avg_num)
        solution: List[int] = []
        seen_groups = set()
        for idx in ordering:
            group = int(idx / 3)
            if group not in seen_groups:
                solution.append(idx)
                seen_groups.add(group)
        return [solution[: i + 1] for i in range(len(solution))]

    ordering = _sorted_indices_desc(avg_num)
    size = int(np.sqrt(len(avg_num)))
    solution = []
    selected_rows = set()
    selected_cols = set()
    for idx in ordering:
        row = idx // size
        col = idx % size
        if row not in selected_rows and col not in selected_cols:
            solution.append(idx)
            selected_rows.add(row)
            selected_cols.add(col)
    return [solution[: i + 1] for i in range(len(solution))]


@dataclass
class ObliqResult:
    objective: float
    bitstring: List[int]
    metadata: dict
