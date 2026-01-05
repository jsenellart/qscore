"""Wrapper for the Lucy GenericVQE photonic solver."""

from __future__ import annotations

from typing import Optional, Sequence

import networkx as nx
import numpy as np
import perceval as pcvl
from networkx import Graph

from run.photonic_vqe_lib.vqe_generic import GenericVqe


def _graph_to_matrix_maxcut(graph: Graph) -> np.ndarray:
    matrix = nx.to_numpy_array(graph)
    np.fill_diagonal(matrix, [-graph.degree[node] for node in graph.nodes()])
    return matrix


def _graph_to_matrix_max_clique(graph: Graph) -> np.ndarray:
    matrix = np.zeros((len(graph), len(graph)))
    np.fill_diagonal(matrix, -1)
    complement = nx.complement(graph)
    for u, v in complement.edges():
        matrix[u, v] = 2
        matrix[v, u] = 2
    return matrix


def _build_processor(backend: Optional[str], token: Optional[str]) -> pcvl.components.AProcessor:
    if backend:
        if token is None:
            raise ValueError("Quandela token must be provided for remote Photonic_VQE backends.")
        processor = pcvl.RemoteProcessor(backend, token)
        processor._rpc_handler.request_timeout = 60
        return processor
    return pcvl.Processor("SLOS")


def run_photonic_vqe(
    graph: Graph,
    problem_type: str,
    shots: int = 10_000,
    method: str = "COBYLA",
    rots: Sequence[str] | None = None,
    entanglement_type: str = "linear",
    ctype: str = "cx",
    input_state: Optional[str] = None,
    backend: Optional[str] = None,
    token: Optional[str] = None,
    max_iter: int = 200,
    cvar_alpha: Optional[float] = None,
) -> list[int]:
    """Solve a Q-score instance using the Lucy Generic VQE implementation."""

    if problem_type not in {"max-cut", "max-clique"}:
        raise ValueError("Photonic_VQE only supports max-cut and max-clique problems.")

    rots = list(rots) if rots else ["Y"]
    processor = _build_processor(backend, token)

    if problem_type == "max-cut":
        qubo_matrix = _graph_to_matrix_maxcut(graph)
    else:
        qubo_matrix = _graph_to_matrix_max_clique(graph)

    vqe = GenericVqe(
        processor=processor,
        qubo_matrix=qubo_matrix.tolist(),
        shots=shots,
        rots=rots,
        entanglement_type=entanglement_type,
        method=method,
        ctype=ctype,
        input_state=input_state,
        maxiter=max_iter,
        alpha=cvar_alpha,
    )

    result = vqe.execute_optimization(maxiter=max_iter)
    bitstring = result.get("optimal bitstring")
    if not bitstring:
        raise RuntimeError("Photonic_VQE did not return a valid bitstring.")

    return [int(bit) for bit in bitstring]
