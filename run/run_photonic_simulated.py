"""
Run a Q-score instance on the photonic simulated solver.
"""

import time
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from networkx import Graph
from strawberryfields.apps import clique, sample


def _nodes_to_bitstring(nodes: List[int], size: int) -> list[int]:
    bitstring = [0] * size
    for node in nodes:
        bitstring[int(node)] = 1
    return bitstring


def run_photonic_simulated(
    G: Graph, size: int, n_samples: int, timeout: Optional[int] = None
) -> Tuple[Optional[list[int]], float, float]:
    """
    Function that solves a Q-score instance on a photonic simulator.
    Can only be used for Max-Clique problem instances.

    Args:
        G: Erdös-Renyí graph problem instance.
        size: Problem size.
        n_samples: Number of samples.
        timeout: timeout parameter.

    Returns:
        Bitstring of the best clique, together with ``(end_time, start_time)``. ``None``
        indicates that no feasible solution was found.
    """
    if G.size() == 0:
        bitstring = _nodes_to_bitstring([0], size) if size > 0 else []
        return bitstring, 0, 0

    # Extract the adjacency matrix
    adj = nx.to_numpy_array(G)

    s = sample.sample(adj, n_mean=50, n_samples=n_samples)

    # Create upper and lower bounds for max cliques to search for
    max_clicks = 2 * np.log2(size)
    min_clicks = max(1, 2 * np.log(size) - 3)
    s = sample.postselect(s, min_clicks, max_clicks)
    subgraphs = sample.to_subgraphs(s, G)

    # Find cliques
    start = time.time()  # We only consider classical runtime
    shrunk = [clique.shrink(sg, G) for sg in subgraphs]
    best_clique = max(shrunk, key=len) if shrunk else None

    end = time.time()
    if timeout is not None and end - start > timeout:
        print("Failed to find a solution within timeout limit.")
        return None, end, start

    if best_clique is None:
        return None, end, start

    return _nodes_to_bitstring(best_clique, size), end, start
