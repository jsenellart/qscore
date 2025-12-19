"""
Run a Q-score instance on the D-Wave Simulated Annealing solver.
"""

import time
from collections import defaultdict
from functools import partial
from typing import Optional

import neal
from dwave.embedding.chain_strength import uniform_torque_compensation


def _sample_to_bitstring(sample, size: int) -> list[int]:
    bitstring = [0] * size
    for key, value in sample.items():
        bitstring[int(key)] = int(value)
    return bitstring


def run_SA(
    Q: defaultdict(int), size: int, num_reads: int, timeout: Optional[int] = None
) -> Optional[list[int]]:
    """
    Function that solves a Q-score instance on the D-Wave Simulated Annealing solver.

    Args:
        Q: QUBO-formulation of Q-score instance.
        size: Problem size.
        num_reads: Number of states to be read from solver.
        timeout: timeout parameter.

    Returns:
        Bitstring of the best found sample, or ``None`` if the timeout is exceeded.
    """
    start = time.time()

    sampler = neal.SimulatedAnnealingSampler()
    chain_strength = partial(uniform_torque_compensation, prefactor=2)
    sampleset = sampler.sample_qubo(
        Q,
        chain_strength=chain_strength,
        num_reads=num_reads,
        label=f"Problem-{size:2d}",
    )

    time_taken = time.time() - start
    if timeout is not None and time_taken > timeout:
        print("Failed to find a solution within timeout limit.")
        return None

    return _sample_to_bitstring(sampleset.first.sample, size)
