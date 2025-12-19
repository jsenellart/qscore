"""
Run a Q-score instance using the D-Wave Tabu solver.
"""
import time
from collections import defaultdict
from typing import Optional

from tabu import TabuSampler


def _sample_to_bitstring(sample, size: int) -> list[int]:
    bitstring = [0] * size
    for key, value in sample.items():
        bitstring[int(key)] = int(value)
    return bitstring


def run_tabu(Q: defaultdict(int), size: int, timeout: Optional[int] = None) -> Optional[list[int]]:
    """
    Function that solves a Q-score instance on the D-Wave tabu solver.

    Args:
        Q: QUBO-formulation of Q-score instance.
        size: Problem size.
        timeout: timeout parameter.

    Returns:
        Bitstring of the best found sample, or ``None`` if the timeout is exceeded.
    """
    start = time.time()

    sampler = TabuSampler()
    sampleset = sampler.sample_qubo(Q, label=f"Problem-{size:2d}")

    time_taken = time.time() - start
    if timeout is not None and time_taken > timeout:
        print("Failed to find a solution within timeout limit.")
        return None

    return _sample_to_bitstring(sampleset.first.sample, size)
