"""
Run a Q-score instance on the D-Wave hybrid solver.
"""
from collections import defaultdict

import dimod
from dwave.system import LeapHybridSampler


def _sample_to_bitstring(sample, size: int) -> list[int]:
    bitstring = [0] * size
    for key, value in sample.items():
        bitstring[int(key)] = int(value)
    return bitstring


def _get_hybrid_sampler() -> LeapHybridSampler:
    """Create a LeapHybridSampler instance only when needed."""
    return LeapHybridSampler(solver={"category": "hybrid"})


def run_hybrid(
    Q: defaultdict(int),
    size: int,
    timeout: int,
) -> list[int]:
    """
    Function that solves a Q-score instance on the D-Wave hybrid solver.

    Args:
        Q: QUBO-formulation of Q-score instance.
        size: Problem size.
        timeout: timeout parameter.

    Returns:
        Bitstring of the best found sample.
    """
    bqm = dimod.BQM.from_qubo(Q)
    sampler = _get_hybrid_sampler()
    sampleset = sampler.sample(
        bqm, label=f"Problem-{size:2d}", time_limit=timeout
    )
    return _sample_to_bitstring(sampleset.first.sample, size)
