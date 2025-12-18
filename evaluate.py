"""Run a Q-score instance on one of the six solver types."""

import argparse
import multiprocessing as mp
import time
from typing import Callable, Optional

import networkx as nx
import numpy as np
from networkx import Graph
from qiskit_optimization.applications import Clique, Maxcut
from utils.max_clique import calculate_beta_max_clique, create_qubo_max_clique
from utils.max_cut import calculate_beta_max_cut, create_qubo_max_cut


def parse_args() -> argparse.Namespace:
    """
    Parser function.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--problem",
        help="String of the problem type.",
        choices=[
            "max-cut",
            "max-clique",
        ],
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--size",
        help="Problem size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-seed",
        "--seed",
        help="Random seed",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-t",
        "--timeout",
        help="Solver timeout",
        type=int,
        default=None,
        required=False,
    )
    (
        parser.add_argument(
            "-n",
            "--num_reads",
            help="Number of reads/samples in case of a QPU or Simulated Annealing solver",
            type=int,
            required=False,
        ),
    )
    parser.add_argument(
        "-provider",
        "--provider",
        help="Name of hardware provider in case `QAOA` is selected.",
        choices=[
            "local simulator",
            "ibm",
            "qi",
        ],
        type=str,
        required=False,
    )
    parser.add_argument(
        "-backend",
        "--backend",
        help="Name of backend in case `QAOA` or `Photonic_quandela` is selected.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-solver",
        "--solver",
        help="String of the D-Wave solver.",
        choices=[
            "Advantage_system4.1",
            "hybrid",
            "tabu",
            "Simulated_Annealing",
            "Photonic_Simulation",
            "Photonic_quandela",
            "QAOA",
        ],
        type=str,
        required=True,
    )

    args = parser.parse_args()
    return args


def _process_wrapper(
    queue: mp.Queue,
    func: Callable,
    args: tuple,
    kwargs: dict,
):
    """Execute ``func`` and push (success, payload) to a queue."""

    try:
        result = func(*args, **kwargs)
        queue.put((True, result))
    except Exception as exc:  # pragma: no cover - propagated to parent
        queue.put((False, exc))


def run_with_timeout(
    func: Callable,
    timeout: Optional[int],
    *args,
    **kwargs,
):
    """Execute ``func`` enforcing a soft timeout.

    Returns a tuple ``(result, timed_out)`` where ``timed_out`` indicates whether
    the timeout was hit. When ``timeout`` is ``None`` or non-positive, the
    callable executes directly.
    """

    if timeout is None or timeout <= 0:
        return func(*args, **kwargs), False

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_process_wrapper,
        args=(queue, func, args, kwargs),
        daemon=True,
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        queue.close()
        queue.join_thread()
        return np.nan, True

    if queue.empty():
        queue.close()
        queue.join_thread()
        return np.nan, True

    success, payload = queue.get()
    queue.close()
    queue.join_thread()
    if success:
        return payload, False
    raise payload


def sample_non_empty_erdos_renyi_graph(
    size: int, probability: float, seed: Optional[int]
):
    """Generate an Erdős–Rényi graph with given edge probability that has edges."""

    max_attempts = 100
    attempt_seed = seed

    for _ in range(max_attempts):
        graph = nx.erdos_renyi_graph(size, probability, seed=attempt_seed)
        if graph.number_of_edges() > 0:
            return graph
        if attempt_seed is None:
            attempt_seed = np.random.randint(0, 1000000)
        else:
            attempt_seed += 1

    raise ValueError("Failed to sample a graph with at least one edge.")


def main(
    problem_type: str,
    size: int,
    solver: str,
    timeout: Optional[int] = None,
    seed: Optional[int] = None,
    num_reads: Optional[int] = None,
    provider: Optional[str] = None,
    backend: Optional[str] = None,
) -> tuple[float, float, float, Graph]:
    """
    Main routine to evaluate a Q-score instance.

    Args:
        problem_type: string of problem type (max-cut or max-clique).
        size: size of the problem instance.
        solver: type of solver being used.
        timeout: maximum time a solver can use.
        seed: random seed for reproducibility.
        num_reads: Number of reads/samples in case of a QPU or Simulated Annealing solver.
        provider: Name of hardware provider in case QAOA is selected.
        backend: Name of backend in case QAOA or photonic is selected.

    Returns:
        objective_result: solution to max-cut or max-clique.
        beta: found beta.
        time: time it took the solver to solve problem instance.
        G: The specific Erdös-Renyí graph.

    Raises:
        NotImplementedError: In case unimplemented problem type is provided.
        NotImplementedError: In case unimplemented solver is provided.
        ValueError: In case missing or invalid solver arguments are provided.
    """
    if num_reads is None and solver in [
        "Advantage_system4.1",
        "Simulated_Annealing",
        "Photonic_Simulation",
        "Photonic_quandela",
    ]:
        raise ValueError("num_reads has not been submitted while required by solver.")

    if seed is None:
        seed = np.random.randint(100000)
    G = sample_non_empty_erdos_renyi_graph(size, 1 / 2, seed)
    if solver == "QAOA":
        from run.run_QAOA import run_QAOA

        if problem_type == "max-cut":
            max_cut = Maxcut(G)
            qp = max_cut.to_quadratic_program()
        elif problem_type == "max-clique":
            max_clique = Clique(G)
            qp = max_clique.to_quadratic_program()

        start_time = time.time()
        objective_result, _timed_out = run_with_timeout(
            run_QAOA,
            timeout,
            qp,
            provider,
            backend,
        )
        end_time = time.time()
    elif solver in ["Photonic_Simulation", "Photonic_quandela"]:
        from run.run_photonic_quandela import run_photonic_quandela
        from run.run_photonic_simulated import run_photonic_simulated

        if problem_type != "max-clique":
            raise ValueError(
                "Photonic solvers can only be used for Max-Clique problems."
            )

        if solver == "Photonic_Simulation":
            objective_result, end_time, start_time = run_photonic_simulated(
                G, size=size, n_samples=num_reads, timeout=timeout
            )
        elif solver == "Photonic_quandela":
            objective_result, end_time, start_time = run_photonic_quandela(
                G, size=size, backend=backend, n_samples=num_reads, timeout=timeout
            )
    else:
        # Create qubo:
        if problem_type == "max-cut":
            Q = create_qubo_max_cut(G)
        elif problem_type == "max-clique":
            Q = create_qubo_max_clique(G)
        else:
            raise NotImplementedError(
                f"Provided problem type {problem_type} is not implemented"
            )

        # Solve problem instance
        if solver == "Advantage_system4.1":
            from run.run_dwave_qpu import run_dwave_qpu

            start_time = time.time()
            objective_result = run_dwave_qpu(Q, size, solver, num_reads, timeout)
            end_time = time.time()
        elif solver == "hybrid":
            from run.run_hybrid import run_hybrid

            start_time = time.time()
            objective_result = run_hybrid(Q, size, timeout)
            end_time = time.time()
        elif solver == "Simulated_Annealing":
            from run.run_SA import run_SA

            start_time = time.time()
            objective_result = run_SA(Q, size, num_reads, timeout)
            end_time = time.time()
        elif solver == "tabu":
            from run.run_tabu import run_tabu

            start_time = time.time()
            objective_result = run_tabu(Q, size, timeout)
            end_time = time.time()
        else:
            raise NotImplementedError(f"Provided Solver {solver} is not implemented")

    # Calculate beta
    if objective_result is None or (
        isinstance(objective_result, float) and np.isnan(objective_result)
    ):
        beta = 0.0
    elif problem_type == "max-cut":
        beta = calculate_beta_max_cut(size, objective_result)
    elif problem_type == "max-clique":
        beta = calculate_beta_max_clique(size, objective_result)

    return objective_result, beta, end_time - start_time, G


if __name__ == "__main__":
    args = parse_args()
    problem_type = args.problem
    size = args.size
    seed = args.seed
    timeout = args.timeout
    num_reads = args.num_reads
    solver = args.solver
    provider = args.provider
    backend = args.backend

    objective_result, beta, time_passed, G = main(
        problem_type=problem_type,
        size=size,
        solver=solver,
        timeout=timeout,
        seed=seed,
        num_reads=num_reads,
        provider=provider,
        backend=backend,
    )
    print(
        f"Finished problem size: {size}, "
        f"objective: {objective_result}, "
        f"beta: {beta:.2f}, "
        f"problem time: {time_passed:.2f}."
    )
