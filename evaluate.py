"""Run a Q-score instance on one of the supported solver types."""

import argparse
import json
import multiprocessing as mp
import time
from typing import Callable, Optional

import networkx as nx
import numpy as np
from networkx import Graph
from qiskit_optimization.applications import Clique, Maxcut
from utils.max_clique import calculate_beta_max_clique, create_qubo_max_clique
from utils.max_cut import calculate_beta_max_cut, create_qubo_max_cut


def _objective_from_bitstring(Q_dict, bitstring):
    vector = np.asarray(bitstring, dtype=float)
    energy = 0.0
    for (i, j), coeff in Q_dict.items():
        if coeff == 0:
            continue
        if i == j:
            energy += coeff * vector[i]
        else:
            energy += coeff * vector[i] * vector[j]
    return -energy


def _build_qubo(problem_type: str, graph: Graph):
    if problem_type == "max-cut":
        return create_qubo_max_cut(graph)
    if problem_type == "max-clique":
        return create_qubo_max_clique(graph)
    raise NotImplementedError(
        f"Provided problem type {problem_type} is not implemented"
    )

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
            "Photonic_CVARVQE",
            "obliq-vqc",
            "obliq-static",
            "obliq-hybrid",
            "QAOA",
        ],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--min_timeout_size",
        help=(
            "Smallest problem size that enforces process-based timeouts for QAOA;"
            " smaller sizes only check the timeout after execution."
        ),
        type=int,
        required=False,
    )
    parser.add_argument(
        "--solver_options",
        help="JSON string with solver-specific keyword arguments (e.g. CVaR-VQE settings).",
        type=str,
        required=False,
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
    min_timeout_size: Optional[int] = None,
    solver_options: Optional[dict] = None,
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
        min_timeout_size: Smallest problem size that should enforce process-based timeouts.
        solver_options: Additional keyword arguments forwarded to solver backends.

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
    qubo_cache = None

    def get_qubo_dict():
        nonlocal qubo_cache
        if qubo_cache is None:
            qubo_cache = _build_qubo(problem_type, G)
        return qubo_cache

    enforce_timeout = (
        timeout is not None
        and timeout > 0
        and (min_timeout_size is None or size >= min_timeout_size)
    )
    if solver == "QAOA":
        from run.run_QAOA import run_QAOA

        if problem_type == "max-cut":
            max_cut = Maxcut(G)
            qp = max_cut.to_quadratic_program()
        elif problem_type == "max-clique":
            max_clique = Clique(G)
            qp = max_clique.to_quadratic_program()

        start_time = time.time()
        if enforce_timeout:
            qaoa_bitstring, _timed_out = run_with_timeout(
                run_QAOA,
                timeout,
                qp,
                provider,
                backend,
            )
        else:
            qaoa_bitstring = run_QAOA(
                qp,
                provider,
                backend,
            )
        end_time = time.time()
        if (
            not enforce_timeout
            and timeout is not None
            and timeout > 0
            and (end_time - start_time) > timeout
        ):
            objective_result = float("nan")
        elif (
            isinstance(qaoa_bitstring, float) and np.isnan(qaoa_bitstring)
        ) or qaoa_bitstring is None:
            objective_result = float("nan")
        else:
            objective_result = _objective_from_bitstring(
                get_qubo_dict(),
                qaoa_bitstring,
            )
    elif solver in ["Photonic_Simulation", "Photonic_quandela"]:
        from run.run_photonic_quandela import run_photonic_quandela
        from run.run_photonic_simulated import run_photonic_simulated

        if problem_type != "max-clique":
            raise ValueError(
                "Photonic solvers can only be used for Max-Clique problems."
            )

        if solver == "Photonic_Simulation":
            bitstring, end_time, start_time = run_photonic_simulated(
                G, size=size, n_samples=num_reads, timeout=timeout
            )
        elif solver == "Photonic_quandela":
            bitstring, end_time, start_time = run_photonic_quandela(
                G, size=size, backend=backend, n_samples=num_reads, timeout=timeout
            )

        if bitstring is None:
            objective_result = float("nan")
        else:
            objective_result = _objective_from_bitstring(
                get_qubo_dict(),
                bitstring,
            )
    elif solver == "Photonic_CVARVQE":
        from run.run_photonic_cvarvqe import run_photonic_cvarvqe

        solver_kwargs = solver_options or {}
        start_time = time.time()
        if enforce_timeout:
            cvar_bitstring, _timed_out = run_with_timeout(
                run_photonic_cvarvqe, 
                timeout, 
                G, 
                problem_type, 
                **solver_kwargs
            )
        else:
            cvar_bitstring = run_photonic_cvarvqe(
                G, problem_type, **solver_kwargs
            )
        end_time = time.time()
        if (
            not enforce_timeout
            and timeout is not None
            and timeout > 0
            and (end_time - start_time) > timeout
        ):
            objective_result = float("nan")
        elif (
            isinstance(cvar_bitstring, float) and np.isnan(cvar_bitstring)
        ) or cvar_bitstring is None:
            objective_result = float("nan")
        else:
            objective_result = _objective_from_bitstring(
                get_qubo_dict(),
                cvar_bitstring,
            )
    elif solver in {"obliq-vqc", "obliq-static", "obliq-hybrid"}:
        from run.run_obliq import run_obliq_solver, _qubo_dict_to_matrix, train_obliq_vqc_coeffs

        Q_dict = get_qubo_dict()
        Q_matrix = _qubo_dict_to_matrix(Q_dict, size)
        solver_kwargs = dict(solver_options or {})
        graph_mode = solver_kwargs.pop("graph_mode", 0)
        nsamples = solver_kwargs.pop("nsamples", 5000)
        num_rep = solver_kwargs.pop("num_rep", 10)
        coeffs = solver_kwargs.pop("coeffs", None)
        train_options = solver_kwargs.pop("train", None)
        real_machine = solver_kwargs.pop("real_machine", False)
        backend_override = solver_kwargs.pop("backend", None)
        token = solver_kwargs.pop("token", None)
        if solver_kwargs:
            raise ValueError(
                f"Unsupported ObliQ solver options: {', '.join(solver_kwargs.keys())}"
            )

        variant_map = {
            "obliq-vqc": "baseline-vqc",
            "obliq-static": "obliq-static",
            "obliq-hybrid": "obliq-hybrid",
        }

        start_time = time.time()
        training_history = None
        if train_options:
            variant_name = variant_map[solver]
            if variant_name == "obliq-static":
                raise ValueError("Coefficient training is only available for VQC or hybrid variants.")
            if isinstance(train_options, bool):
                train_options = {}
            trained_coeffs, training_history = train_obliq_vqc_coeffs(
                Q_matrix,
                variant=variant_name,
                initial_coeffs=coeffs,
                nsamples=nsamples,
                num_rep=num_rep,
                graph_mode=graph_mode,
                real_machine=real_machine,
                backend=backend_override or backend,
                token=token,
                **train_options,
            )
            coeffs = trained_coeffs

        obliq_args = dict(
            Q=Q_matrix,
            variant=variant_map[solver],
            nsamples=nsamples,
            num_rep=num_rep,
            graph_mode=graph_mode,
            coeffs=coeffs,
            real_machine=real_machine,
            backend=backend_override or backend,
            token=token,
        )

        if enforce_timeout:
            obliq_result, _timed_out = run_with_timeout(
                run_obliq_solver,
                timeout,
                **obliq_args,
            )
        else:
            obliq_result = run_obliq_solver(**obliq_args)
        if training_history and hasattr(obliq_result, "metadata"):
            obliq_result.metadata["training"] = training_history
        end_time = time.time()
        if (
            not enforce_timeout
            and timeout is not None
            and timeout > 0
            and (end_time - start_time) > timeout
        ):
            objective_result = float("nan")
        elif isinstance(obliq_result, float) and np.isnan(obliq_result):
            objective_result = float("nan")
        else:
            objective_result = _objective_from_bitstring(Q_dict, obliq_result.bitstring)
    else:
        Q = get_qubo_dict()

        solver_bitstring = None
        # Solve problem instance
        if solver == "Advantage_system4.1":
            from run.run_dwave_qpu import run_dwave_qpu

            start_time = time.time()
            solver_bitstring = run_dwave_qpu(Q, size, solver, num_reads, timeout)
            end_time = time.time()
        elif solver == "hybrid":
            from run.run_hybrid import run_hybrid

            start_time = time.time()
            solver_bitstring = run_hybrid(Q, size, timeout)
            end_time = time.time()
        elif solver == "Simulated_Annealing":
            from run.run_SA import run_SA

            start_time = time.time()
            solver_bitstring = run_SA(Q, size, num_reads, timeout)
            end_time = time.time()
        elif solver == "tabu":
            from run.run_tabu import run_tabu

            start_time = time.time()
            solver_bitstring = run_tabu(Q, size, timeout)
            end_time = time.time()
        else:
            raise NotImplementedError(f"Provided Solver {solver} is not implemented")

        if solver_bitstring is None or (
            isinstance(solver_bitstring, float) and np.isnan(solver_bitstring)
        ):
            objective_result = float("nan")
        else:
            objective_result = _objective_from_bitstring(Q, solver_bitstring)

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
    min_timeout_size = args.min_timeout_size
    solver_options = None
    if args.solver_options:
        try:
            solver_options = json.loads(args.solver_options)
        except json.JSONDecodeError as exc:  # pragma: no cover - CLI validation
            raise ValueError(f"Invalid solver options JSON: {exc}") from exc

    objective_result, beta, time_passed, G = main(
        problem_type=problem_type,
        size=size,
        solver=solver,
        timeout=timeout,
        seed=seed,
        num_reads=num_reads,
        provider=provider,
        backend=backend,
        min_timeout_size=min_timeout_size,
        solver_options=solver_options,
    )
    print(
        f"Finished problem size: {size}, "
        f"objective: {objective_result}, "
        f"beta: {beta:.2f}, "
        f"problem time: {time_passed:.2f}."
    )
