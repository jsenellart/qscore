"""Run the Q-score instances for various sizes and save data."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple

import networkx as nx
import numpy as np
from networkx import Graph
from networkx.algorithms.approximation.maxcut import one_exchange

from evaluate import main


def _run_single_instance(
    args: Tuple[
        str,
        int,
        str,
        int,
        Optional[int],
        Optional[int],
        Optional[str],
        Optional[str],
        bool,
    ]
) -> Tuple[float, float, Optional[Graph]]:
    """Helper suitable for multiprocessing pools."""

    (
        problem_type,
        size,
        solver,
        timeout,
        seed,
        num_reads,
        provider,
        backend,
        keep_graph,
    ) = args
    objective_result, _, elapsed_time, graph = main(
        problem_type=problem_type,
        size=size,
        timeout=timeout,
        solver=solver,
        seed=seed,
        num_reads=num_reads,
        provider=provider,
        backend=backend,
    )
    return objective_result, elapsed_time, graph if keep_graph else None


def calculate_qscore(
    nb_instances_per_size: int,
    size_range: list,
    file_name: str,
    include_exact_results: bool,
    problem_type: str,
    timeout: int,
    solver: str,
    seed: int,
    num_reads: int,
    provider: str,
    backend: str,
    parallel_workers: int = 1,
):
    """
    Run multiple Q-score instances for various problem sizes.
    Results are written to a json file.

    Args:
        nb_instances_per_size: Number of instances per graph size.
        size_range: Different graph sizes.
        file_name: Path where results will be saved.
        include_exact_results: Whether to include exact results for computing beta.
            Only advisable for small problem sizes.

        problem_type: see parse_args in evaluate.py.
        timeout: see parse_args in evaluate.py.
        solver: see parse_args in evaluate.py.
        seed: see parse_args in evaluate.py.
        num_reads: see parse_args in evaluate.py.
        provider: see parse_args in evaluate.py.
        backend: see parse_args in evaluate.py.
        parallel_workers: Number of worker processes to use for parallel execution.

    Raises:
        FileExistsError: When the provided path already exists.
    """
    # Check if file already exists
    if os.path.exists(f"data{os.sep}" + file_name):
        raise FileExistsError(
            "Path already exists. Aborted as data otherwise might be overwritten!"
        )
    else:
        # Create data template
        all_data = {str(size): None for size in size_range}
        all_data["settings"] = {
            "_NB_INSTANCES_PER_SIZE": nb_instances_per_size,
            "_SIZE_RANGE": size_range,
            "FILE_NAME": file_name,
            "INCLUDE_EXACT_RESULTS": include_exact_results,
            "PROBLEM_TYPE": problem_type,
            "TIMEOUT": timeout,
            "SOLVER": solver,
            "SEED": seed,
            "NUM_READS": num_reads,
            "PROVIDER": provider,
            "BACKEND": backend,
            "PARALLEL_WORKERS": parallel_workers,
        }
    keep_graph = include_exact_results
    executor: Optional[ProcessPoolExecutor] = None
    if parallel_workers and parallel_workers > 1:
        executor = ProcessPoolExecutor(max_workers=parallel_workers)

    try:
        for size in size_range:
            result, times = [], []
            graphs = [] if keep_graph else None

            seeds_for_size = [seed + idx for idx in range(nb_instances_per_size)]
            task_args = [
                (
                    problem_type,
                    size,
                    solver,
                    timeout,
                    instance_seed,
                    num_reads,
                    provider,
                    backend,
                    keep_graph,
                )
                for instance_seed in seeds_for_size
            ]

            if executor:
                instance_results = list(executor.map(_run_single_instance, task_args))
            else:
                instance_results = [
                    _run_single_instance(task_arg) for task_arg in task_args
                ]

            for objective_result, elapsed_time, graph in instance_results:
                result.append(objective_result)
                times.append(elapsed_time)
                if graphs is not None:
                    graphs.append(graph)

            all_data[str(size)] = {"result": result, "times": times}

            if keep_graph and graphs is not None:
                if problem_type == "max-clique":
                    exact_results = [
                        nx.max_weight_clique(G, weight=None)[1] for G in graphs
                    ]
                elif problem_type == "max-cut":
                    exact_results = [one_exchange(G)[0] for G in graphs]
                all_data[str(size)]["exact-result"] = exact_results

            with open(f"data{os.sep}" + file_name, "w") as f:
                json.dump(all_data, f)

            seed = seeds_for_size[-1] + 1

            print(
                f"Finished problem size: {size}, "
                f"average objective: {np.array(result).mean()}, "
                f"average problem time: {np.array(times).mean():2f}."
            )
    finally:
        if executor:
            executor.shutdown(wait=True)


if __name__ == "__main__":
    # Input arguments
    _NB_INSTANCES_PER_SIZE = 100
    _SIZE_RANGE = list(range(2, 30, 1))
    FILE_NAME = "qaoa-sim.json"
    INCLUDE_EXACT_RESULTS = False
    PROBLEM_TYPE = "max-cut"
    TIMEOUT = 60
    SOLVER = "QAOA"
    _SEED = 101200
    NUM_READS = 1024
    PROVIDER = None
    BACKEND = None
    _PARALLEL_WORKERS = 8

    calculate_qscore(
        nb_instances_per_size=_NB_INSTANCES_PER_SIZE,
        size_range=_SIZE_RANGE,
        file_name=FILE_NAME,
        include_exact_results=INCLUDE_EXACT_RESULTS,
        problem_type=PROBLEM_TYPE,
        timeout=TIMEOUT,
        solver=SOLVER,
        seed=_SEED,
        num_reads=NUM_READS,
        provider=PROVIDER,
        backend=BACKEND,
        parallel_workers=_PARALLEL_WORKERS,
    )
