"""Photonic CVaR-VQE QUBO solver wrapper based on Quandela's reference implementation."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Union

import networkx as nx
import numpy as np
import perceval as pcvl
from networkx import Graph
from perceval.algorithm import Sampler
from perceval.components import GenericInterferometer
from perceval.components.unitary_components import BS
from scipy.optimize import minimize

def _parify_samples_threshold(samples: Dict[pcvl.BasicState, int], j: int):
    """Apply the parity function described in the reference implementation."""

    new_samples: Dict[pcvl.BasicState, int] = {}
    for sample, count in samples.items():
        parified_sample = [(int(i == 0) + j) % 2 for i in sample]
        basic_state = pcvl.BasicState(parified_sample)
        new_samples[basic_state] = new_samples.get(basic_state, 0) + count
    return new_samples


def _compute_cvar(probabilities: Iterable[float], values: Iterable[float], alpha: float):
    """Compute Conditional Value at Risk for the sampled values."""

    sorted_indices = np.argsort(values)
    probs = np.array(list(probabilities))[sorted_indices]
    vals = np.array(list(values))[sorted_indices]
    cvar = 0.0
    total_prob = 0.0
    for prob, val in zip(probs, vals):
        if prob >= alpha - total_prob:
            prob = alpha - total_prob
        total_prob += prob
        cvar += prob * val
        if total_prob >= alpha:
            break
    cvar /= max(total_prob, np.finfo(float).eps)
    return cvar


def _set_parameters_circuit(parameters_circuit, values: List[float]):
    """Assign numerical values to the circuit parameters."""

    for idx, parameter in enumerate(parameters_circuit):
        parameter.set_value(values[idx])


def _compute_samples(
    circuit: pcvl.Circuit,
    input_state: pcvl.BasicState,
    nb_samples: int,
    j: int,
    run_on_qpu: bool,
    run_on_gpu: bool,
    platform: str,
):
    """Draw samples from the photonic circuit using the requested backend."""

    processor = pcvl.Processor("SLOS", circuit)
    sampler = Sampler(processor)
    if run_on_qpu:
        processor = pcvl.RemoteProcessor(platform)
        processor.set_circuit(circuit)
        sampler = Sampler(processor)
    if run_on_gpu:
        processor = pcvl.RemoteProcessor(platform)
        processor.set_circuit(circuit)
        sampler = Sampler(processor, max_shots_per_call=nb_samples)
    processor.with_input(input_state)
    processor.min_detected_photons_filter(input_state.n)
    samples = sampler.sample_count(nb_samples)["results"]
    return _parify_samples_threshold(samples, j)


def _graph_to_matrix_maxcut(graph: Graph) -> np.ndarray:
    """Map a Max-Cut instance to the QUBO matrix form used by the solver."""

    matrix = nx.to_numpy_array(graph)
    np.fill_diagonal(matrix, [-graph.degree[node] for node in graph.nodes()])
    return matrix


def _graph_to_matrix_max_clique(graph: Graph) -> np.ndarray:
    """Map a Max-Clique instance to the QUBO matrix form used by the solver."""

    matrix = np.zeros((len(graph), len(graph)))
    np.fill_diagonal(matrix, -1)
    complement = nx.complement(graph)
    for u, v in complement.edges():
        matrix[u, v] = 2
        matrix[v, u] = 2
    return matrix


def _device_setup(nb_modes: int, nb_inputs: int):
    """Configure the photonic processor inputs and interferometer."""

    inputs: List[pcvl.BasicState] = []
    for k in range(1, nb_inputs + 1):
        interval = 2**k
        pattern = [int((i + interval / 2 + 1) % interval == 0) for i in range(nb_modes)]
        inputs.append(pcvl.BasicState(pattern))

    circuit = GenericInterferometer(nb_modes, lambda idx: BS(theta=pcvl.P(f"theta{idx}")))
    return circuit, inputs


def _expectation_value(vec_state: np.ndarray, matrix: np.ndarray, offset: float):
    return float(np.dot(vec_state.conjugate(), np.dot(matrix, vec_state)) + offset)


def _sorting_samples(
    configuration_samples: Dict[pcvl.BasicState, int],
    H: np.ndarray,
    E_max: float,
    start_time: float,
    offset: float,
    goal: str,
    nb_iterations: int,
):
    configuration_values = {
        key: _expectation_value(np.array(list(key)), H, offset)
        for key in configuration_samples
    }
    best_state = min(configuration_values, key=configuration_values.get)
    best_value = configuration_values[best_state]
    if goal == "max":
        best_value = -best_value
        E_max = -E_max
    return {
        "average_value": E_max,
        "best_state": best_state,
        "best_value": best_value,
        "time": time.time() - start_time,
        "optimised_circuit_output": configuration_samples,
        "nb_iterations": nb_iterations,
    }


def _run_configuration(
    circuit: pcvl.Circuit,
    j: int,
    input_state: pcvl.BasicState,
    H: np.ndarray,
    nb_samples: int,
    run_on_qpu: bool,
    run_on_gpu: bool,
    platform: str,
    offset: float,
    max_iter: int,
    cvar_alpha: float,
):
    """Optimize circuit parameters for a given configuration and sample outcome."""

    iterations = {"count": 0}

    def loss(parameters):
        _set_parameters_circuit(parameters_circuit, parameters)
        samples = _compute_samples(
            circuit, input_state, nb_samples, j, run_on_qpu, run_on_gpu, platform
        )
        probabilities = [value / nb_samples for value in samples.values()]
        values = [_expectation_value(np.array(list(sample)), H, offset) for sample in samples]
        exp_value = _compute_cvar(probabilities, values, cvar_alpha)
        iterations["count"] += 1
        return exp_value

    parameters_circuit = circuit.get_parameters()
    init_parameters = [np.pi for _ in parameters_circuit]
    min_required_iterations = len(parameters_circuit) + 2
    effective_maxiter = max(max_iter, min_required_iterations)
    best_parameters = minimize(
        loss,
        init_parameters,
        method="COBYLA",
        options={"maxiter": effective_maxiter},
        bounds=[(0, 2 * np.pi) for _ in init_parameters],
    ).x
    _set_parameters_circuit(parameters_circuit, best_parameters)
    samples = _compute_samples(
        circuit, input_state, nb_samples, j, run_on_qpu, run_on_gpu, platform
    )

    return samples, iterations["count"]


def qubo_solver(
    graph: Graph,
    problem_type: str,
    nb_samples: int = 2048,
    nb_inputs: int = 1,
    run_on_qpu: bool = False,
    run_on_gpu: bool = False,
    platform: Optional[str] = None,
    offset: float = 0.0,
    goal: str = "min",
    max_iter: int = 5,
    cvar_alpha: float = 1.0,
):
    """Run the CVaR-VQE optimizer for the requested QUBO instance."""

    start_time = time.time()
    H = (
        _graph_to_matrix_maxcut(graph)
        if problem_type == "max-cut"
        else _graph_to_matrix_max_clique(graph)
    )
    nb_modes = len(H)
    circuit, inputs = _device_setup(nb_modes, nb_inputs)

    js = [0, 1]
    E_max = float("inf")
    configuration_samples: Optional[Dict[pcvl.BasicState, int]] = None
    total_iterations = 0

    for parity in js:
        for input_state in inputs:
            current_sample, iteration_count = _run_configuration(
                circuit,
                parity,
                input_state,
                H,
                nb_samples,
                run_on_qpu,
                run_on_gpu,
                platform,
                offset,
                max_iter,
                cvar_alpha,
            )
            total_iterations += iteration_count
            energy = 0.0
            sum_count = 0
            for sample, count in current_sample.items():
                vec = np.array(list(sample))
                obj = _expectation_value(vec, H, offset)
                energy += obj * count
                sum_count += count

            average_energy = energy / max(sum_count, 1)
            if average_energy < E_max:
                E_max = average_energy
                configuration_samples = current_sample

    if configuration_samples is None:
        return {
            "average_value": float("inf"),
            "best_state": pcvl.BasicState([0] * nb_modes),
            "best_value": float("inf"),
            "time": time.time() - start_time,
            "optimised_circuit_output": {},
            "nb_iterations": total_iterations,
        }

    return _sorting_samples(
        configuration_samples,
        H,
        E_max,
        start_time,
        offset,
        goal,
        total_iterations,
    )


def _bitstring_from_basic_state(state: pcvl.BasicState) -> np.ndarray:
    return np.array(list(state)).astype(int)



def run_photonic_cvarvqe(
    graph: Graph,
    problem_type: str,
    **solver_kwargs,
) -> List[int]:
    """Solve the given instance with the photonic CVaR-VQE solver."""

    result = qubo_solver(graph, problem_type, **solver_kwargs)
    bitstring = _bitstring_from_basic_state(result["best_state"])
    return bitstring.astype(int).tolist()
