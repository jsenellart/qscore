"""Photonic ObliQ solvers (baseline VQC, Static, Hybrid)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler
from perceval.components.unitary_components import BS, PERM, PS
from scipy.optimize import minimize

from run.obliq_lib.obliq_utils import (
    ObliqResult,
    qubo_objective,
    solution_guesses,
    string_to_list,
)


@dataclass
class SimulationConfig:
    size: int
    nsamples: int = 1_024
    real_machine: bool = False
    backend: Optional[str] = None
    token: Optional[str] = None


def _qubo_dict_to_matrix(Q_dict, size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=float)
    for (i, j), value in Q_dict.items():
        if i == j:
            matrix[i, i] += value
        else:
            coeff = value / 2.0
            matrix[i, j] += coeff
            matrix[j, i] += coeff
    return matrix


def _basic_state_from_ones(size: int, ones: Iterable[int]) -> pcvl.BasicState:
    occupation = [0] * size
    for idx in ones:
        occupation[idx] = 1
    return pcvl.BasicState(occupation)


def _simulate_distribution(circuit: pcvl.Circuit, ones: List[int], config: SimulationConfig) -> Dict[str, float]:
    input_state = _basic_state_from_ones(config.size, ones)
    processor: pcvl.Processor | pcvl.RemoteProcessor
    if config.real_machine or config.backend:
        backend = config.backend or "qpu:ascella"
        processor = pcvl.RemoteProcessor(backend, config.token)
        processor.set_circuit(circuit)
    else:
        processor = pcvl.Processor("SLOS", circuit)

    processor.with_input(input_state)
    processor.min_detected_photons_filter(1)

    if config.real_machine or config.backend:
        sampler = Sampler(processor, max_shots_per_call=config.nsamples)
        sampler.default_job_name = "ObliQ sampling run"
        raw_counts = sampler.sample_count(config.nsamples)["results"]
        total = sum(raw_counts.values()) or 1
        return {str(state): count / total for state, count in raw_counts.items()}

    sampler = Sampler(processor)
    return {str(state): prob for state, prob in sampler.probs(config.nsamples)["results"].items()}


def _normalize_qubo(Q: np.ndarray) -> np.ndarray:
    Q_min = np.min(Q)
    Q_max = np.max(Q)
    if np.isclose(Q_max, Q_min):
        return np.zeros_like(Q)
    return (Q - Q_min) / (Q_max - Q_min)


def _add_anchor_layers(circ: pcvl.Circuit, Q_norm: np.ndarray, num_rep: int) -> None:
    size = Q_norm.shape[0]
    pair_list = list({(i, j) for i in range(size) for j in range(i + 1, size)})
    for _ in range(num_rep):
        for i, j in pair_list:
            weight = Q_norm[i, j] / (num_rep**2)
            if weight <= 0:
                continue
            weight = min(weight, 1 - 1e-9)
            theta = np.arccos(np.sqrt(1 - weight)) * 0.5
            if j == i + 1:
                circ.add((i, i + 1), BS(theta, 0, 0, 0, 0))
            else:
                perm = list(range(size))
                perm[i + 1], perm[j] = perm[j], perm[i + 1]
                circ.add(tuple(range(size)), PERM(perm))
                circ.add((i, i + 1), BS(theta, 0, 0, 0, 0))
                circ.add(tuple(range(size)), PERM(perm))


def _expected_coeff_count(size: int) -> int:
    return max(2, 8 * size - 6)


def _prepare_coeff_vector(coeffs: Optional[Sequence[float]], size: int) -> np.ndarray:
    expected = _expected_coeff_count(size)
    if coeffs is None:
        return np.zeros(expected, dtype=float)
    coeff_array = np.asarray(coeffs, dtype=float)
    if coeff_array.size < expected:
        raise ValueError(
            f"Expected at least {expected} coefficients, received {coeff_array.size}"
        )
    return coeff_array


def _add_vqc_layers(circ: pcvl.Circuit, size: int, coeffs: Sequence[float]) -> None:
    k = 0
    for _ in range(2):
        for i in range(size):
            circ.add(i, PS(coeffs[k]))
            k += 1
        for i in range(size - 1):
            circ.add((i, i + 1), BS(coeffs[k]))
            k += 1
        for i in range(size):
            circ.add(i, PS(coeffs[k]))
            k += 1
        for i in reversed(range(1, size - 1)):
            circ.add((i - 1, i), BS(coeffs[k]))
            k += 1


def _estimate_static_bitstring(
    Q: np.ndarray,
    config: SimulationConfig,
    num_rep: int,
    graph_mode: int,
) -> List[int]:
    """Run the static (anchor-only) circuit to obtain a seed bitstring."""

    circuit = pcvl.Circuit(config.size)
    Q_norm = _normalize_qubo(Q)
    _add_anchor_layers(circuit, Q_norm, num_rep)
    distribution = _simulate_distribution(circuit, list(range(config.size)), config)
    result = _distribution_to_result(distribution, Q, graph_mode)
    return result.bitstring


def _distribution_to_result(
    distribution: Dict[str, float], Q: np.ndarray, graph_mode: int
) -> ObliqResult:
    size = Q.shape[0]
    avg = np.zeros(size)
    best_prob = -1.0
    best_vec = np.zeros(size)

    for state_str, prob in distribution.items():
        numeric = np.array(string_to_list(state_str))
        vector = np.where(numeric >= 1, 1, 0)
        if vector.sum() < 1:
            continue
        avg += vector * prob
        if prob > best_prob:
            best_prob = prob
            best_vec = vector

    candidates = solution_guesses(avg, graph_mode)
    best_value = float("inf")
    best_state = best_vec.copy()

    def _evaluate(indices: List[int]) -> None:
        nonlocal best_value, best_state
        state = np.zeros(size)
        state[indices] = 1
        value = qubo_objective(Q, state)
        if value < best_value or (np.isclose(value, best_value) and state.sum() > best_state.sum()):
            best_value = value
            best_state = state

    for indices in candidates:
        _evaluate(indices)

    candidate_idx = np.where(best_vec >= 1)[0].tolist()
    if candidate_idx:
        _evaluate(candidate_idx)

    return ObliqResult(
        objective=-best_value,
        bitstring=best_state.astype(int).tolist(),
        metadata={
            "graph_mode": graph_mode,
            "candidate_count": len(candidates),
            "energy": best_value,
        },
    )


def run_obliq_solver(
    Q: np.ndarray,
    variant: str = "obliq-static",
    nsamples: int = 1_024,
    num_rep: int = 10,
    graph_mode: int = 0,
    coeffs: Optional[Sequence[float]] = None,
    real_machine: bool = False,
    backend: Optional[str] = None,
    token: Optional[str] = None,
    seed_bitstring: Optional[Sequence[int]] = None,
) -> ObliqResult:
    """Run one of the ObliQ solver variants on the given QUBO matrix."""

    Q = np.array(Q, dtype=float)
    scale = np.max(np.abs(Q)) or 1.0
    Q = Q / scale
    size = Q.shape[0]
    config = SimulationConfig(
        size=size,
        nsamples=nsamples,
        real_machine=real_machine,
        backend=backend,
        token=token,
    )
    variant = variant.lower()
    vqc_variants = {
        "baseline-vqc",
        "vqc",
        "baseline",
        "obliq",
        "hybrid",
        "obliq-hybrid",
    }
    seeded_bitstring: Optional[List[int]] = None
    if seed_bitstring is not None and variant in vqc_variants:
        seeded_bitstring = [int(x) for x in seed_bitstring]

    circuit = pcvl.Circuit(size)

    if variant in {"obliq-static", "static", "anchor"}:
        Q_norm = _normalize_qubo(Q)
        _add_anchor_layers(circuit, Q_norm, num_rep)
    elif variant in {"baseline-vqc", "vqc", "baseline"}:
        coeff_array = _prepare_coeff_vector(coeffs, size)
        _add_vqc_layers(circuit, size, coeff_array)
    elif variant in {"obliq", "hybrid", "obliq-hybrid"}:
        Q_norm = _normalize_qubo(Q)
        _add_anchor_layers(circuit, Q_norm, num_rep)
        coeff_array = _prepare_coeff_vector(coeffs, size)
        _add_vqc_layers(circuit, size, coeff_array)
    else:
        raise ValueError(
            "Unknown ObliQ variant. Use 'baseline-vqc', 'obliq-static', or 'obliq-hybrid'."
        )

    if variant in {"obliq", "hybrid", "obliq-hybrid"} and seeded_bitstring is None:
        seeded_bitstring = _estimate_static_bitstring(Q, config, num_rep, graph_mode)

    ones = list(range(size))
    if seeded_bitstring is not None and variant in vqc_variants:
        ones = [idx for idx, bit in enumerate(seeded_bitstring) if bit]
        if not ones:
            ones = list(range(size))

    distribution = _simulate_distribution(circuit, ones, config)
    result = _distribution_to_result(distribution, Q, graph_mode)
    result.objective *= scale
    if "energy" in result.metadata:
        result.metadata["normalized_energy"] = result.metadata["energy"]
        result.metadata["energy"] = result.metadata["energy"] * scale
    result.metadata.update(
        {
            "variant": variant,
            "nsamples": nsamples,
            "num_rep": num_rep,
            "coeffs_provided": coeffs is not None,
        }
    )
    if seeded_bitstring is not None and variant in vqc_variants:
        result.metadata["input_bitstring"] = seeded_bitstring
    return result


def _finite_difference_gradient(
    coeffs: np.ndarray,
    index: int,
    step: float,
    evaluate_fn: Callable[[np.ndarray], float],
) -> float:
    plus = coeffs.copy()
    minus = coeffs.copy()
    plus[index] += step
    minus[index] -= step
    loss_plus = evaluate_fn(plus)
    loss_minus = evaluate_fn(minus)
    return (loss_plus - loss_minus) / (2 * step)


def train_obliq_vqc_coeffs(
    Q: np.ndarray,
    variant: str = "baseline-vqc",
    initial_coeffs: Optional[Sequence[float]] = None,
    nsamples: int = 5_000,
    num_rep: int = 10,
    graph_mode: int = 0,
    real_machine: bool = False,
    backend: Optional[str] = None,
    token: Optional[str] = None,
    max_iter: int = 5,
    learning_rate: float = 0.05,
    finite_diff_step: float = np.pi / 4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    verbose: bool = False,
    optimizer: str = "adam",
) -> tuple[list[float], dict]:
    """Optimize VQC coefficients via Adam (default) or gradient-free COBYLA."""

    if variant not in {"baseline-vqc", "obliq-hybrid"}:
        raise ValueError("Coefficient training is only supported for VQC-enabled variants.")
    if real_machine:
        raise ValueError("VQC coefficient training requires a simulator backend.")

    coeffs = _prepare_coeff_vector(initial_coeffs, Q.shape[0]).astype(float)
    m = np.zeros_like(coeffs)
    v = np.zeros_like(coeffs)
    t = 0
    history: list[float] = []
    seeded_bitstring: Optional[List[int]] = None
    if variant == "obliq-hybrid":
        seed_config = SimulationConfig(
            size=Q.shape[0],
            nsamples=nsamples,
            real_machine=real_machine,
            backend=backend,
            token=token,
        )
        seeded_bitstring = _estimate_static_bitstring(Q, seed_config, num_rep, graph_mode)

    optimizer = optimizer.lower()
    if optimizer not in {"adam", "cobyla"}:
        raise ValueError("Optimizer must be either 'adam' or 'cobyla'.")

    def _run_solver_for_coeffs(current_coeffs: np.ndarray) -> float:
        result = run_obliq_solver(
            Q=Q,
            variant=variant,
            nsamples=nsamples,
            num_rep=num_rep,
            graph_mode=graph_mode,
            coeffs=current_coeffs.tolist(),
            real_machine=real_machine,
            backend=backend,
            token=token,
            seed_bitstring=seeded_bitstring,
        )
        return float(result.metadata["energy"])

    def evaluate(current_coeffs: np.ndarray, record: bool = False) -> float:
        energy = _run_solver_for_coeffs(np.asarray(current_coeffs, dtype=float))
        if record:
            history.append(float(energy))
        return energy

    if optimizer == "cobyla":
        def objective(vec: np.ndarray) -> float:
            return evaluate(vec, record=True)

        if max_iter is None:
            maxiter_budget = max(coeffs.size + 2, coeffs.size * 20)
        else:
            maxiter_budget = max(coeffs.size + 2, max_iter)
        options = {"maxiter": int(maxiter_budget)}
        result = minimize(
            objective,
            coeffs,
            method="COBYLA",
            options=options,
        )
        coeffs = np.asarray(result.x, dtype=float)
        metadata: dict = {
            "energies": history,
            "optimizer": "cobyla",
            "success": bool(result.success),
            "status": int(result.status),
        }
        if result.message:
            metadata["message"] = result.message
        return coeffs.tolist(), metadata

    loss_fn = lambda vec: evaluate(vec, record=False)
    current_loss = evaluate(coeffs, record=True)

    for iteration in range(1, max_iter + 1):
        grads = np.zeros_like(coeffs)
        for idx in range(coeffs.size):
            grads[idx] = _finite_difference_gradient(
                coeffs,
                idx,
                finite_diff_step,
                loss_fn,
            )

        t += 1
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        coeffs -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        current_loss = evaluate(coeffs, record=True)
        if verbose:
            print(f"[ObliQ train] iteration {iteration}: energy={current_loss:.4f}")

    return coeffs.tolist(), {"energies": history, "optimizer": "adam"}
