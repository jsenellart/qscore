"""Run a Q-score instance using a custom QAOA implementation."""
import os
import time
from multiprocessing import AuthenticationError
from typing import Optional

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.primitives import BackendSamplerV2, StatevectorSampler
from qiskit.providers.backend import Backend
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    OptimizationResultStatus,
)
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems import QuadraticProgram

HUB = "ibm-q"
GROUP = "open"
PROJECT = "main"


def initialize_backend(
    provider: Optional[str],
    backend: Optional[str],
) -> Optional[Backend]:
    """Initialize optional backend for IBM or QI providers.

    Args:
        provider: Name of hardware provider.
        backend: Name of specific hardware.

    Returns:
        Backend instance for hardware providers; ``None`` for local sampling.

    Raises:
        AuthenticationError: if authentication to ibm or qi failed.
        ValueError: if provider or backend is not among supported options.
    """

    if provider is None or provider.lower() in {
        "local simulator",
        "qasm simulator",
        "statevector simulator",
    }:
        return None

    provider_lower = provider.lower()

    if provider_lower == "ibm":
        if backend is None:
            raise ValueError("Backend name must be provided when using IBM provider.")
        try:
            runtime_service = QiskitRuntimeService(instance=f"{HUB}/{GROUP}/{PROJECT}")
        except Exception as exc:
            raise AuthenticationError("Authentication with IBM failed.") from exc
        try:
            return runtime_service.backend(backend)
        except QiskitBackendNotFoundError as exc:
            raise ValueError(f"Backend {backend} not among possible options.") from exc

    elif provider_lower == "qi":
        if backend is None:
            raise ValueError("Backend name must be provided when using QI provider.")
        try:
            from quantuminspire import credentials as qicredentials
            from quantuminspire.qiskit import QI

            token = qicredentials.load_account()
            qi_authentication = qicredentials.get_token_authentication(token)
            QI_URL = os.getenv("API_URL", "https://api.quantum-inspire.com/")
            project_name = f"TNO Q-score {int(time.time())}"

            QI.set_authentication(qi_authentication, QI_URL, project_name=project_name)
            return QI.get_backend(backend)
        except Exception as exc:
            raise AuthenticationError("Authentication with QI failed.") from exc

    raise ValueError(
        f"Provider {provider} and backend {backend} not among possible options."
    )


def _extract_ising_terms(
    operator: SparsePauliOp,
) -> tuple[list[tuple[int, float]], list[tuple[int, int, float]]]:
    """Split an Ising operator into single- and two-qubit Z terms."""

    single_z_terms: list[tuple[int, float]] = []
    zz_terms: list[tuple[int, int, float]] = []

    for coeff, pauli in zip(operator.coeffs, operator.paulis):
        if abs(coeff.imag) > 1e-8:
            raise ValueError("Found complex coefficient in Ising operator.")

        label = pauli.to_label()
        qubits = [len(label) - 1 - idx for idx, char in enumerate(label) if char == "Z"]

        if not qubits:
            continue  # constant term

        value = float(coeff.real)
        if len(qubits) == 1:
            single_z_terms.append((qubits[0], value))
        elif len(qubits) == 2:
            zz_terms.append((qubits[0], qubits[1], value))
        else:
            raise ValueError("Higher-order interactions are not supported by this QAOA ansatz.")

    return single_z_terms, zz_terms


def _build_qaoa_ansatz(
    num_qubits: int,
    single_z_terms: list[tuple[int, float]],
    zz_terms: list[tuple[int, int, float]],
    reps: int = 1,
) -> QuantumCircuit:
    """Construct a parameterized QAOA circuit without deprecated n-local helpers."""

    beta_params = ParameterVector("beta", reps)
    gamma_params = ParameterVector("gamma", reps)
    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))

    for layer in range(reps):
        gamma = gamma_params[layer]
        for qubit, coeff in single_z_terms:
            if coeff != 0:
                circuit.rz(2 * gamma * coeff, qubit)
        for qubit_i, qubit_j, coeff in zz_terms:
            if coeff != 0:
                circuit.rzz(2 * gamma * coeff, qubit_i, qubit_j)
        beta = beta_params[layer]
        for qubit in range(num_qubits):
            circuit.rx(2 * beta, qubit)

    return circuit


def run_QAOA(
    qp: QuadraticProgram,
    provider: Optional[str] = None,
    backend: Optional[str] = None,
    number_of_shots: Optional[int] = None,
    max_attempts: Optional[int] = 10,
) -> float:
    """
    Function that solves a Q-score instance using QAOA.

    Args:
        qp: Quadratic Problem of Q-score instance.
        backend: Name of the quantum instance backend.
        number_of_shots: Number of shots for hardware.

    Returns:
        The found objective value.

    Raises:
        ValueError: if no feasible solution was found in max_attempts attempts.
    """
    if number_of_shots is None:
        number_of_shots = 1024

    backend_instance = initialize_backend(provider, backend)
    if backend_instance is None:
        sampler = StatevectorSampler()
    else:
        sampler = BackendSamplerV2(
            backend=backend_instance,
            options={"default_shots": number_of_shots},
        )

    converter = QuadraticProgramToQubo()
    qubo_problem = converter.convert(qp)
    operator, _ = qubo_problem.to_ising()
    single_z_terms, zz_terms = _extract_ising_terms(operator)
    ansatz = _build_qaoa_ansatz(operator.num_qubits, single_z_terms, zz_terms, reps=1)
    optimizer = COBYLA(maxiter=100)
    qaoa_mes = SamplingVQE(ansatz=ansatz, sampler=sampler, optimizer=optimizer)

    qaoa = MinimumEigenOptimizer(qaoa_mes)
    for _ in range(max_attempts):
        qaoa_result = qaoa.solve(qubo_problem)
        if qaoa_result.status == OptimizationResultStatus.SUCCESS:
            original_solution = converter.interpret(qaoa_result.x)
            return qp.objective.evaluate(original_solution)
    raise ValueError("Could not find feasible solution")


if __name__ == "__main__":
    b = initialize_backend("qi", "QX single-node simulator")
