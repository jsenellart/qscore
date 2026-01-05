import time
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

# import relevant libraries
from .helpers.circuit_builder import AnsatzFactory
from .helpers.rotations import apply_all_phase_shifters, params
from .helpers.rotations import *  # noqa: F401,F403 - legacy helper exports
from .helpers.qwc import *  # noqa: F401,F403 - legacy helper exports
from .helpers.qubit2fock import *  # noqa: F401,F403 - legacy helper exports
from .helpers.qubo2ising import *  # noqa: F401,F403 - legacy helper exports
from perceval.algorithm import Sampler
from perceval.components import AProcessor


class GenericVqe:
    """
    A class for executing a Variational Quantum Eigensolver (VQE) on a given Hamiltonian.

    The GenericVqe class offers functionalities to optimize quantum circuits to approximate the ground state of
    a specified Hamiltonian. It provides a variety of options, including different types of entanglement,
    optimization methods, and quantum gate sets.

    Attributes:
        hnames (list of str, optional): Names or Pauli strings of Hamiltonian terms. For example, ['XIZ', 'IZX'].
            Default is None.
        hweights (list of float, optional): Coefficients or weights for each Hamiltonian term, aligning with `hnames`.
            Default is None.
        rots (list of str, optional): Specifies the types of single-qubit rotations to be used in the quantum circuit.
            Each rotation type should be a string representing a Rx, Ry or Rz gate ('X', 'Y', or 'Z'). Default is ['Y'].
        input_state (str, optional): Initial state for the quantum system, given as a binary string. Default is None.
        entanglement_type (str, optional): Type of entanglement pattern for multi-qubit gates. Default is "maximal".
        method (str, optional): Optimization algorithm to use. Default is "COBYLA".
        shots (int or float, optional): Number of measurements for each quantum circuit evaluation. Default is 10e5.
        ctype (str, optional): Specifies the type of controlled gate used in the quantum circuit. Default is "cx".
        offset (float, optional): A constant offset to the energy. Default is 0.
    """

    def __init__(
        self,
        processor: AProcessor,
        hnames: list = None,
        hweights: list = None,
        rots: list = ["Y"],
        input_state: str = None,
        entanglement_type: str = "linear",
        method: str = "COBYLA",
        shots: int = 1000000,
        ctype: str = None,
        offset: float = 0.0,
        progress_cb: Callable = None,
        maxiter=200,
        geometry=None,
        qubo_matrix=None,
        exact=None,
        alpha=None,
    ):

        self.progress_cb = progress_cb
        self._processor = processor
        self.apply_reverse = False
        # self.offset = offset
        self.maxiter = maxiter
        self.geometry = geometry
        self.offset = offset
        self.qubo_matrix = qubo_matrix
        self._qubo_matrix_np = (
            np.array(qubo_matrix, dtype=float) if qubo_matrix is not None else None
        )
        self.exact = exact
        if hnames:
            hnames = [s[::-1] for s in hnames]

        if qubo_matrix:
            hnames, hweights = qubo_to_ising_ham(self._qubo_matrix_np)
            offset2 = qubo_to_ising_with_offset(self._qubo_matrix_np)
            ctype = "cx"
            self.offset = offset2
            self.entanglement_type = "linear"

        # Control type
        if ctype not in ["cx", "cz"]:
            raise ValueError(
                f"Invalid ctype '{ctype}'. It should be either 'cx' or 'cz'."
            )
        self.ctype = ctype

        # Extract the identity term

        identity_str = "I" * int(len(hnames[0]))
        if identity_str in hnames:
            idx = hnames.index(identity_str)
            self.identity = hweights[idx]
            hnames.pop(idx)
            hweights.pop(idx)
        else:
            self.identity = 0

        # Extract necessary information
        # Check if hnames and hweights are provided either from direct input or file
        if len(hnames) == 0 or len(hweights) == 0:
            # raise ValueError(f"{hnames}, f{hweights}")
            raise ValueError(
                "Hamiltonian terms and values must be provided either directly or from a file."
            )
        self.hnames = list(hnames)
        self.hweights = list(hweights)
        self.num_qubits = int(len(hnames[0]))

        # Validate and set input state
        if input_state:
            if len(input_state) != self.num_qubits or not all(
                [bit in ["0", "1"] for bit in input_state]
            ):
                raise ValueError(
                    "Invalid input state format. It should be a binary string of length equal to the number of qubits."
                )
            self.input_state = input_state
        else:
            # Default to all zeros if no input state is provided
            self.input_state = "0" * self.num_qubits

        self.qubit_num = self.num_qubits
        modes_available = 128
        self.group_size_array, self.total_modes = OptimalGroups(
            modes_available, self.qubit_num
        )
        self.group_size_array = [int(x) for x in self.group_size_array]

        qwc = QWCGrouping("lf", self.hnames, self.hweights)
        self.groups = list(qwc.create_qwc_groups().values())
        self.measurement_bases = [
            extract_measurement_bases(group_data) for group_data in self.groups
        ]

        modes_available = 128

        group_size_array, total_modes = OptimalGroups(modes_available, self.qubit_num)
        group_size_array = [int(x) for x in group_size_array]
        if self.qubo_matrix:
            self.measurement_bases = ["Z" * self.num_qubits]
            self.params = params(self.measurement_bases, group_size_array)
        else:
            self.params = params(self.measurement_bases, group_size_array)

        self.rots = rots

        self.entanglement_type = entanglement_type
        self.method = method
        self.shots = shots
        self.alpha = alpha

        # Results and iteration data
        self.loss_values = []
        self.avg = 0
        self.param_iterations = []
        self.iteration = 0

    def minimize_loss(self, lp):
        """
        Minimizes the loss value for a given set of parameters and groupings of Hamiltonian terms.

        Args:
            lp (list of float): Parameters for the variational circuit.

        Returns:
            float: The computed loss value.
        """

        params = self.params
        modes_available = 128

        group_size_array, total_modes = OptimalGroups(modes_available, self.qubit_num)
        #  print(total_modes)
        group_size_array = [int(x) for x in group_size_array]
        for idx, v in enumerate(lp):
            while lp[idx] < 0:
                lp[idx] += 2 * np.pi
            while lp[idx] >= 2 * np.pi:
                lp[idx] -= 2 * np.pi

        factory = AnsatzFactory()

        circ = factory.build_qubit_circuit(
            group_size_array, lp, self.rots, self.entanglement_type, self.ctype
        )
        new_circ = circ.add(0, apply_all_phase_shifters(group_size_array))
        processor = self._processor

        processor.set_circuit(new_circ)
        input_state = to_fock_state(self.input_state, self.group_size_array)
        processor.with_input(input_state)
        processor.min_detected_photons_filter(input_state.n)
        sampler = Sampler(
            processor, max_shots_per_call=self.shots * len(self.measurement_bases)
        )

        sampler.add_iteration_list(params)

        job = sampler.sample_count
        job.name = f"vqe:avg[{self.iteration}]"
        job_results = job.execute_sync(self.shots)

        for idx, v in enumerate(lp):
            while lp[idx] < 0:
                lp[idx] += 2 * np.pi
            while lp[idx] >= 2 * np.pi:
                lp[idx] -= 2 * np.pi

        averages = []

        for i, res in enumerate(job_results["results_list"]):
            # print(res)
            self.avg = 0

            # Sample
            output_dict = {}

            sum_valid_outputs = 0

            for res1 in res["results"]:
                qb_state = fock_to_qubit_state(res1, group_size_array)
                if qb_state:
                    sum_valid_outputs += res["results"][res1]

            for res1 in res["results"]:
                qb_state1 = fock_to_qubit_state(res1, group_size_array)
                if qb_state1:
                    output_dict[qb_state1] = res["results"][res1] / sum_valid_outputs

            # print(res)

            if self.qubo_matrix:
                # print("cvar")
                if self.qubit_num > 10:
                    alpha = 0.15
                if self.qubit_num > 5:
                    alpha = 0.25
                else:
                    alpha = 0.4

                if self.alpha:
                    alpha = self.alpha
                # alpha = 0.25
                values_bitstring = {}
                # val = 0
                for output_state in output_dict.keys():

                    val = 0
                    for j, term in self.groups[0]:
                        state = 0
                        for measure in rotate_measurements_2(self.qubit_num, term):
                            state += int(output_state[measure])
                        if state % 2 == 1:
                            val -= output_dict[output_state] * float(j)
                        else:
                            val += output_dict[output_state] * float(j)
                    values_bitstring[output_state] = val

            if self.qubo_matrix:
                probs_uns = np.array(list(output_dict.values()))
                vals = np.array(list(values_bitstring.values()))
                vals_test = np.divide(
                    vals, probs_uns, out=np.zeros_like(vals), where=probs_uns != 0
                )
                sorted_indices = np.argsort(vals_test)
                vals_final = np.array(vals_test)[sorted_indices]
                # Re-arrange the probabilities of each bitstring according to the sorting.
                probs = np.array(probs_uns)[sorted_indices]

                # Compute the CVaR value.
                cvar = 0
                total_lim = 0
                for b, (p, v) in enumerate(zip(probs, vals_final)):
                    done = False
                    if p >= alpha - total_lim:
                        p = alpha - total_lim
                        done = True
                    total_lim += p
                    cvar += p * v
                if total_lim > 0:
                    cvar /= total_lim
                else:
                    cvar = 0
            # cvar += self.offset

            else:
                self.output_dict = output_dict

                for j, term in self.groups[i]:
                    # print(j)
                    # print(term)
                    # for j, term in self.groups:

                    for output_state in output_dict.keys():

                        state = 0
                        for measure in rotate_measurements_2(self.num_qubits, term):
                            state += int(output_state[measure])

                        if state % 2 == 1:
                            self.avg -= output_dict[output_state] * float(j)
                        else:
                            self.avg += output_dict[output_state] * float(j)
                averages.append(self.avg)

        if self.qubo_matrix:
            # print(self.offset)

            loss = cvar + self.offset
            self.loss_values.append(loss)
            self.iteration += 1
            self.output_dict = output_dict
            # maximum = max(output_dict, key=output_dict.get)  # Just use 'min' instead of 'max' for minimum.
            # print(maximum, output_dict[maximum])
            output_dict = {}
            # print(output_dict)
            # print({f"{self.iteration}": self.loss_values[-1]})
            my_dict = output_dict
            # bitstring = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)[:4]
            # bitstring = max(self.output_dict, key=self.output_dict.get)
            best_bitstring = self._select_best_bitstring()
            if best_bitstring is None:
                self.bitstring = None
                cvar = 0.0  # or a neutral fallback
            else:
                self.bitstring = best_bitstring
            intermediate = float(cvar)
            # print(bitstring)
            if self.progress_cb:
                # req = self.progress_cb(self.iteration / self.maxiter, f" iteration {self.iteration}/{self.maxiter}",
                #                      {"loss": intermediate})
                req = self.progress_cb(bitstring, loss)

                if req.get("cancel_requested"):
                    raise RuntimeError("Cancel requested")
        else:
            # averages.append(self.avg)
            loss = sum(averages) + self.offset + self.identity

            self.loss_values.append(loss)

            self.output_dict = output_dict

            self.iteration += 1
            """if self.exact:
                print(
                    "Iteration " + str(self.iteration) + ": Loss " + str(loss) + "    diff :" + str(self.exact - loss))
            else:
                print("Iteration " + str(self.iteration) + ": Loss " + str(loss))"""

            self.param_iterations.append(lp)

            # print(self.iteration, self.loss_values[-1])  # We could use some logging instead of prints

            # Callback to stop computation if user asks for it
            if self.progress_cb:
                # req = self.progress_cb(self.iteration / self.maxiter,
                #                       f" iteration {self.iteration}/{self.maxiter}",
                #                      {f"loss_{self.iteration}": self.loss_values})
                req = self.progress_cb(loss)
                if req.get("cancel_requested"):
                    raise RuntimeError("Cancel requested")

        return loss

    def _qubo_energy(self, bitstring: str) -> float:
        if self._qubo_matrix_np is None:
            raise ValueError("QUBO matrix not initialized for energy evaluation.")
        clean_state = bitstring if isinstance(bitstring, str) else "".join(bitstring)
        vector = np.fromiter((int(bit) for bit in clean_state), dtype=float, count=len(clean_state))
        return float(vector @ self._qubo_matrix_np @ vector)

    def _select_best_bitstring(self) -> Optional[str]:
        if not self.output_dict:
            return None
        if self._qubo_matrix_np is None:
            return max(self.output_dict, key=self.output_dict.get)

        best_state: Optional[str] = None
        best_energy = float("inf")
        best_prob = -1.0
        for state, prob in self.output_dict.items():
            energy = self._qubo_energy(state)
            if energy < best_energy - 1e-12 or (
                np.isclose(energy, best_energy) and prob > best_prob
            ):
                best_energy = energy
                best_prob = prob
                best_state = state
        return best_state

    def execute_optimization(self, maxiter: int = 300, xatol: float = 1e-4):
        """Run the classical optimizer that tunes the variational parameters."""

        self.maxiter = maxiter
        options = {"tol": 1e-1, "maxiter": self.maxiter}

        self.avg = 0

        modes_available = 128
        group_size_array, _ = OptimalGroups(modes_available, self.qubit_num)
        group_size_array = [int(x) for x in group_size_array]

        total_parameters = AnsatzFactory.calculate_total_parameters(
            group_size_array, self.rots
        )
        min_required_iterations = total_parameters + 2
        options["maxiter"] = max(options["maxiter"], min_required_iterations)

        init_param = 2 * np.pi * np.random.rand(total_parameters)

        result = minimize(
            self.minimize_loss, init_param, method=self.method, options=options
        )

        if self.qubo_matrix:
            return {
                "minimum": -min(self.loss_values),
                "optimal bitstring": self.bitstring,
                "optimizer_status": result.status,
            }

        if self.geometry:
            return {
                "minimum": min(self.loss_values),
                "Difference with theoretical result": self.exact
                - min(self.loss_values),
                "optimizer_status": result.status,
            }

        return {"loss": str(self.loss_values), "minimum": min(self.loss_values), "optimizer_status": result.status}
