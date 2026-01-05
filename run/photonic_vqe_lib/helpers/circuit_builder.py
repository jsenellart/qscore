"""This script defines utilities for building an ansatz necessary for vqe for a given group size array"""

import perceval as pcvl

from .entanglement_new import *


class AnsatzFactory:

    def __init__(self):
        self._circ = None
        self._lp = None
        self._angle_offset = 0
        self._layers = None
        # Map layers to their corresponding functions
        self._layer_funcs = {
            'X': apply_rotations_to_qubits_X,
            'Y': apply_rotations_to_qubits_Y,
            'Z': apply_rotations_to_qubits_Z
        }

    @staticmethod
    def _two_qubit_unitary(angle_list):
        """circuit ansatz for any 2 qubit unitary"""

        circ = pcvl.Circuit(4, name="U")

        #Single Qubit unitary
        circ.add(0, AnsatzFactory.RY02(angle_list[0]), merge=True)
        circ.add(0, AnsatzFactory.RY12(angle_list[1]), merge=True)

        circ.add(0, AnsatzFactory.InternalCX2(), merge=True)

        circ.add(0, AnsatzFactory.RY02(angle_list[2]), merge=True)
        circ.add(0, AnsatzFactory.RY12(angle_list[3]), merge=True)

        circ.add(0, AnsatzFactory.InternalCX2(), merge=True)

        circ.add(0, AnsatzFactory.RY02(angle_list[4]), merge=True)
        circ.add(0, AnsatzFactory.RY12(angle_list[5]), merge=True)

        circ.add(0, AnsatzFactory.InternalCX2(), merge=True)

        circ.add(0, AnsatzFactory.RY02(angle_list[6]), merge=True)
        circ.add(0, AnsatzFactory.RY12(angle_list[7]), merge=True)

        return circ

    @staticmethod
    def RY12(angle):
        circ = pcvl.Circuit(4, name="RY1")
        ry = pcvl.BS.Ry(theta=angle)
        circ.add((0, 1), ry)
        circ.add((2, 3), ry)
        return circ

    @staticmethod
    def RY02(angle):
        circ = pcvl.Circuit(4, name="RY0")

        circ.add(0, comp.PERM([0, 2, 1, 3]))
        ry = pcvl.BS.Ry(theta=angle)
        circ.add((0, 1), ry)
        circ.add((2, 3), ry)

        circ.add(0, comp.PERM([0, 2, 1, 3]))
        return circ

    @staticmethod
    def InternalCX2():
        circ = pcvl.Circuit(4, name="CX")
        circ.add(0, comp.PERM([0, 1, 3, 2]))
        return circ

    def apply_layer_operations(self, offset, size):

        """
        Applies a set of layer operations to a segment of the quantum circuit.

        Args:
            offset (int): The starting mode on which to apply the layer operations.
            size (int): The size of the qubit group for which to apply layer operations.

        Returns:
            int: The updated angle_offset after applying all the layers.
        """
        for layer in self._layers:
            self._circ.add(offset,
                           self._layer_funcs[layer](0, self._lp[self._angle_offset:self._angle_offset + size], size))
            self._angle_offset += size

    def apply_permutation(self, current_cumulative_size, shift_count):

        """
        Applies a permutation to the quantum circuit to create ancillary modes
        for controlled operations. This function modifies the circuit object
        'circ' in place.

        Args:
            current_cumulative_size (int): The current size of the qubit groups
                                           up to the current point in the circuit.
            shift_count (int): A count variable to keep track of the number of
                               shifts applied to the ancillary modes. Helps in
                               determining where the next ancillary modes will be
                               placed.
        """

        # Create initial permutation as identity
        perm = list(range(self._circ.m))

        # Define the swap positions based on the current_cumulative_size
        first_ancilla_swap = current_cumulative_size + 1 - 2 * shift_count
        second_ancilla_swap = current_cumulative_size + 2 - 2 * shift_count

        # Swap the positions with the last two positions minus the shift count
        perm[-2 - 2 * shift_count], perm[first_ancilla_swap] = perm[first_ancilla_swap], perm[-2 - 2 * shift_count]
        perm[-1 - 2 * shift_count], perm[second_ancilla_swap] = perm[second_ancilla_swap], perm[-1 - 2 * shift_count]

        # print(perm)
        self._circ.add(0, pcvl.PERM(perm))

    def build_qubit_circuit(self, qubit_group_sizes, lp, layers, entanglement_type="linear", ctype="cx", apply_reverse=False):
        """
        Builds a quantum circuit based on specified parameters. The circuit is generated
        for multiple groups of qubits with custom operations and entanglement.

        Args:
            qubit_group_sizes (list of int): List of sizes for each group of qubits.
            lp (list of float): List of angles for the parameterized gates.
            layers (list of str): Types of rotation layers to apply ('X', 'Y', 'Z').
            entanglement_type (str, optional): The type of entanglement to use ("maximal", "circular", or others). Defaults to "maximal".
            ctype (str, optional): The type of controlled operation to use ("cz" or "cx"). Defaults to "cx".
            apply_reverse (bool, optional): Whether to apply the reverse of the controlled operation. Defaults to False.

        Returns:
            pcvl.Circuit: The constructed quantum circuit.
        """

        total_modes = sum([2**n for n in qubit_group_sizes]) + 2*(len(qubit_group_sizes)-1)
        self._circ = pcvl.Circuit(total_modes, name="Machine Learning")
        self._layers = layers
        self._angle_offset = 0
        self._lp = lp
        offset = 0

        if ctype == "cx":
            if entanglement_type == "maximal":
                U_n_func = U_n_maximal
              #  U_n_reverse_func = U_n_maximal_reverse if apply_reverse else None
            elif entanglement_type == "circular":
                U_n_func = U_n_circular
              #  U_n_reverse_func = U_n_circular_reverse if apply_reverse else None
            else:
                U_n_func = U_n
              #  U_n_reverse_func = U_n_reverse if apply_reverse else None

        elif ctype == "cz":
            if entanglement_type == "maximal":
                U_n_func = U_n_maximal_CZ
              #  U_n_reverse_func = U_n_maximal_reverse_CZ if apply_reverse else None
          #  elif entanglement_type == "circular":
             #   raise NotImplementedError("CZ circular entanglement not implemented")  # To remove when below is defined
           #     U_n_func = U_n_circular_CZ  # Not defined
              #  U_n_reverse_func = U_n_circular_reverse_CZ if apply_reverse else None
            else:
                U_n_func = U_n_CZ
              #  U_n_reverse_func = U_n_reverse_CZ if apply_reverse else None

        if len(qubit_group_sizes) == 1 and qubit_group_sizes[0] == 2:
            return self._two_qubit_unitary(lp)

        if len(qubit_group_sizes) == 1 and qubit_group_sizes[0] == 3:
            self.apply_layer_operations(offset, qubit_group_sizes[0])
            self._circ.add(offset, U_n_func(qubit_group_sizes[0]))
           # if U_n_reverse_func:
           #     self._circ.add(offset, U_n_reverse_func(qubit_group_sizes[0]))

            self.apply_layer_operations(offset, qubit_group_sizes[0])
            return self._circ


        """
        This next block handles the circuit construction for cases where the number of qubit groups is two or more. 
        The function adds unitary operations, controlled gates, and other necessary operations to the 
        circuit for each group of qubits in the problem.
    
        1. Apply operations for the first group:
           - Use the apply_layer_operations() function to set rotational layers based on the angles.
           - Add entanglement specified by U_n_func.
           - Optionally, add the reverse of the entanglement operations if U_n_reverse_func is specified.
    
        2. Update the offset and current_cumulative_size to consider the size of the next qubit groups.
    
        3. Loop over middle groups:
           - For each middle group, apply layer operations, entanglement operations, and their reverse if needed.
           - A permutation is applied before and after adding a Generalized Controlled-Z (CZ) gate.
           - The offset and current_cumulative_size are updated after each iteration.
    
        4. Apply operations and CZ for the last group:
           - Just like the first and middle groups, operations are applied.
           - A final set of operations is added for all groups after resetting the offset.
    
        Args:
            qubit_group_sizes (list): List of sizes for each qubit group.
            circ (Circuit object): The circuit where the operations are being added.
            U_n_func (function): Function to apply unitary operations.
            U_n_reverse_func (function, optional): Function to apply the reverse of unitary operations.
            offset (int): The initial offset for adding operations in the circuit.
            angle_offset (int): The initial angle offset for rotations.
    
        Returns:
            circ (Circuit object): The updated circuit with all the required operations for each qubit group.
        """

        if len(qubit_group_sizes) >= 2:
            # Apply operations for the first group
            self.apply_layer_operations(offset, qubit_group_sizes[0])
            self._circ.add(offset, U_n_func(qubit_group_sizes[0]))
           # if U_n_reverse_func:
           #     self._circ.add(offset, U_n_reverse_func(qubit_group_sizes[0]))

            self.apply_layer_operations(offset, qubit_group_sizes[0])
            offset += 2**qubit_group_sizes[0]

            # Apply operations and CZs for middle groups
            current_cumulative_size = 2**qubit_group_sizes[0] + 2**qubit_group_sizes[1]
            shift_count = 0

            for idx, size in enumerate(qubit_group_sizes[1:-1]):
                angle_offset = self.apply_layer_operations(offset, angle_offset, size)
                self._circ.add(offset, U_n_func(size))
              #  if U_n_reverse_func:
              #      self._circ.add(offset, U_n_reverse_func(size))

                self.apply_layer_operations(offset, size)

                # Permutation before the CZ operation
                self.apply_permutation(current_cumulative_size, shift_count)
                self._circ.add(offset - 2**qubit_group_sizes[idx], GeneralizedCZ(qubit_group_sizes[idx], size))
                self.apply_permutation(current_cumulative_size, shift_count)
                shift_count += 1  # Increment the shift count after each CZ

                self.apply_layer_operations(offset, size)
             #   if U_n_reverse_func:
              #      self._circ.add(offset, U_n_reverse_func(size))

                self._circ.add(offset, U_n_func(size))
                self.apply_layer_operations(offset, size)

                # Update the current_cumulative_size for the next iteration
                current_cumulative_size += 2**size
                offset += 2**size

            # Apply operations and CZ for the last group
            self.apply_layer_operations(offset, qubit_group_sizes[-1])
            self._circ.add(offset, U_n_func(qubit_group_sizes[-1]))

           # if U_n_reverse_func:
            #    self._circ.add(offset, U_n_reverse_func(qubit_group_sizes[-1]))

            self.apply_layer_operations(offset, qubit_group_sizes[-1])
            self._circ.add(offset - 2**qubit_group_sizes[-2],
                           GeneralizedCZ(qubit_group_sizes[-2], qubit_group_sizes[-1]))
            # offset += 2**qubit_group_sizes[-1]

            # Reset offset and apply the final set of operations for all groups
            offset = 0
            for size in qubit_group_sizes:
                self.apply_layer_operations(offset, size)
              #  if U_n_reverse_func:
              #      self._circ.add(offset, U_n_reverse_func(size))

                self._circ.add(offset, U_n_func(size))
                self.apply_layer_operations(offset, size)
                offset += 2**size
        return self._circ

    @staticmethod
    def calculate_total_parameters(qubit_group_sizes, layers):

        """
        Calculate the total number of parameters needed for a quantum circuit
        with the given qubit group sizes and layers.

        Args:
            qubit_group_sizes (list of int): A list containing the sizes of each qubit group in the circuit.
            layers (list of str): A list specifying the types of layers (e.g., 'X', 'Y', 'Z') in the circuit.

        Returns:
            int: The total number of parameters required for the circuit.

        Note:
            The function takes into account special cases:
            1) If there's only one group of size 2, it returns 8.
            2) Otherwise, it calculates based on the depth and size of each group.
        """

        if len(qubit_group_sizes) == 1 and qubit_group_sizes[0] == 2:
            return 8
        if len(qubit_group_sizes) == 1:
            size = qubit_group_sizes[0]
            depth = 2  # single group path applies two rotation layers
            return depth * size * len(layers)
        if qubit_group_sizes[0]==2 and qubit_group_sizes[1]==2:
            return 16

        if len(qubit_group_sizes) == 2:

            # Calculate the total depth for each group
            depths = [2,2]

            # Calculate parameters per depth for each group
           # parameters = [2*depth * size  for depth, size in zip(depths, qubit_group_sizes)]
            parameters = [4*size for size in qubit_group_sizes]
           # print()
            #print(sum(parameters))
            return sum(parameters)
            #return 44

        if len(qubit_group_sizes) > 2:

            # Calculate the total depth for each group
            depths = [2 if (i == 0 or i == len(qubit_group_sizes) - 1) else 3 for i in range(len(qubit_group_sizes))]

            # Calculate parameters per depth for each group
            parameters = [depth * size * 2 for depth, size in zip(depths, qubit_group_sizes)]

            return sum(parameters)

        raise ValueError(f"Could not compute a number of parameter with {qubit_group_sizes}, {layers}")
