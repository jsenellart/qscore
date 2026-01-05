import perceval as pcvl

import perceval.components.unitary_components as comp


def InternalSwap(qubit1: int, qubit2: int, num_qubits:int = 4):
    """
    Creates an internal SWAP gate that swaps two qubits within a quantum circuit.
    
    Args:
        qubit1 (int): The index of the first qubit to be swapped.
        qubit2 (int): The index of the second qubit to be swapped.
        num_qubits (int, optional): The total number of qubits in the circuit. Default is 4.

    Returns:
        pcvl.Circuit: A Perceval circuit object representing the SWAP operation.

    Raises:
        ValueError: If either of the qubit indices are invalid or if they are the same.

    Example:
        InternalSwap(1, 3, 4) will swap the second and fourth qubits in a 4-qubit system.

    Note:
        The function constructs a permutation that describes the SWAP operation for
        the given qubits and adds that permutation to the Perceval circuit.
    """

    # Check if qubit1 and qubit2 are valid qubit indices
    if qubit1 >= num_qubits or qubit2 >= num_qubits or qubit1 == qubit2:
        raise ValueError("Invalid qubit indices for SWAP.")
    
    # Calculate the number of possible states
    num_states = 2**num_qubits

    # Create a list of all possible states
    states = [bin(i)[2:].zfill(num_qubits) for i in range(num_states)]

    # Generate the swapped states list
    swapped_states = []
    for state in states:
        state_list = list(state)
        # Swap the bits of the two qubits
        state_list[qubit1], state_list[qubit2] = state_list[qubit2], state_list[qubit1]
        swapped_states.append(''.join(state_list))

    # Map the states to their indices to get the permutation
    permutation = [states.index(swap_state) for swap_state in swapped_states]

    # Construct the circuit
    circ = pcvl.Circuit(num_states, name=f"SWAP{qubit1}{qubit2}")
    circ.add(tuple(range(num_states)), comp.PERM(permutation))
    return circ


def _generate_rotation_kth_qubit(gate_layer: pcvl.Circuit, nqubits: int, k: int, circuit_name: str):
    """Apply the Hadamard gate to the k-th qubit of n qubits."""
    if k == nqubits - 1:
        return gate_layer

    circ = pcvl.Circuit(2 ** nqubits, name=circuit_name)

    # Add the internal swap gate to swap the k-th qubit to the (n-1)-th qubit.
    circ.add(0, InternalSwap(k, nqubits - 1, nqubits), merge=True)

    # Apply the rotation gate on the (nqubits-1)-th qubit.
    circ.add(0, gate_layer, merge=True)

    # Revert the modes to their original positions.
    circ.add(0, InternalSwap(k, nqubits - 1, nqubits), merge=True)

    return circ


def _generate_rotation_last_qubit(gate: pcvl.Circuit, nqubits: int, circuit_name: str):
    """Apply the RZ gate to nth qubit."""
    circ = pcvl.Circuit(2 ** nqubits, name=circuit_name)
    for i in range(0, 2 ** nqubits, 2):
        circ.add(i, gate)
    return circ


def G_RYn(angle, n):
    """Apply the RY gate to nth qubit."""
    return _generate_rotation_last_qubit(
        pcvl.BS.Ry(theta=angle, phi_tl=0, phi_bl=0, phi_tr=0, phi_br=0), nqubits=n, circuit_name=f"RY{n}")


def G_RYk(angle, n, k):
    """Apply the RY gate to the k-th qubit of n qubits."""
    return _generate_rotation_kth_qubit(G_RYn(angle, n), n, k, f"RY{k}")


def G_RZn(angle, n):
    """Apply the RZ gate to nth qubit."""
    return _generate_rotation_last_qubit(
        pcvl.BS.Rx(theta=0, phi_tl=-angle/2, phi_bl=angle/2, phi_tr=0, phi_br=0), nqubits=n, circuit_name=f"RZ{n}")


def G_RZk(angle, n, k):
    """Apply the RZ gate to the k-th qubit of n qubits."""
    return _generate_rotation_kth_qubit(G_RZn(angle, n), n, k, f"RZ{k}")


def G_RXn(angle, n):
    """Apply the RX gate to nth qubit."""
    return _generate_rotation_last_qubit(
        pcvl.BS.Rx(theta=angle), nqubits=n, circuit_name=f"RX{n}")


def G_RXk(angle, n, k):
    """Apply the RX gate to the k-th qubit of n qubits."""
    return _generate_rotation_kth_qubit(G_RXn(angle, n), n, k, f"RX{k}")


def G_RHn(n):
    """Apply the Hadamard gate to nth qubit."""
    return _generate_rotation_last_qubit(pcvl.BS.H(), nqubits=n, circuit_name=f"RH{n-1}")


def G_RHk(n, k):
    """Apply the Hadamard gate to the k-th qubit of n qubits."""
    return _generate_rotation_kth_qubit(G_RHn(n), n, k, f"RH{k}")


def G_RYadn(n):
    """Apply the Pauli Y measurement gate to nth qubit."""
    return _generate_rotation_last_qubit(
        pcvl.BS.H(5 * np.pi / 2, np.pi, np.pi / 2), nqubits=n, circuit_name=f"RYad{n-1}")


def G_RYadk(n, k):
    """Apply the Pauli Y measurement to the k-th qubit of n qubits."""
    return _generate_rotation_kth_qubit(G_RYadn(n), n, k, f"RYad{k}")


def apply_rotations_to_group(base, angle, n):
    """Apply the RY gate to each qubit in a group of n qubits starting from a specific base mode."""
    circ = pcvl.Circuit(2**n, name=f"GroupRY{n}")
    
    # Apply the rotation gate on the last qubit of the group.
    circ.add(base, G_RYn(angle, n), merge=True)
    
    # Apply the rotation gate on each of the other qubits of the group.
    for k in range(n-1):
        circ.add(base, G_RYk(angle, n, k), merge=True)
    
    return circ

def apply_rotations_to_qubits_Y(base, angle_list, n):
    """Apply the RY gate to each qubit in a group of n qubits based on an angle list."""
    if len(angle_list) != n:
        raise ValueError("Angle list should match the number of qubits in the group.")
    
    circ = pcvl.Circuit(2**n, name=f"GroupRYMulti{n}")
    
    # Apply the rotation gate for each qubit based on the provided angle.
    for idx, angle in enumerate(angle_list):
        circ.add(base, G_RYk(angle, n, idx), merge=True)
    
    return circ

def apply_rotations_to_qubits_X(base, angle_list, n):
    """Apply the RX gate to each qubit in a group of n qubits based on an angle list."""
    if len(angle_list) != n:
        raise ValueError("Angle list should match the number of qubits in the group.")
    
    circ = pcvl.Circuit(2**n, name=f"GroupRYMulti{n}")
    
    # Apply the rotation gate for each qubit based on the provided angle.
    for idx, angle in enumerate(angle_list):
        circ.add(base, G_RXk(angle, n, idx), merge=True)
    
    return circ

def apply_rotations_to_qubits_Z(base, angle_list, n):
    """Apply the RZ gate to each qubit in a group of n qubits based on an angle list."""
    if len(angle_list) != n:
        raise ValueError("Angle list should match the number of qubits in the group.")
    
    circ = pcvl.Circuit(2**n, name=f"GroupRYMulti{n}")
    
    # Apply the rotation gate for each qubit based on the provided angle.
    for idx, angle in enumerate(angle_list):
        circ.add(base, G_RZk(angle, n, idx), merge=True)
    
    return circ


import numpy as np
import string

def set_parameters2(pauli_string, group_sizes):
    """
    Set the phase shift parameters based on the Pauli string and group sizes.

    Args:
    - pauli_string (str): The string containing Pauli operators.
    - group_sizes (list): List containing the number of qubits in each group.

    Returns:
    - dict: Dictionary containing the phase shift values for each rotation.
    """
    
    # Ensure the pauli string length matches the total number of qubits
    assert len(pauli_string) == sum(group_sizes), "Mismatch between Pauli string length and total qubits."
    pauli_string = pauli_string[::-1]
    # Define the phase values for each Pauli operator
    phase_values = {
        'X': {'i': 1.517189, 'j': 4.765996},
        'Y': {'i': 3*np.pi/2, 'j': np.pi},
        'Z': {'i': np.pi, 'j': np.pi},
        'I': {'i': np.pi, 'j': np.pi}
    }
    
    parameters = {}
    offset = 0
    
    for group_index, group_size in enumerate(group_sizes):
        for qubit_index in range(group_size):
            # Get the corresponding Pauli operator for this qubit
            pauli_op = pauli_string[offset + qubit_index]
            
            # For each qubit, generate phase shifters for all modes within the group
            modes = 2**group_size
            for i in range(0, modes, 2):
                # Determine the letter associated with the mode
                letter = string.ascii_lowercase[i // 2]
                
                # Set the phase shift values using the systematic naming convention
                for suffix in ['i', 'j']:
                    name = f"Rot_{group_index}_{letter}{qubit_index}{suffix}"
                    parameters[name] = phase_values[pauli_op][suffix]
                
        offset += group_size

    return {'circuit_params': parameters}


def set_parameters(pauli_string, group_sizes):
    """
    Set the phase shift parameters based on the Pauli string and group sizes.

    Args:
    - pauli_string (str): The string containing Pauli operators.
    - group_sizes (list): List containing the number of qubits in each group.

    Returns:
    - dict: Dictionary containing the phase shift values for each rotation, nested under 'circuit_params'.
    """

    # Ensure the pauli string length matches the total number of qubits
    assert len(pauli_string) == sum(group_sizes), "Mismatch between Pauli string length and total qubits."
    pauli_string = pauli_string[::-1]
    # Define the phase values for each Pauli operator
    phase_values = {
        'X': {'i': 1.517189, 'j': 4.765996},
        'Y': {'i': 3 * np.pi / 2, 'j': np.pi},
        'Z': {'i': np.pi, 'j': np.pi},
        'I': {'i': np.pi, 'j': np.pi}
    }

    parameters = {}
    offset = 0

    for group_index, group_size in enumerate(group_sizes):
        for qubit_index in range(group_size):
            # Get the corresponding Pauli operator for this qubit
            pauli_op = pauli_string[offset + qubit_index]

            # For each qubit, generate phase shifters for all modes within the group
            modes = 2 ** group_size
            for i in range(0, modes, 2):
                # Generate the letter associated with the mode using ASCII values
                letter = chr(97 + (i // 2))  # ASCII value of 'a' is 97

                # Set the phase shift values using the systematic naming convention
                for suffix in ['i', 'j']:
                    name = f"Rot_{group_index}_{letter}{qubit_index}{suffix}"
                    parameters[name] = phase_values[pauli_op][suffix]

        offset += group_size

    # Nest the parameters under 'circuit_params'
    return {'circuit_params': parameters}


#print(set_parameters('XZXY', [2,2]))

def params(measurement_bases, group_size_array):
    params = []
    for i in range(len(measurement_bases)):
        a = set_parameters(measurement_bases[i], group_size_array)
        params.append(a)
    return params
#print(params(["ZZZZ"], [2,2]))
#measurement_bases = ["XXXX", "XYXY", "XZZX"]
#group_size_array = [2,2]
#print(params(measurement_bases, group_size_array)[0])
#print(params(measurement_bases, group_size_array)[1])
#print(params(measurement_bases, group_size_array)[2])


def create_phase_shifter_circuit(n, group_index, k, offset=0):
    """Create a phase shifter circuit for a qudit group.

    Args:
    - n (int): Number of qubits in the group.
    - group_index (int): Index of the group being processed.
    - k (int): The specific qubit index in the group the RY gate is targeting.
    - offset (int, optional): Mode offset for the starting point in the group. Defaults to 0.

    Returns:
    - pcvl.Circuit: The constructed phase shifter circuit.
    """
    modes = 2**n
    circ = pcvl.Circuit(modes)
    phase_shifters = {}

    for i in range(0, modes, 2):
        # Get the corresponding letter for the mode
        letter = string.ascii_lowercase[i // 2]

        # Define the phase shifter parameters
        phase_shifters[f"Rot_{group_index}_{letter}{k}i"] = pcvl.P(f"Rot_{group_index}_{letter}{k}i")
        phase_shifters[f"Rot_{group_index}_{letter}{k}j"] = pcvl.P(f"Rot_{group_index}_{letter}{k}j")

        # Create the phase shifter circuit for the current mode
        hadamard_circuit = pcvl.Circuit(2) \
                           // pcvl.PS(phase_shifters[f"Rot_{group_index}_{letter}{k}i"]) \
                           // pcvl.BS() \
                           // (0, pcvl.PS(phase_shifters[f"Rot_{group_index}_{letter}{k}j"])) \
                           // pcvl.BS()

        circ.add((i + offset, i + offset + 1), hadamard_circuit, merge=True)

    return circ



import numpy as np
import string

def get_mode_letter(index):
    alphabet = string.ascii_lowercase
    length = len(alphabet)

    if index < length:
        return alphabet[index]
    else:
        return alphabet[(index // length) - 1] + alphabet[index % length]


def set_parameters(pauli_string, group_sizes):
    assert len(pauli_string) == sum(group_sizes), "Mismatch between Pauli string length and total qubits."
    pauli_string = pauli_string[::-1]

    phase_values = {
        'X': {'i': 1.517189, 'j': 4.765996},
        'Y': {'i': 3 * np.pi / 2, 'j': np.pi},
        'Z': {'i': np.pi, 'j': np.pi},
        'I': {'i': np.pi, 'j': np.pi}
    }

    parameters = {}
    offset = 0

    for group_index, group_size in enumerate(group_sizes):
        for qubit_index in range(group_size):
            pauli_op = pauli_string[offset + qubit_index]
            modes = 2 ** group_size
            for i in range(0, modes, 2):
                letter = get_mode_letter(i // 2)
                for suffix in ['i', 'j']:
                    name = f"Rot_{group_index}_{letter}{qubit_index}{suffix}"
                    parameters[name] = phase_values[pauli_op][suffix]

        offset += group_size

    return {'circuit_params': parameters}

def create_phase_shifter_circuit(n, group_index, k, offset=0):
    modes = 2**n
    circ = pcvl.Circuit(modes)
    phase_shifters = {}

    for i in range(0, modes, 2):
        letter = get_mode_letter(i // 2)

        phase_shifters[f"Rot_{group_index}_{letter}{k}i"] = pcvl.P(f"Rot_{group_index}_{letter}{k}i")
        phase_shifters[f"Rot_{group_index}_{letter}{k}j"] = pcvl.P(f"Rot_{group_index}_{letter}{k}j")

        hadamard_circuit = pcvl.Circuit(2) \
                           // pcvl.PS(phase_shifters[f"Rot_{group_index}_{letter}{k}i"]) \
                           // pcvl.BS() \
                           // (0, pcvl.PS(phase_shifters[f"Rot_{group_index}_{letter}{k}j"])) \
                           // pcvl.BS()

        circ.add((i + offset, i + offset + 1), hadamard_circuit, merge=True)

    return circ



def Phase_kthqubit( n, k, group_index, offset=0):
    """
    Apply the phases to the k-th qubit of n qubits.
    
    Args:
    - angle (float): The rotation angle.
    - n (int): Total number of qubits in the group.
    - k (int): Index of the qubit being targeted (0-based).
    - group_index (int): Index of the group being processed.
    - offset (int, optional): Mode offset for the starting point in the group. Defaults to 0.
    
    Returns:
    - pcvl.Circuit: The constructed circuit.
    """
    modes = 2**n
    circ = pcvl.Circuit(modes)

    # Qiskit notation adjustment
    k_qiskit = k

    # If targeting the last qubit, apply the phase shifter circuit directly.
    if k_qiskit == n - 1:
        return create_phase_shifter_circuit(n, group_index, k, offset)

    # Otherwise, perform the internal swap, apply the rotation, and swap back.
    
    # Internal swap to move the k-th qubit to the (n-1)-th position.
    circ.add(0, InternalSwap(k_qiskit, n - 1, n), merge=True)
    
    # Apply the rotation on the (n-1)-th qubit.
    ry_circuit = create_phase_shifter_circuit(n, group_index, k, offset)
    circ.add(0, ry_circuit, merge=True)

    # Swap them back to their original positions.
    circ.add(0, InternalSwap(k_qiskit, n - 1, n), merge=True)

    return circ


def apply_all_phase_shifters(group_sizes):
    """
    Apply phase shifters to all qubits across all groups in a single circuit.
    
    Args:
    - group_sizes (list): List containing the number of qubits in each group.
    
    Returns:
    - pcvl.Circuit: Combined circuit for all groups.
    """

    def Phase_kthqubit(n, k, group_index ,offset ):
        modes = 2**n 
        circ = pcvl.Circuit(modes)
        k_qiskit = k

        if k_qiskit == n - 1:
            return create_phase_shifter_circuit(n, group_index, k)

        circ.add(0, InternalSwap(k_qiskit, n - 1, n), merge=True)
        ry_circuit = create_phase_shifter_circuit(n, group_index, k)
        circ.add(0, ry_circuit, merge=True)
        circ.add(0, InternalSwap(k_qiskit, n - 1, n), merge=True)

        return circ

    # Initialize the main circuit
    total_modes = sum(2**n for n in group_sizes)
    main_circuit = pcvl.Circuit(total_modes)

    offset = 0
    for group_index, group_size in enumerate(group_sizes):
        n = group_sizes[group_index]
        
        for k in range(n):
            phase_circuit_k = Phase_kthqubit(n, k, group_index, offset)
            main_circuit.add(offset, phase_circuit_k, merge=True)
        
        offset += 2**n

    return main_circuit


def rotate_group_to_z_basis(circuit: pcvl.Circuit, pauli_string: str, group_config: list):
    """Rotate qubits in the given groups back to the Z basis depending on the Pauli string."""
    
    # Copy the circuit to avoid in-place modifications.
    new_circuit = circuit.copy()
    pauli_string = pauli_string[::-1]

    # Calculate the total number of modes
    total_modes = sum([2**g for g in group_config])

    # We'll start by determining the starting mode for each group
    starting_modes = [0] + [sum([2**g for g in group_config[:idx]]) for idx in range(1, len(group_config))]

    # Iterate over the group_config and apply rotations as required by pauli_string
    pauli_index = 0
    for idx, group_size in enumerate(group_config):
        for k in range(group_size):
            pauli_op = pauli_string[pauli_index]
          
            target_mode = starting_modes[idx]

            if pauli_op == 'X':
                # If we need to rotate an X, we apply a Hadamard rotation.
                new_circuit.add(target_mode, G_RHk(group_size, k), merge=True)

            elif pauli_op == 'Y':
                # If we need to rotate a Y, we apply a Y-Hadamard rotation.
                new_circuit.add(target_mode, G_RYadk(group_size, k), merge=True)

            # We don't need to apply any rotations for Z, so we skip it.
            pauli_index += 1

    return new_circuit
