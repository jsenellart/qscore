#toggling G_RHK makes the below CX into a CZ.

"""This script defines certain utilities for creating entanglement, local and otherwise which the user can change as they please"""


import perceval as pcvl


from .rotations import *



def RalphCZ():
    
    """
    Generates an unbalanced CZ gate using Ralph's configuration.

    Returns:
        pcvl.Circuit: Custom CZ gate as a Perceval circuit object.
    """
    cnot = pcvl.Circuit(6, name="Ralph CZ")
    cnot.add((0,1,2,3,4,5), comp.PERM([1,2,3,4,0,5]))
    cnot.add((0, 1), pcvl.BS.H(pcvl.BS.r_to_theta(1/3), phi_tl = -np.pi/2, phi_bl = np.pi, phi_tr = np.pi / 2))
    #cnot.add((3, 4), pcvl.BS.H())
    cnot.add((2, 3), pcvl.BS.H(pcvl.BS.r_to_theta(1/3), phi_tl = -np.pi/2, phi_bl = np.pi, phi_tr = np.pi / 2))
    cnot.add((4, 5), pcvl.BS.H(pcvl.BS.r_to_theta(1/3)))
    #cnot.add((3, 4), pcvl.BS.H())
    cnot.add((0,1,2,3,4,5), comp.PERM([4,0,1,2,3,5]))
    #print("called")
    return cnot

def RalphCNOT():
    
    """
    Generates a custom CNOT gate using Ralph's configuration.

    Returns:
        pcvl.Circuit: Custom CNOT gate as a Perceval circuit object.
    """
    cnot = pcvl.Circuit(6, name="Ralph CZ")
    cnot.add((0,1,2,3,4,5), comp.PERM([1,2,3,4,0,5]))
    cnot.add((0, 1), pcvl.BS.H(pcvl.BS.r_to_theta(1/3), phi_tl = -np.pi/2, phi_bl = np.pi, phi_tr = np.pi / 2))
    cnot.add((3, 4), pcvl.BS.H())
    cnot.add((2, 3), pcvl.BS.H(pcvl.BS.r_to_theta(1/3), phi_tl = -np.pi/2, phi_bl = np.pi, phi_tr = np.pi / 2))
    cnot.add((4, 5), pcvl.BS.H(pcvl.BS.r_to_theta(1/3)))
    cnot.add((3, 4), pcvl.BS.H())
    cnot.add((0,1,2,3,4,5), comp.PERM([4,0,1,2,3,5]))
    #print("called")
    return cnot

def GeneralizedCZ(n, m):
    """
    Generates a generalized CZ gate for `n` control and `m` target qubits.
    
    Args:
        n (int): Number of control qubits.
        m (int): Number of target qubits.
    
    Returns:
        pcvl.Circuit: Generalized CZ gate as a Perceval circuit object.
    """
    total_modes = 2**n + 2**m + 2  # +2 for auxiliary modes

    # Identify controls, targets, and swap positions
    control1, control2 = 2**n - 2, 2**n - 1
    target1, target2 = 2**n + 2**m - 2, 2**n + 2**m - 1
    swap1, swap2 = target1 - 2, target1 - 1

    # Create the full list of modes
    modes = list(range(total_modes))

    # Perform the swapping operations
    modes[control1], modes[swap1] = modes[swap1], modes[control1]
    modes[control2], modes[swap2] = modes[swap2], modes[control2]
    circ = pcvl.Circuit(total_modes, name="GeneralizedCZ")
    circ.add(tuple(range(total_modes)), comp.PERM(modes))
    circ.add((target1 - 2), RalphCZ())
    circ.add(tuple(range(total_modes)), comp.PERM(modes))  # Inverse permutation

    return circ


def generate_permutation_for_controlled_op(control, target, num_qubits):
    
    """ Generate the permutation required for a controlled operation (either CNOT or CZ) """
    
    N = 2**num_qubits
    perm = list(range(N))

    for i in range(N):
        binary = format(i, f'0{num_qubits}b')
        
        if binary[control] == '1':
            flipped = list(binary)
            flipped[target] = '0' if flipped[target] == '1' else '1'
            perm[i] = int("".join(flipped), 2)
    
    return list(tuple(perm))

def create_internal_controlled_op(op_type, control, target, num_qubits):
    circ = pcvl.Circuit(2**num_qubits, name=f"{op_type}")
    perm = generate_permutation_for_controlled_op(control, target, num_qubits)

    if op_type == "CZ":
        circ.add(0, G_RHk(num_qubits, target))
    
    circ.add(0, comp.PERM(perm))
    
    if op_type == "CZ":
        circ.add(0, G_RHk(num_qubits, target))

    return circ

def generate_chained_controlled_ops(op_type, n, reverse=False, circular=False, maximal=False):
    """
    Generates a circuit with a chain of controlled operations.

    Args:
        op_type (str): Type of controlled operation ("CZ" or "CX").
        n (int): Number of qubits in the circuit.
        reverse (bool): Whether to reverse the order of the operations.
        circular (bool): Whether to make the operations circular.
        maximal (bool): Whether to generate all possible combinations of i and j.

    Returns:
        pcvl.Circuit: A circuit containing the chained controlled operations.
    """

    circ = pcvl.Circuit(2**n, name=f"{op_type}{n}")
    
    range_values = range(n-1, 0, -1) if reverse else range(n-1)
    
    for i in range_values:
        j = i + 1 if not reverse else i - 1
        circ.add(0, create_internal_controlled_op(op_type, i, j, n))

    if circular:
        i, j = (0, n-1) if not reverse else (n-1, 0)
        circ.add(0, create_internal_controlled_op(op_type, i, j, n))

    if maximal:
        for i in range(n):
            for j in range(n):
                if i != j:
                    circ.add(0, create_internal_controlled_op(op_type, i, j, n))

    return circ


# Examples:
def U_n(n):
    return generate_chained_controlled_ops("CX", n)


def U_n_circular(n):
    return generate_chained_controlled_ops("CX", n, circular=True)


def U_n_circular_reverse(n):
    return generate_chained_controlled_ops("CX", n, reverse=True, circular=True)


def U_n_reverse(n):
    return generate_chained_controlled_ops("CX", n, reverse=True)


def U_n_reverse_CZ(n):
    return generate_chained_controlled_ops("CZ", n, reverse=True)


def U_n_CZ(n):
    return generate_chained_controlled_ops("CZ", n)


def U_n_maximal_CZ(n):
    return generate_chained_controlled_ops("CZ", n, maximal=True)


def U_n_maximal_reverse_CZ(n):
    return generate_chained_controlled_ops("CZ", n, reverse=True, maximal=True)


def U_n_maximal(n):
    return generate_chained_controlled_ops("CX", n, maximal=True)


def U_n_maximal_reverse(n):
    return generate_chained_controlled_ops("CX", n, reverse=True, maximal=True)
