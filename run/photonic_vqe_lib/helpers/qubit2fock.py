import perceval as pcvl

def fock_to_qubit_state(fock_state, group_sizes):
    fock_state = [i for i in fock_state]
    # Calculate the length of the Fock state including postselected modes
    expected_length = sum([2**size for size in group_sizes]) + 2 * (len(group_sizes) - 1)
    if len(fock_state) != expected_length:
        return False

    # Initialize variables
    offset = 0
    qubit_state_binary = ""

    # Process each group
    for i, size in enumerate(group_sizes):
        group_length = 2**size

        # Extract the relevant part of the Fock state for this group
        group_fock_state = fock_state[offset:offset + group_length]

        # Check if the group state is valid
        if group_fock_state.count(1) != 1:
            return False

        # Find the position (index) where the state is 1
        state_index = group_fock_state.index(1)

        # Convert this index to a binary string representing the qubit state
        binary_state = format(state_index, f'0{size}b')
        qubit_state_binary += binary_state

        # Update the offset to the end of the current group
        offset += group_length

    # Check for photons in remaining modes (post-selected modes)
    if any(fock_state[offset:]):
        return False

    return qubit_state_binary

def to_fock_state(qubit_state, group_sizes):
    # Initialize variables
    offset = 0
    fock_state = []

    # Process each group
    for size in group_sizes:
        group_length = 2**size

        # Extract the relevant part of the qubit state for this group
        group_qubit_state = qubit_state[offset:offset + size]

        # Convert the binary string to an integer index
        state_index = int(group_qubit_state, 2)

        # Create the group Fock state with one photon in the 'state_index' position
        group_fock_state = [1 if i == state_index else 0 for i in range(group_length)]
        fock_state += group_fock_state

        # Update the offset to the end of the current group
        offset += size

    # Add zeros for post-selected modes
    expected_length = sum([2**size for size in group_sizes]) + 2 * (len(group_sizes) - 1)
    fock_state += [0] * (expected_length - len(fock_state))
    fock_state = pcvl.BasicState(fock_state)

    return fock_state

# Example usage
#qubit_state = "0101"
#group_sizes = [2, 2]
#fock_state = to_fock_state(qubit_state, group_sizes)
#print(fock_state)  # Example output for the provided qubit state and group sizes



def ato_fock_state(qubit_state, group_sizes):
    """
    input:
        qubit_state (str): the qubit state that we want to convert to a Fock state.
        group_sizes (lst): a list containing the size of each group.

    output:
        fock_state (pcvl.BasicState): the corresponding fock state.
    """

    # Create a list containing the initial qubit state divided in parts according to the group sizes
    qubit_state_lst = [qubit_state[0:+size] for size in group_sizes]

    vector = [0] * (sum([2 ** size for size in group_sizes]) + 2 * (len(group_sizes) - 1))
    hvector = [0] * (sum([2 ** size for size in group_sizes])) + [1] * 2 * (len(group_sizes) - 1)

    offset = 0
    for idx, size_bits in enumerate(qubit_state_lst):
        vector[offset + int(size_bits, 2)] = 1
        hvector[offset + int(size_bits, 2)] = 1
        offset += 2 ** group_sizes[idx]

    fock_state = pcvl.BasicState(vector)

    return fock_state

#print(to_fock_state("0011",[2,2]))
