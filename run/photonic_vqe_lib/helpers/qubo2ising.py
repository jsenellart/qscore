import numpy as np


def qubo_to_ising_ham(qubo_matrix: np.ndarray):
    """
    Function that converts qubo matrix to ising Hamiltonian operators.
    :param qubo_matrix (np.ndarray): qubo matrix.
    :return: hnames (list), hweights (list): lists of ising Hamiltonian operators and their weights.
    """
    hnames = []
    hweights = []

    # Processing linear terms
    for num1, row in enumerate(qubo_matrix):
        weight = -1 * sum(row) / 2
        string_temp = [('Z' if num1 == num2 else 'I') for num2 in range(len(row))]
        hnames.append(''.join(string_temp))
        hweights.append(weight)

    # Processing quadratic terms
    for n1, row in enumerate(qubo_matrix):
        for n2, r in enumerate(row):
            if r != 0 and n2 > n1:
                string_temp_q = ['I' for _ in range(len(row))]
                string_temp_q[n1] = 'Z'
                string_temp_q[n2] = 'Z'
                hnames.append(''.join(string_temp_q))
                hweights.append(r / 2)
    hnames2 =[]
    for i in hnames:
        hnames2.append(i[::-1])


    return hnames2, hweights

def qubo_to_ising_with_offset(qubo_matrix: np.ndarray) -> tuple[dict, int]:
    """
    Function that converts qubo matrix to ising Hamiltonian operators.
    Input:
        qubo_matrix (np.ndarray): qubo matrix.
    Output:
        ising_ham_dict (dict): ising Hamiltonian operators.
        offset (int): the offset value of the Hamiltonian
    """
    offset = 0
    linear_coef = []
    strings_linear = []

    for num1, row in enumerate(qubo_matrix):
        weight_lin = sum(row) / 2
        linear_coef.append(-1 * weight_lin)
        offset += weight_lin
        string_temp = []
        for num2, column in enumerate(qubo_matrix.transpose()):
            if num1 == num2:
                string_temp.append('Z')
            else:
                string_temp.append('I')
        strings_linear.append(''.join(string_temp))

    quadratic_coef = []
    strings_quadratic = []
    for n1, row in enumerate(qubo_matrix):
        for n2, r in enumerate(row):
            string_temp_q = ['I' for _ in range(len(row))]
            if r != 0 and n2 > n1:
                string_temp_q[n1] = 'Z'
                string_temp_q[n2] = 'Z'
                weight_quad = r / 2
                quadratic_coef.append(weight_quad)
                offset += -1 * weight_quad
            if 'Z' in string_temp_q:
                strings_quadratic.append(''.join(string_temp_q))

    ising_ham_dict = {}
    for n, string1 in enumerate(strings_linear):
        ising_ham_dict[string1] = linear_coef[n]

    for n, string2 in enumerate(strings_quadratic):
        ising_ham_dict[string2] = quadratic_coef[n]

    return offset

# Example usage
H = np.array([[31, 500, 500, 500, -500, -500, 0, 0, 0, 0, 0],
              [500, 32, 500, 500, 0, 0, -500, -500, 0, 0, 0],
              [500, 500, 32, 500, 0, 0, 0, 0, -500, -500, 0],
              [500, 500, 500, 32, 0, 0, 0, 0, 0, 0, -500],
              [-500, 0, 0, 0, 536, 500, 0, 0, 0, 0, 0],
              [-500, 0, 0, 0, 500, 538, 0, 0, 0, 0, 0],
              [0, -500, 0, 0, 0, 0, 536, 500, 0, 0, 0],
              [0, -500, 0, 0, 0, 0, 500, 538, 0, 0, 0],
              [0, 0, -500, 0, 0, 0, 0, 0, 536, 500, 0],
              [0, 0, -500, 0, 0, 0, 0, 0, 500, 538, 0],
              [0, 0, 0, -500, 0, 0, 0, 0, 0, 0, 538]])

#hnames, hweights = qubo_to_ising_ham(H)
H = np.array([np.array([-2.,  1.,  0.,  1.]), np.array([ 1., -2.,  1.,  0.]), np.array([ 0.,  1., -2.,  1.]), np.array([ 1.,  0.,  1., -2.])])

#print("Hamiltonian Operators:", hnames)
#print("Corresponding Weights:", hweights)


#print(qubo_to_ising_with_offset(H))


# Example usage
H = np.array([[31, 500, 500, 500, -500, -500, 0, 0, 0, 0, 0],
              [500, 32, 500, 500, 0, 0, -500, -500, 0, 0, 0],
              [500, 500, 32, 500, 0, 0, 0, 0, -500, -500, 0],
              [500, 500, 500, 32, 0, 0, 0, 0, 0, 0, -500],
              [-500, 0, 0, 0, 536, 500, 0, 0, 0, 0, 0],
              [-500, 0, 0, 0, 500, 538, 0, 0, 0, 0, 0],
              [0, -500, 0, 0, 0, 0, 536, 500, 0, 0, 0],
              [0, -500, 0, 0, 0, 0, 500, 538, 0, 0, 0],
              [0, 0, -500, 0, 0, 0, 0, 0, 536, 500, 0],
              [0, 0, -500, 0, 0, 0, 0, 0, 500, 538, 0],
              [0, 0, 0, -500, 0, 0, 0, 0, 0, 0, 538]])

#hnames, hweights = qubo_to_ising_ham(H)

#print("Hamiltonian Operators:", hnames)
#print("Corresponding Weights:", hweights)


import numpy as np


def qubo_to_ising_ham2(qubo_matrix: np.ndarray):
    """
    Function that converts qubo matrix to ising Hamiltonian operators and returns additional lists of names and weights.

    Input:
        qubo_matrix (np.ndarray): qubo matrix.

    Output:
        ising_ham_dict (dict): ising Hamiltonian operators.
        hnames (list): List of Hamiltonian operator names (Pauli strings).
        hweights (list): List of corresponding weights for the Hamiltonian operators.
        offset (int): the offset value of the Hamiltonian
    """
    offset = 0
    linear_coef = []
    strings_linear = []

    for num1, row in enumerate(qubo_matrix):
        weight_lin = sum(row) / 2
        linear_coef.append(-1 * weight_lin)
        offset += weight_lin
        string_temp = []
        for num2, column in enumerate(qubo_matrix.transpose()):
            if num1 == num2:
                string_temp.append('Z')
            else:
                string_temp.append('I')
        strings_linear.append(''.join(string_temp))

    quadratic_coef = []
    strings_quadratic = []
    for n1, row in enumerate(qubo_matrix):
        for n2, r in enumerate(row):
            string_temp_q = ['I' for _ in range(len(row))]
            if r != 0 and n2 > n1:
                string_temp_q[n1] = 'Z'
                string_temp_q[n2] = 'Z'
                weight_quad = r / 2
                quadratic_coef.append(weight_quad)
                offset += -1 * weight_quad
            if 'Z' in string_temp_q:
                strings_quadratic.append(''.join(string_temp_q))

    ising_ham_dict = {}
    for n, string1 in enumerate(strings_linear):
        ising_ham_dict[string1] = linear_coef[n]

    for n, string2 in enumerate(strings_quadratic):
        ising_ham_dict[string2] = quadratic_coef[n]

    # Extract names and weights from the dictionary
    hnames = list(ising_ham_dict.keys())
    hnames = [s[::-1] for s in hnames]
    hweights = list(ising_ham_dict.values())

    return offset


# Example usage
#qubo_matrix = np.array([[31, 500, 500, 500, -500, -500, 0, 0, 0, 0, 0],
 #                       [500, 32, 500, 500, 0, 0, -500, -500, 0, 0, 0],
  #                      [500, 500, 32, 500, 0, 0, 0, 0, -500, -500, 0],
   #                     [500, 500, 500, 32, 0, 0, 0, 0, 0, 0, -500],
   #                     [-500, 0, 0, 0, 536, 500, 0, 0, 0, 0, 0],
   #                     [-500, 0, 0, 0, 500, 538, 0, 0, 0, 0, 0],
   #                     [0, -500, 0, 0, 0, 0, 536, 500, 0, 0, 0],
   #                     [0, -500, 0, 0, 0, 0, 500, 538, 0, 0, 0],
   #                     [0, 0, -500, 0, 0, 0, 0, 0, 536, 500, 0],
   #                     [0, 0, -500, 0, 0, 0, 0, 0, 500, 538, 0],
   #                     [0, 0, 0, -500, 0, 0, 0, 0, 0, 0, 538]])

#hnames, hweights, offset = qubo_to_ising_with_offset(qubo_matrix)
#print("Ising Hamiltonian Dictionary:", ising_ham_dict)
#print("Hamiltonian Names:", hnames)
#print("Hamiltonian Weights:", hweights)
#print("Offset:", offset)
