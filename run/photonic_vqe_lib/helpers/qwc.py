import networkx as nx
import itertools
import numpy as np
import perceval as pcvl


def PopulateGSA(empty_array, qubit_num, num_groups):
    """find group sizes for hqe grouping"""
    new_group_size_array = empty_array
    for i in range(len(new_group_size_array)):
        new_group_size_array[i] = int(np.floor(qubit_num/num_groups))
    remainder = qubit_num % num_groups
    for i in range(remainder):
        new_group_size_array[i] += 1 
    return new_group_size_array

def CalcModes(group_size_array):
    """calculate the number of modes needed"""
    total_modes = 0 
    for i in group_size_array:
        total_modes += 2**i
    total_modes += 1*2*(len(group_size_array)-1)
    return total_modes

def CZReq(group_size_array):
    return (len(group_size_array)-1)*1
    
def OptimalGroups(modes_available, qubit_num):
    start = 2 if qubit_num > 3 else 1
    for num_groups in range(start, qubit_num//2 + 1):
        #print("Num groups:", num_groups)
        empty_array = np.zeros(num_groups)
        #print("Empty array:", empty_array)
        group_size_array = PopulateGSA(empty_array, qubit_num, num_groups)
        #print("Group Size Array:", group_size_array)
        total_modes = CalcModes(group_size_array)
        #print("Total modes:", total_modes)
        if modes_available >= total_modes:
            # print("Solution with fewest physical entangling gates found (all heralded/unbalanced)")
            # print("Group Size Array:", group_size_array)
            # print("Total Modes Required:", total_modes)
            # print("CZs Required for", 1, "minimal entangling layer(s):", CZReq(group_size_array))
            return group_size_array, total_modes
  # print("No solution found;", modes_available, "is too few modes for", qubit_num, "qubits and", num_layers, "layer(s)")

def generate_state_vectors(group_sizes):
    """This block generates the basic state dictionary for any group size array to allow for post selection and
    output dictionary mapping afterwards"""
    state_strings = {}
    hstate_strings = {}
    
    # Generate binary strings for each group size
    binary_strings = [[''.join(map(str, t)) for t in itertools.product([0, 1], repeat=size)] for size in group_sizes]
    
    for combination in itertools.product(*binary_strings):
        key = ''.join(combination)
        
        vector = [0] * (sum([2**size for size in group_sizes]) + 2*(len(group_sizes) - 1))
        hvector = [0] * (sum([2**size for size in group_sizes])) + [1]*2*(len(group_sizes) - 1)
        
        offset = 0
        for idx, size_bits in enumerate(combination):
            vector[offset + int(size_bits, 2)] = 1
            hvector[offset + int(size_bits, 2)] = 1
            offset += 2**group_sizes[idx]
        
        state_strings[key] = pcvl.BasicState(vector)
        hstate_strings[key] = pcvl.BasicState(hvector)
    
    return state_strings, hstate_strings





class QWCGrouping:
    def __init__(self, method, hnames=None, hweights=None):
        """
        Initialize the QWCGrouping class.

        :param method: Coloring method to use (either 'greedy' or 'lf' for largest first).
        :param hnames: List of Hamiltonian term names (strings like 'XIZ' representing operators on qubits).
        :param hweights: List of Hamiltonian term weights (coefficients for each term).
        """
        self.method = method
        self.Hamiltonian = self.construct_Hamiltonian(hnames, hweights)
        self.num_terms = len(self.Hamiltonian)
        self.num_qubits = len(self.Hamiltonian[0][1])
        self.adjacency_list = None
        self.coloring = None

    def construct_Hamiltonian(self, hnames, hweights):
        """
        Construct the Hamiltonian from given names and weights.
        
        :param hnames: List of Hamiltonian term names.
        :param hweights: List of Hamiltonian term weights.
        :return: List of tuples pairing weights with their respective term names.
        """
        if not hnames or not hweights:
            raise ValueError("hnames and hweights must be provided.")
        return list(zip(hweights, hnames))

    def construct_complement_qwc_graph(self):
        """
        Construct the complement of the QWC graph using the Hamiltonian terms.
        This graph will have an edge between terms that are NOT qubit-wise commuting.
        """
        self.adjacency_list = {i: [] for i in range(self.num_terms)}
        for i in range(self.num_terms):
            for j in range(i + 1, self.num_terms):
                if not self.is_qwc(self.Hamiltonian[i][1], self.Hamiltonian[j][1]):
                    self.adjacency_list[i].append(j)
                    self.adjacency_list[j].append(i)

    def convert_adjlist_to_graph(self):
        """
        Convert the adjacency list to a NetworkX graph representation.
        
        :return: NetworkX graph representation.
        """
        G = nx.Graph()
        for node, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return G

    def greedy_coloring(self):
        """
        Color the graph using the greedy coloring strategy.
        """
        G = self.convert_adjlist_to_graph()
        self.coloring = nx.coloring.greedy_color(G)

    def largest_first_coloring(self):
        """
        Color the graph using the largest-first coloring strategy.
        """
        G = self.convert_adjlist_to_graph()
        self.coloring = nx.coloring.greedy_color(G, strategy='largest_first')

    def color_graph(self):
        """
        Color the graph using the specified method.
        """
        if self.method == 'greedy':
            self.greedy_coloring()
        elif self.method == 'lf':
            self.largest_first_coloring()
        else:
            raise ValueError(f'Invalid method: {self.method}. Valid methods are greedy, lf.')

    def create_qwc_groups(self):
        """
        Create groups of qubit-wise commuting terms.
        
        :return: Dictionary with color as key and list of qubit-wise commuting terms as value.
        """
        self.construct_complement_qwc_graph()
        self.color_graph()
        qwc_groups = {color: [] for color in self.coloring.values()}
        for node, color in self.coloring.items():
            qwc_groups[color].append(self.Hamiltonian[node])
        return qwc_groups
    
    def all_terms_are_qwc(self):
        """
        Check if all terms in the Hamiltonian are qubit-wise commuting with each other.

        :return: True if all terms are qubit-wise commuting, False otherwise.
        """
        for i in range(self.num_terms):
            for j in range(i + 1, self.num_terms):
                if not self.is_qwc(self.Hamiltonian[i][1], self.Hamiltonian[j][1]):
                    return False
        return True


    
    def create_qwc_groups(self):
        """
        Create groups of qubit-wise commuting terms.

        :return: Dictionary with color as key and list of qubit-wise commuting terms as value.
        """
        if self.all_terms_are_qwc():
            # If all terms are QWC, return them all in a single group
            return {0: self.Hamiltonian}

        self.construct_complement_qwc_graph()
        self.color_graph()
        qwc_groups = {color: [] for color in self.coloring.values()}
        for node, color in self.coloring.items():
            qwc_groups[color].append(self.Hamiltonian[node])
        return qwc_groups


    def is_qwc(self, term1, term2):
        """
        Check if two terms are qubit-wise commuting.
        
        :param term1: First term to check.
        :param term2: Second term to check.
        :return: True if the terms are qubit-wise commuting, False otherwise.
        """
        for p1, p2 in zip(term1, term2):
            if p1 != 'I' and p2 != 'I' and p1 != p2:
                return False
        return True
    
    
    
def rotate_measurements_2(num_qubits, pauli_string: str):
    """
    Determines which qubits should be measured based on the given Pauli string.
    The qubit indexing is counted backwards, following Qiskit's convention.

    :param num_qubits: Total number of qubits in the quantum system.
    :param pauli_string: String of Pauli operators (e.g., 'XIYZ').
    :return: List of indices of qubits that should be measured.
    """
    
    measured_qubits = []

    # Iterate through each Pauli operator in the string
    for i, pauli_op in enumerate(pauli_string):
        
        # If the Pauli operator is 'X', 'Y', or 'Z', the corresponding qubit should be measured.
        # Qiskit's indexing convention is used, which counts qubits from right to left.
        if pauli_op in ['X', 'Y', 'Z']:
            measured_qubits.append(num_qubits - 1 - i)

    return measured_qubits












def extract_measurement_bases(group_data):
    """
    Extract the term with the fewest 'I' characters from the provided group data.
    If all terms are QWC with an all 'Z' string, then return the all 'Z' string.

    Parameters:
    - group_data (list of tuples): List containing tuples where the second element is a string of operators.

    Returns:
    - str: Measurement base string.
    """
    num_qubits = len(group_data[0][1])
    all_Zs = 'Z' * num_qubits
    
    # Check if all terms are QWC with all_Zs
    if all(is_qwc(term[1], all_Zs) for term in group_data):
        return all_Zs

    return min(group_data, key=lambda x: x[1].count('I'))[1]



def extract_measurement_bases(group_data):
    """
    Extract a suitable measurement base from the provided group data.

    Parameters:
    - group_data (list of tuples): List containing tuples where the second element is a string of operators.

    Returns:
    - str: Measurement base string.
    """
    num_qubits = len(group_data[0][1])
    measurement_base = ''

    for i in range(num_qubits):
        # Collect all operators at position i
        operators_at_i = [term[1][i] for term in group_data]

        # Check for a common non-identity operator
        non_identity_operators = set(operators_at_i) - {'I'}
        if len(non_identity_operators) == 1:
            # There is a single non-identity operator common to all terms
            measurement_base += non_identity_operators.pop()
        else:
            # Default to 'Z' if there's no common non-identity operator or multiple different ones
            measurement_base += 'Z'

    return measurement_base



def is_qwc(term1, term2):
    """
    Check if two terms are qubit-wise commuting.
    
    :param term1: First term to check.
    :param term2: Second term to check.
    :return: True if the terms are qubit-wise commuting, False otherwise.
    """
    for p1, p2 in zip(term1, term2):
        if p1 != 'I' and p2 != 'I' and p1 != p2:
            return False
    return True


