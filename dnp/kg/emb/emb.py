import numpy as np
import networkx as nx

def adjacency_matrix(G):
    """
    Adjacency matrix for the input graph.

    :param G: Input graph.
    :return: Adjacency matrix.
    """
    return nx.adjacency_matrix(G, sorted(G.nodes)).toarray()

def degree_vector(G):
    """
    Degree vector for the input graph.

    :param G: Input graph.
    :return: Degree vector.
    """
    return np.array([a[1] for a in sorted(G.degree(weight='weight'), key=lambda a: a[0])])

def standard_random_walk_transition_matrix(G):
    """
    Transition matrix for the standard random-walk given the input graph.

    :param G: Input graph.
    :return: Standard random-walk transition matrix.
    """

    D_1 = np.diag(1 / degree_vector(G))
    A = adjacency_matrix(G)
    return np.asarray(np.matmul(D_1, A))

G = nx.read_edgelist("test.edges")
M = standard_random_walk_transition_matrix(G)
print(M)

