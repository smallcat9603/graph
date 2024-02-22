import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from scipy.sparse.linalg import svds, eigsh
from sklearn.preprocessing import scale

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

def stationary_distribution(M):
    """
    Stationary distribution given the transition matrix.

    :param M: Transition matrix.
    :return: Stationary distribution.
    """

    # We solve (M^T - I) x = 0 and 1 x = 1. Combine them and let A = [M^T - I; 1], b = [0; 1]. We have A x = b.
    n = M.shape[0]
    A = np.concatenate([M.T - np.identity(n), np.ones(shape=(1,n))], axis=0)
    b = np.concatenate([np.zeros(n), [1]], axis=0)

    # Solve A^T A x = A^T x instead (since A is not square).
    x = np.linalg.solve(A.T @ A, A.T @ b)

    return x

def autocovariance_matrix(M, tau, b=1):
    """
    Autocovariance matrix given a transition matrix. X M^tau/b -x x^T

    :param M: Transition matrix.
    :param tau: Markov time.
    :param b: Number of negative samples used in the sampling algorithm.
    :return: Autocovariance matrix.
    """

    x = stationary_distribution(M)
    X = np.diag(x)
    M_tau = np.linalg.matrix_power(M, tau)

    return X @ M_tau/b - np.outer(x, x) 

def rescale_embeddings(u):
    """
    Rescale the embedding matrix by mean removal and variance scaling.

    :param u: Embeddings.
    :return: Rescaled embeddings.
    """
    shape = u.shape
    scaled = scale(u.flatten())
    return np.reshape(scaled, shape)

def preprocess_similarity_matrix(R):
    """
    Preprocess the similarity matrix.

    :param R: Similarity matrix.
    :return: Preprocessed similarity matrix.
    """

    # R = R.copy()

    # Replace nan with 0 and negative infinity with min value in the matrix.
    R[np.isnan(R)] = 0
    # R[np.isinf(R)] = np.inf
    R[np.isinf(R)] = R.min()

    return R

def postprocess_decomposition(u, s, v=None):
    """
    Postprocess the decomposed vectors and values into final embeddings.

    :param u: Eigenvectors (or left singular vectors)
    :param s: Eigenvalues (or singular values)
    :param v: Right singular vectors.
    :return: Embeddings.
    """

    dim = len(s)

    # Weight the vectors with square root of values.
    for i in range(dim):
        u[:, i] *= np.sqrt(s[i])
        if v is not None:
            v[:, i] *= np.sqrt(s[i])

    # Unify the sign of vectors for reproducible results.
    for i in range(dim):
        if u[0, i] < 0:
            u[:, i] *= -1
            if v is not None:
                v[:, i] *= -1

    # Rescale the embedding matrix.
    if v is not None:
        return rescale_embeddings(u), rescale_embeddings(v)
    else:
        return rescale_embeddings(u)

def standard_random_walk_transition_matrix(G, graph_tool="ig"):
    """
    Transition matrix for the standard random-walk given the input graph.

    :param G: Input graph.
    :return: Standard random-walk transition matrix.
    """

    """
    e.g, 0-1, 0-2, 1-2, 2-3
    degree_vector = np.array(G.degree()) = [2 2 3 1]
    1/degree_vector = [0.5 0.5 0.33 1.]
    """

    if graph_tool == "ig":
        D_1 = np.diag(1/np.array(G.degree()))
        A = np.array(G.get_adjacency().data)
    elif graph_tool == "nx":
        D_1 = np.diag(1/degree_vector(G))
        A = adjacency_matrix(G)
        
    return np.matmul(D_1, A)

def renum_nodes_contiguous(file, file_new):
    node_map = {}
    with open(file, "r") as f:
        edgelist = [line.strip().split() for line in f]
        edgelist = [list(map(int, edge)) for edge in edgelist]
        nodes = set([edge[0] for edge in edgelist] + [edge[1] for edge in edgelist])
        node_map = {node: i for i, node in enumerate(nodes)} 
        edgelist_new = [(node_map[edge[0]], node_map[edge[1]]) for edge in edgelist]
        df = pd.DataFrame(edgelist_new)
        df.to_csv(file_new, sep=" ", index=False, header=False)

graph_tool = "nx"
edgefile = "test.edges"
if graph_tool == "ig":
    renum_nodes_contiguous(edgefile, edgefile) # node id is from 0
    G = ig.Graph.Read_Edgelist(edgefile, directed=False)
    print(G.summary())
elif graph_tool == "nx":
    G = nx.read_edgelist(edgefile, nodetype=int, data=False, create_using=nx.Graph)
    print(G)

M = standard_random_walk_transition_matrix(G, graph_tool)
# print(M)
tau = 3 # markov_time
R = autocovariance_matrix(M, tau)
# print(R)
R = preprocess_similarity_matrix(R)
# print(R)
s, u = eigsh(A=R, k=128, which='LA', maxiter=R.shape[0] * 20)
# print(u)
u = postprocess_decomposition(u, s)
# print(u)

np.savetxt("test1.embeddings", u, fmt='%.16f')
