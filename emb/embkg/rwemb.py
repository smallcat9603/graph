"""Random-walk based node embedding (RWEM), without any UI dependency.

Pipeline: adjacency A -> transition matrix M (= D^-1 A) -> optional label mixing
-> stationary distribution x -> similarity R (autocovariance or PMI at Markov
time tau) -> top-`dim` eigenvectors of R -> rescaled embedding.

Ported unchanged from dnp/kg/pages/lib/flow.py (the code behind the ADMA 2025
and Innovation Discovery 2026 results); tests/test_rwemb_regression.py checks
that the outputs match.
"""

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import scale


def standard_random_walk_transition_matrix(G, graph_tool="igraph"):
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

    if graph_tool == "igraph":
        degree_vector = np.array(G.degree())
        D_1 = np.diag(1/degree_vector)
        A = np.array(G.get_adjacency().data)
    elif graph_tool == "networkx":
        degree_vector = np.array([a[1] for a in sorted(G.degree(weight='weight'), key=lambda a: a[0])])
        D_1 = np.diag(1/degree_vector)
        A = nx.adjacency_matrix(G, sorted(G.nodes)).toarray()

    return np.matmul(D_1, A)


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

    # Solve A^T A x = A^T b instead (since A is not square).
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


def PMI_matrix(M, tau, b=1):
    """
    PMI matrix given a transition matrix. log(X M^tau/b) - log(x x^T)

    :param M: transition matrix
    :param tau: Markov time
    :param b: Number of negative samples used in the sampling algorithm.
    :return: PMI matrix.
    """

    x = stationary_distribution(M)
    X = np.diag(x)
    M_tau = np.linalg.matrix_power(M, tau)

    return np.log(X @ M_tau/b) - np.log(np.outer(x, x))


def preprocess_similarity_matrix(R):
    """
    Preprocess the similarity matrix.

    :param R: Similarity matrix.
    :return: Preprocessed similarity matrix.
    """

    # Replace nan with 0 and +-inf with the min finite value in the matrix
    # (-inf is first turned into +inf so that R.min() is finite).
    R[np.isnan(R)] = 0
    R[np.isinf(R)] = np.inf
    R[np.isinf(R)] = R.min()

    return R


def rescale_embeddings(u):
    """
    Rescale the embedding matrix by mean removal and variance scaling.

    :param u: Embeddings.
    :return: Rescaled embeddings.
    """
    shape = u.shape
    scaled = scale(u.flatten())
    return np.reshape(scaled, shape)


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


def get_node_labels(df):
    """Map node -> list of labels from a two-column (node, label) DataFrame."""
    df.columns = ["node", "label"]
    node_labels = {}
    df_rows = df.shape[0]
    for i in range(df_rows):
        node = df["node"][i]
        label = df["label"][i]
        if node in node_labels:
            node_labels[node].append(label)
        else:
            node_labels[node] = [label]
    return node_labels


def get_category_list(node_labels):
    """First label in 1..9 of each node 0..n-1, or 0 if none."""
    category = []
    n = len(node_labels)
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(n):
        for c in categories:
            if c in node_labels[i]:
                category.append(c)
                break
        else:
            category.append(0)
    return category


def transition_matrix_node_label_node(n, node_labels):
    """Row-normalized node-label-node transition matrix (shared-label counts)."""
    arr_nln = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            common_count = len(set(node_labels[i]) & set(node_labels[j]))
            if common_count > 0:
                arr_nln[i, j] = common_count
                arr_nln[j, i] = common_count
    arr_sum = np.sum(arr_nln, axis=1)
    for i in range(n):
        if arr_sum[i] > 0:
            arr_nln[i] /= arr_sum[i]
    return arr_nln, arr_sum


def M_attr(n, M, node_labels, node_labels2, alpha, beta):
    """Mix the topology transition matrix with one or two label transition matrices.

    alpha: weight of topology; beta: split of the label share between label 1 and label 2.
    Modifies M in place and returns it.
    """
    arr_nln, arr_sum = transition_matrix_node_label_node(n, node_labels)
    if node_labels2 is None:
        for i in range(n):
            if arr_sum[i] > 0:
                M[i] = M[i]*alpha + arr_nln[i]*(1-alpha)
    else:
        arr_nln2, arr_sum2 = transition_matrix_node_label_node(n, node_labels2)
        for i in range(n):
            if arr_sum[i] > 0 and arr_sum2[i] == 0:
                M[i] = M[i]*alpha + arr_nln[i]*(1-alpha)
            elif arr_sum[i] == 0 and arr_sum2[i] > 0:
                M[i] = M[i]*alpha + arr_nln2[i]*(1-alpha)
            elif arr_sum[i] > 0 and arr_sum2[i] > 0:
                M[i] = M[i]*alpha + arr_nln[i]*(1-alpha)*beta + arr_nln2[i]*(1-alpha)*(1-beta)
    return M


def node_emb(G, sim, tau, dim, graph_tool, df_label=None, df_label2=None, alpha=1.0, beta=1.0, on_step=None):
    """Embed the nodes of G.

    :param sim: "Autocovariance" or "PMI".
    :param df_label, df_label2: optional (node, label) DataFrames for label mixing.
    :param on_step: optional callback(title, matrix) to inspect intermediate matrices.
    :return: (u, category) — embedding matrix (n x dim) and per-node category list.
    """
    def step(title, mat):
        if on_step is not None:
            on_step(title, mat)

    M = standard_random_walk_transition_matrix(G, graph_tool=graph_tool)
    step("Transition Matrix", M)

    n = M.shape[0]
    category = [0]*n
    if df_label is not None:
        node_labels = get_node_labels(df_label)
        category = get_category_list(node_labels)

        node_labels2 = None
        if df_label2 is not None:
            node_labels2 = get_node_labels(df_label2)
        M = M_attr(n=n, M=M, node_labels=node_labels, node_labels2=node_labels2, alpha=alpha, beta=beta)
        step("Transition Matrix with Attributes", M)

    if sim == "Autocovariance":
        R = autocovariance_matrix(M, tau)
    elif sim == "PMI":
        R = PMI_matrix(M, tau)
    step(f"{sim} Matrix", R)

    R = preprocess_similarity_matrix(R)
    step(f"{sim} Matrix (clean)", R)

    s, u = eigsh(A=R, k=dim, which='LA', maxiter=R.shape[0] * 20)
    step("Eigenvectors", u)

    u = postprocess_decomposition(u, s)
    step("Embedding Matrix", u)

    return u, category
