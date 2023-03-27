import sys
import networkx as nx
from draw import draw_graph


def sbm(inter_density=0.01, inner_density=0.3, num_communities=3, size=100, seed=0) -> nx.Graph:
    sizes = [size] * num_communities
    probs = []
    for i in range(num_communities):
        prob = []
        for j in range(num_communities):
            if i == j:
                prob.append(inner_density)
            else:
                prob.append(inter_density)
        probs.append(prob)
    return nx.stochastic_block_model(sizes, probs, seed=seed)
