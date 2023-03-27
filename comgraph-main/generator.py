from distutils.command.upload import upload
import networkx as nx
import itertools

from draw import draw_graph
from image import upload_to_imgbb


def one_chain_graph():
    num_nodes = 1000
    g = nx.Graph()
    for i in range(num_nodes - 1):
        g.add_edge(i, i + 1)
    filename = "tmp/graph.png"
    nx.write_edgelist(g, f"graph/onechain-{num_nodes}.gr", data=False)
    # draw_graph(g, filename=filename)
    # upload_to_imgbb(filename)


def clique_graph():
    clique_size = 30
    num_cliques = 3
    g = nx.Graph()
    # create cliques
    for clique_id in range(num_cliques):
        perms = itertools.permutations(
            range(clique_id * clique_size, (clique_id + 1) * clique_size), 2)
        for u, v in perms:
            g.add_edge(u, v)

    # create bridges
    for clique_id in range(num_cliques - 1):
        g.add_edge((clique_id + 1) * clique_size -
                   1, (clique_id + 1) * clique_size)
    nx.write_edgelist(
        g, f"graph/clique-{clique_size}-{num_cliques}.gr", data=False)
    # filename = "tmp/graph.png"
    # draw_graph(g, filename=filename)
    # upload_to_imgbb(filename)
    pass


if __name__ == "__main__":
    # clique_graph()
    one_chain_graph()
    pass
