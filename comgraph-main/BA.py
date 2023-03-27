import networkx as nx
from info_graph import print_info
import math


def write_to_file(g):
    filepath = "graph/barabasi-albert/ba-" + str(g.number_of_nodes()) + \
        "-" + str(g.number_of_edges()) + ".gr"
    f = open(filepath, "w")
    print("\"" + filepath + "\",")
    for e in g.edges():
        f.write(str(e[0]) + " " + str(e[1]) + "\n")


if __name__ == "__main__":
    n = 10000
    for i in range(1, 101):
        g = nx.generators.random_graphs.barabasi_albert_graph(n, i)
        write_to_file(g)
