import networkx as nx


def read_graph(path):
    f = open(path)
    G = nx.Graph()
    for line in f.readlines():
        if "#" in line:
            continue
        values = line.split()
        if len(values) == 1:
            u = int(line)
            G.add_node(u)
        if len(values) == 2:
            u, v = map(int, values)
            if u == v:
                continue
            G.add_edge(u, v)
        if len(values) == 3:
            u, v = int(values[0]), int(values[1])
            w = float(values[2])
            G.add_edge(u, v, weight=w)
    return G


def read_digraph(path):
    f = open(path)
    DiG = nx.DiGraph()
    for line in f.readlines():
        if "#" in line:
            continue
        values = line.split()
        if len(values) == 2:
            u, v = map(int, line.split())
            DiG.add_edge(u, v)
        if len(values) == 3:
            u, v = int(values[0]), int(values[1])
            w = float(values[2])
            DiG.add_edge(u, v, weight=w)
    return DiG
