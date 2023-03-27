from collections import defaultdict
import networkx as nx


DATASET = "aucs"
# DATASET = "Airports"
# DATASET = "dkpol"
# DATASET = "Rattus"


f = open(f"generator/original/{DATASET}.mpx")

nodes = defaultdict(dict)
edges = defaultdict(set)
nd_id = 0

ga = nx.Graph()
gb = nx.Graph()

for line in f.readlines():
    u, v, layer = line.replace("\n", "").split(',')
    if u not in nodes[layer]:
        nodes[layer][u] = nd_id
        nd_id += 1
    u = nodes[layer][u]
    if v not in nodes[layer]:
        nodes[layer][v] = nd_id
        nd_id += 1
    v = nodes[layer][v]

    edges[layer].add((min(u, v), max(u, v)))

for layer, nds in nodes.items():
    print(layer, len(nds), len(edges[layer]))
