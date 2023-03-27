from collections import defaultdict
import networkx as nx

DATASET = "aucs"
# DATASET = "Airports"
# DATASET = "dkpol"
# DATASET = "Rattus"


f = open(f"generator/original/{DATASET}.mpx")

special_layers = {
    "aucs": ("lunch", "facebook", "leisure", "work", "coauthor"),
    "Airports": ("Lufthansa", "Ryanair", "Scandinavian_Airlines", "Wideroe", "Air_Lingus", "TAP_Portugal", "LOT_Polish_Airlines", "Transavia_Holland", "Germanwings", "Alitalia", "Flybe", "Vueling_Airlines", "Swiss_International_Air_Lines", "Niki", "SunExpress", "Air_Berlin", "Air_Baltic", "TNT_Airways", "KLM", "Austrian_Airlines", "Netjets", "Iberia", "Olympic_Air", "European_Air_Transport", "Czech_Airlines", "Easyjet", "Panagra_Airways", "Aegean_Airlines", "Norwegian_Air_Shuttle", "Air_France", "Wizz_Air", "British_Airways", "Turkish_Airlines", "Brussels_Airlines", "Malev_Hungarian_Airlines", "Finnair", "Air_Nostrum"),
    "dkpol": ("ff", "Re", "RT"),
    "Rattus": ("PA", "DI", 'colocalization', 'association', 'additive_genetic_interaction_defined_by_inequality', 'suppressive_genetic_interaction_defined_by_inequality'),
}

nodes = dict()
edges = set()
nd_id = 0

gs = [nx.Graph() for _ in special_layers[DATASET]]
layer2id = {layer: i for i,
            layer in enumerate(special_layers[DATASET])}
s = set()
for line in f.readlines():
    u, v, layer = line.replace("\n", "").split(',')
    if u not in nodes:
        nodes[u] = nd_id
        nd_id += 1
    u = nodes[u]
    if v not in nodes:
        nodes[v] = nd_id
        nd_id += 1
    v = nodes[v]

    edges.add((min(u, v), max(u, v)))
    try:
        gs[layer2id[layer]].add_edge(u, v)
    except KeyError:
        s.add(layer)
for layer in s:
    print(f"\"{layer}\"", end=", ")

for i, layer in enumerate(special_layers[DATASET]):
    nx.write_edgelist(
        gs[i], f"graph/{DATASET}-{layer}.gr", data=False)
