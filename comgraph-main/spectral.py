from asyncio import subprocess
from cmath import exp
from multiprocessing.sharedctypes import Value
import networkx as nx
from sklearn import cluster
from read_graph import *
import sys
from sklearn.cluster import SpectralClustering
import numpy as np
from draw_tree import *
from read_community import *
from collections import deque
import warnings
import time
warnings.simplefilter('ignore', FutureWarning)


def spectral_clustering(g, objective) -> tuple:
    # Spectral Ordering of the nodes from the normalized motif Laplacian matrix
    if objective == "conductance":
        so = nx.spectral_ordering(g)
        conds = sweep(g, so)
        min_index = conds.index(min(conds))
        return so[:min_index + 1], so[min_index + 1:]
    elif objective == "cheeger":
        so = nx.spectral_ordering(g)
        conds = sweep_cheeger(g, so)
        min_index = conds.index(min(conds))
        return so[:min_index + 1], so[min_index + 1:]
    elif objective == "k-means":
        adj_mat = nx.to_numpy_matrix(g)
        sc = SpectralClustering(2, affinity='precomputed', n_init=100)
        sc.fit(adj_mat)
        inside, outside = [], []
        for i, nd in enumerate(g.nodes()):
            if sc.labels_[i]:
                inside.append(nd)
            else:
                outside.append(nd)
        return inside, outside
    else:
        print("You can't arrive here.")
        exit(0)


def sweep(g: nx.Graph, spectral_ordering: list):
    vol = 0
    total_vol = g.size()
    vol_comp = total_vol
    cut = 0
    seen = set()
    conds = []
    for v in spectral_ordering:
        seen.add(v)
        for nbr in g.neighbors(v):
            if v == nbr:
                continue
            if 'weight' in g[v][nbr]:
                w = g[v][nbr]['weight']
            else:
                w = 1
            if nbr in seen:
                cut -= w
            else:
                cut += w
            vol += w
            vol_comp -= w
        mvol = min(vol, vol_comp)
        if mvol <= 0:
            break
        conds.append(cut / mvol)
    return conds


def sweep_cheeger(g: nx.Graph, spectral_ordering: list):
    vol = 0
    total_vol = g.number_of_nodes()
    vol_comp = total_vol
    cut = 0
    seen = set()
    conds = []
    for v in spectral_ordering:
        seen.add(v)
        for nbr in g.neighbors(v):
            if v == nbr:
                continue
            if 'weight' in g[v][nbr]:
                w = g[v][nbr]['weight']
            else:
                w = 1
            if nbr in seen:
                cut -= w
            else:
                cut += w
        vol += 1
        vol_comp -= 1
        mvol = min(vol, vol_comp)
        if mvol <= 0:
            break
        conds.append(cut / mvol)
    return conds


def cheeger(G: nx.Graph, nodes):
    deg_sum = 0
    for nd in nodes:
        deg_sum += G.degree(nd)
    return nx.conductance(G, nodes) * deg_sum / len(nodes)


def spectral_conductance(G: nx.Graph, g: nx.Graph, cluster_id: int, f, objective: str):
    if cluster_id == 0:
        obj_val = 0
    elif objective == "conductance":
        obj_val = nx.conductance(G, g.nodes())
    elif objective == "cheeger":
        obj_val = cheeger(G, g.nodes())
    elif objective == "k-means":
        obj_val = nx.conductance(G, g.nodes())
    else:
        print("You can't arrive here.")
        exit(0)
    line = f'{cluster_id},{obj_val},{len(g)}\n'
    f.write(line)
    # print(cluster_id, "ongoing")
    if cluster_id > 2 ** 10 - 2 or len(g) <= 1:
        return
    inside, outside = spectral_clustering(g, objective)
    if len(inside) > 0:
        try:
            spectral_conductance(G, g.subgraph(
                inside), cluster_id * 2 + 1, f, objective)
        except ValueError:
            pass
    if len(outside) > 0:
        try:
            spectral_conductance(G, g.subgraph(outside),
                                 cluster_id * 2 + 2, f, objective)
        except ValueError:
            pass


def global_clustering(G: nx.Graph, g: nx.Graph, cluster_id: int, objective: str, threshold: float, clusters: dict):
    if cluster_id == 0:
        obj_val = 0
    elif objective == "conductance":
        obj_val = nx.conductance(G, g.nodes())
    elif objective == "cheeger":
        obj_val = cheeger(G, g.nodes())
    elif objective == "k-means":
        obj_val = nx.conductance(G, g.nodes())
    else:
        print("You can't arrive here.")
        exit(0)
    if obj_val > threshold or len(g) <= 1:
        return -1
    # print(cluster_id, "ongoing")
    inside, outside = spectral_clustering(g, objective)
    if len(inside) > 0:
        try:
            obj_val = global_clustering(G, g.subgraph(
                inside), cluster_id * 2 + 1, objective, threshold, clusters)
            if obj_val < 0:  # if global_clustering() returns - 1
                raise ValueError
        except ValueError:
            clusters[cluster_id] = [nd for nd in g.nodes()]
            return 1
    if len(outside) > 0:
        try:
            obj_val = global_clustering(G, g.subgraph(outside),
                                        cluster_id * 2 + 2, objective, threshold, clusters)
            if obj_val < 0:  # if global_clustering() returns - 1
                raise ValueError
        except ValueError:
            clusters[cluster_id] = [nd for nd in g.nodes()]
            max_id = max(clusters.keys())
            q = deque([2 * cluster_id + 1])
            while q:
                cid = q.popleft()
                if cid > max_id:
                    break
                if cid in clusters:
                    del clusters[cid]
                q.append(2 * cid + 1)
                q.append(2 * cid + 2)
            return 1
    return 1


def exp_spectral(G: nx.Graph, graph_name: str, objective: str = "conductance"):
    objectives = set(["conductance", "cheeger", "k-means"])
    if objective not in objectives:
        print(objective, " is not a valid objective; It must be one of ",
              objectives, sep='')
        return
    f = open("output/" + graph_name + "-" + objective +
             ".csv", "w")
    f.write("cluster_id," + objective + ",size\n")
    spectral_conductance(G, G, 0, f, objective)
    f.close()


def exp_global_clsutering(G: nx.Graph, graph_name: str, objective, threshold: float):
    objectives = set(["conductance", "cheeger", "k-means"])
    if objective not in objectives:
        print(objective, " is not a valid objective; It must be one of ",
              objectives, sep='')
        return
    clusters = dict()
    global_clustering(G, G, 0, objective, threshold, clusters)
    try:
        f = open("cluster/" + graph_name + "/" + objective +
                 "-" + str(threshold) + ".output", "w")
    except FileNotFoundError:
        subprocess.run(["mkdir", "cluster/" + graph_name])
        f = open("cluster/" + graph_name + "/" + objective +
                 "-" + str(threshold) + ".output", "w")
    for cluster_id, nodes in clusters.items():
        line = f'{cluster_id}'
        for nd in nodes:
            line += f' {nd}'
        line += "\n"
        f.write(line)
    f.close()
    cluster_name = "cluster/" + graph_name + "/" + \
        objective + "-" + str(threshold) + ".output"


def draw_tree_conductance(graph_name, objective):
    f = open("output/" + graph_name + "-" + objective + ".csv")
    nodes = dict()
    edges = []
    for line in f.readlines()[1:]:
        cluster_id, conductance, size = line.split(',')
        cluster_id = int(cluster_id)
        conductance = float(conductance)
        size = int(size)
        nodes[cluster_id] = f'{cluster_id}\n{conductance}, {size}'
        if cluster_id > 0 and cluster_id < 31:
            parent = (cluster_id - 1) // 2
            edges.append((nodes[parent], nodes[cluster_id]))
    draw_tree(edges, graph_name, objective)


def compute_global_cluster(g: nx.Graph, threshold: float = 0.4):
    clusters = dict()
    global_clustering(g, g, 0, "k-means", threshold, clusters)
    result = []
    for nodes in clusters.values():
        result.append(set(nodes))
    return result


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3, 4]:
        print("python3", sys.argv[0], "graph_path", "objective", "(threshold)")
        exit()
    gpath = sys.argv[1]
    g = read_graph(gpath)
    graph_name = gpath[6:gpath.rfind(".")]
    if len(sys.argv) == 2:
        objective = "k-means"
    else:
        objective = sys.argv[2]
    if len(sys.argv) in [2, 3]:
        # thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
        #               0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        for threshold in thresholds:
            print("threshold:", threshold, end='')
            start = time.time()
            exp_global_clsutering(g, graph_name, objective, threshold)
            end = time.time()
            print(", time:", end-start)
    elif len(sys.argv) == 4:
        threshold = float(sys.argv[3])
        exp_global_clsutering(g, graph_name, objective, threshold)
    # exp_spectral(g, graph_name, objective)
    # draw_tree_conductance(graph_name, objective)
