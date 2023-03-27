from read_graph import *
from read_community import *
from draw import *
from motif_type import *
import time
import random
from charts import *
import random
import networkx as nx
import os
import numpy as np
import pandas as pd
from image import *
import sys
from appr import *
from dappr import DAPPR
from mllcd import MLLCD
from sbm import *
from notification import *
from similarity import *
from eta import ETA
from typing import Set, Dict
import statistics
from collections import defaultdict, deque
from scipy.stats.mstats import gmean
from utility import *
from info_graph import *
from sheets import *
from scipy.stats import pearsonr as correl
import copy
import metis
INF = float('inf')
LARGE = 10000


def merge_two_graphs(ga: nx.Graph, gb: nx.Graph, data: bool = True):
    g = nx.Graph()
    g.add_nodes_from(ga.nodes(data=data))
    g.add_nodes_from(gb.nodes(data=data))
    if data:
        for u, v, d in ga.edges(data=True):
            if 'weight' in d:
                g.add_edge(u, v, weight=d['weight'])
            else:
                g.add_edge(u, v, weight=1)
        for u, v, d in gb.edges(data=True):
            if g.has_edge(u, v):
                if 'weight' in g.get_edge_data(u, v) and 'weight' in d:
                    g[u][v]['weight'] += d['weight']
                elif 'weight' in g.get_edge_data(u, v):
                    g[u][v]['weight'] += 1
                elif 'weight' in d:
                    g[u][v]['weight'] = 1 + d['weight']
                else:
                    g[u][v]['weight'] = 2
            else:
                if 'weight' in d:
                    g.add_edge(u, v, weight=d['weight'])
                else:
                    g.add_edge(u, v, weight=1)
    else:
        g.add_edges_from(ga.edges())
        g.add_edges_from(gb.edges())
    return g


def merge_two_graphs_with_supernode(ga: nx.Graph, gb: nx.Graph, r: float = 0.5, gamma: float = 1):
    if r < 0 or 1 < r:
        print(f"r must be in [0, 1] but r  = {r}", file=sys.stderr)
        exit(0)
    ndsa = set([nd for nd in ga])
    ndsb = set([nd for nd in gb])
    nm = len(ndsa.union(ndsb))
    ma = ga.number_of_edges()
    mb = gb.number_of_edges()
    wa = r * (ma + mb) / (r * ma + (1 - r) * mb + gamma * nm)
    wb = (1 - r) * (ma + mb) / (r * ma + (1 - r) * mb + gamma * nm)
    nga = ga.__class__()
    ngb = gb.__class__()
    nga.add_nodes_from(ga.nodes())
    ngb.add_nodes_from(gb.nodes())
    for u, v in ga.edges():
        nga.add_edge(u, v, weight=wa)
    for u, v in gb.edges():
        ngb.add_edge(u, v, weight=wb)
    g = nx.Graph()
    g.add_nodes_from(nga.nodes(data=True))
    g.add_nodes_from(ngb.nodes(data=True))
    for u, v, d in nga.edges(data=True):
        if 'weight' in d:
            g.add_edge(u, v, weight=d['weight'])
        else:
            g.add_edge(u, v, weight=1)
    for u, v, d in ngb.edges(data=True):
        if g.has_edge(u, v):
            if 'weight' in g.get_edge_data(u, v) and 'weight' in d:
                g[u][v]['weight'] += d['weight']
            elif 'weight' in g.get_edge_data(u, v):
                g[u][v]['weight'] += 1
            elif 'weight' in d:
                g[u][v]['weight'] = 1 + d['weight']
            else:
                g[u][v]['weight'] = 2
        else:
            if 'weight' in d:
                g.add_edge(u, v, weight=d['weight'])
            else:
                g.add_edge(u, v, weight=1)
    original_nodes = [nd for nd in g]
    for nd in original_nodes:
        g.add_edge(nd, SUPERNODE_ID, weight=(wa + wb) * gamma)
    return g


def normalize_edge_weight(
    ga: nx.Graph,
    gb: nx.Graph,
    r: float,
):
    if r < 0 or 1 < r:
        print(f"r must be in [0, 1] but r  = {r}", file=sys.stderr)
        exit(0)
    ma = ga.number_of_edges()
    mb = gb.number_of_edges()
    wa = r * (ma + mb) / (r * ma + (1 - r) * mb)
    wb = (1 - r) * (ma + mb) / (r * ma + (1 - r) * mb)
    nga = ga.__class__()
    ngb = gb.__class__()
    nga.add_nodes_from(ga.nodes())
    ngb.add_nodes_from(gb.nodes())
    for u, v in ga.edges():
        nga.add_edge(u, v, weight=wa)
    for u, v in gb.edges():
        ngb.add_edge(u, v, weight=wb)
    return nga, ngb


def randomize_graph_id(orig: nx.Graph):
    g = nx.Graph()
    g.add_nodes_from(orig.nodes(data=True))
    ids1 = [nd for nd in orig]
    ids2 = [nd for nd in orig]
    random.shuffle(ids1)
    random.shuffle(ids2)
    renumbered = {ids1[i]: ids2[i] for i, _ in enumerate(ids1)}
    for u, v, d in orig.edges(data=True):
        g.add_edge(renumbered[u], renumbered[v])
        g[renumbered[u]][renumbered[v]].update(d)
    return g


def relabel_all_nodes(g: nx.Graph, addition: int, excepted: Set[int] = set()) -> nx.Graph:
    mapping = dict()
    for nd in g.nodes():
        if nd not in excepted:
            mapping[nd] = nd + addition
    return nx.relabel_nodes(g, mapping)


def get_gname(file_path):
    if file_path[:5] == "graph":
        return file_path[6:file_path.rfind(".")]
    elif file_path[:15] == "processed_graph":
        return file_path[16:file_path.rfind(".")]
    else:
        print("file_path error")
        exit(0)


def compute_appr(path, degree_normalized=True):
    g = read_graph(path)
    graph_name = get_gname(path)
    try:
        dir = "cluster/" + graph_name
        print(dir)
        os.makedirs(dir)
    except FileExistsError:
        pass
    if degree_normalized:
        f = open("cluster/" + graph_name + "/appr.txt", "w")
        f_top = open("cluster/" + graph_name + "/appr-top.txt", "w")
        f_cond = open("cluster/" + graph_name +
                      "/appr-conductance.txt", "w")
    else:
        f = open("cluster/" + graph_name + "/appr-ndn.txt", "w")
        f_top = open("cluster/" + graph_name + "/appr-top-ndn.txt", "w")
        f_cond = open("cluster/" + graph_name +
                      "/appr-conductance-ndn.txt", "w")
    appr = APPR(g)
    eta = ETA()
    percent = len(g) / 100
    percent_done = 1
    for cnt, seed in enumerate(g, 1):
        cluster = appr.compute_appr(seed, degree_normalized=degree_normalized)
        cluster.sort()
        line = f'{seed}'
        for nd in cluster:
            line += f' {nd}'
        line += '\n'
        f.write(line)

        top = appr.get_node_in_order()
        line = f'{seed}'
        for nd in top:
            line += f' {nd}'
        line += '\n'
        f_top.write(line)

        conds = appr.get_cond_profile()
        line = f'{seed}'
        for cond in conds:
            line += f' {cond}'
        line += '\n'
        f_cond.write(line)

        if len(top) != len(conds):
            print("problem")

        if cnt / percent >= percent_done:
            print(
                f'Completed: {percent_done}%',
                f'ETA: {round(eta.eta(cnt / len(g)), 1)}s',
                sep=', '
            )
            percent_done += 1
    f.close()
    f_top.close()
    f_cond.close()


def read_clusters(
    filepath: str,
    return_type=set,
    val_type=int,
) -> Dict[int, Set[int]]:
    f = open(filepath)
    clusters = {}
    for line in f.readlines():
        nodes = list(map(val_type, line.split()))
        seed = nodes[0]
        cluster = nodes[1:]
        if return_type == set:
            clusters[seed] = set(cluster)
        elif return_type == list:
            clusters[seed] = cluster
    f.close()
    return clusters


def create_n_hop_graph(g: nx.Graph, seed: int, n: int, include_boundary_edges=False) -> nx.Graph:
    q = deque([(seed, 0)])
    seen = set([seed])
    dists = defaultdict(set)
    dists[0].add(seed)
    while q:
        nd, dist = q.popleft()
        if not include_boundary_edges and dist >= n - 1:
            break
        elif dist >= n:
            break
        for nbr in g.neighbors(nd):
            if nbr not in seen:
                seen.add(nbr)
                dists[dist + 1].add(nbr)
                q.append((nbr, dist + 1))
    subgraph = g.subgraph(seen).copy()
    if include_boundary_edges:
        return subgraph
    far = max(dists.keys())
    for nd in dists[far]:
        subgraph.add_edges_from(g.edges(nd, data=True))
    return subgraph


def compute_q1_med_q3(l: List):
    return np.percentile(l, 25), np.percentile(l, 50), np.percentile(l, 75)


def exp_ppr_distribution():
    USE_SBM = True
    COMPUTE_BY_APPR = True

    if USE_SBM:
        id_a, id_b = "0.2", "0.4"
        ga = read_graph(f"tmp/graph/" + id_a + "-0.01-3-100-normalorder.gr")
        gb = read_graph(f"tmp/graph/" + id_b + "-0.01-3-100-normalorder.gr")
        SEED = 0
    else:
        id_a, id_b = "Email-Enron", "CA-GrQc"
        ga = read_graph("graph/" + id_a + ".txt")
        gb = read_graph("graph/" + id_b + ".txt")
        SEED = 0

    appra = APPR(ga)
    apprb = APPR(gb)
    RANKING_THRESHOLD = 300
    appra.compute_appr(SEED)
    apprb.compute_appr(SEED)

    if COMPUTE_BY_APPR:
        appr_vec_a = list(appra.get_appr_vec().values())
        appr_vec_b = list(apprb.get_appr_vec().values())
    else:
        appr_vec_a = list(nx.pagerank(ga, personalization={
            SEED: 1}, alpha=0.98, tol=0.0001).values())
        appr_vec_b = list(nx.pagerank(gb, personalization={
            SEED: 1}, alpha=0.98, tol=0.0001).values())

    appr_vec_a += [0] * (len(ga) - len(appr_vec_a))
    appr_vec_b += [0] * (len(gb) - len(appr_vec_b))
    appr_vec_a.sort(reverse=True)
    appr_vec_b.sort(reverse=True)
    appr_vec_a = appr_vec_a[1:RANKING_THRESHOLD]
    appr_vec_b = appr_vec_b[1:RANKING_THRESHOLD]
    rankings = [i + 2 for i in range(1, RANKING_THRESHOLD)]
    draw_chart(
        rankings,
        [appr_vec_a, appr_vec_b],
        labels=[id_a, id_b],
        left=0,
    )


def exp_unique_frequency():
    dataset = "SBM"
    # dataset = "REALWORLD"
    # dataset = "CLIQUE-CHAIN"
    wa, wb = 1, 1

    if dataset == "SBM":
        id_a, id_b = "0.1", "0.3"
        ga = read_graph(f"tmp/graph/" + id_a + "-0.01-3-100-normalorder.gr")
        gb = read_graph(f"tmp/graph/" + id_b + "-0.01-3-100-normalorder.gr")
        SEED = 0
    elif dataset == "REALWORLD":
        id_a, id_b = "Email-Enron", "CA-GrQc"
        ga = read_graph("graph/" + id_a + ".txt")
        gb = read_graph("graph/" + id_b + ".txt")
        SEED = 13
    elif dataset == "CLIQUE-CHAIN":
        id_a, id_b = "clique", "chain"
        ga = read_graph(f"graph/{id_a}-50-3.gr")
        gb = read_graph(f"graph/one{id_b}-1000.gr")
        SEED = 0
        gb = relabel_all_nodes(
            gb, addition=ga.number_of_nodes(), excepted=set([SEED]))
    nga, ngb = normalize_edge_weight(ga, gb, wb/wa)
    gm = merge_two_graphs(nga, ngb, data=True)
    # gm = merge_two_graphs(ga, gb, data=False)
    appra = APPR(ga)
    apprb = APPR(gb)
    apprm = APPR(gm)
    ca = set(appra.compute_appr(SEED))
    cb = set(apprb.compute_appr(SEED))
    print(len(ca), len(cb))
    cm = apprm.compute_appr(SEED)
    # apprm.compute_allocation_appr(SEED, ga, gb)
    vm = apprm.get_appr_vec()
    for nd in gm:
        if nd not in vm:
            vm[nd] = 0
    ndppr = [(k, v) for k, v in vm.items()]
    ndppr.sort(key=lambda x: x[1], reverse=True)
    a_unique, mutual, b_unique, unrelated, ranking = 0, 0, 0, 0, 1
    a_uniques, mutuals, b_uniques, unrelateds, rankings = [], [], [], [], []
    for nd, _ in ndppr[:min(len(cm) * 2, len(ndppr))]:
        if nd in ca and nd in cb:
            mutual += 1
        elif nd in ca:
            a_unique += 1
        elif nd in cb:
            b_unique += 1
        else:
            unrelated += 1
        ranking += 1
        a_uniques.append(a_unique / len(gm))
        mutuals.append(mutual / len(gm))
        b_uniques.append(b_unique / len(gm))
        unrelateds.append(unrelated / len(gm))
        rankings.append(ranking)
    filename = "tmp/yeah.png"
    draw_chart(
        rankings,
        [mutuals, a_uniques, b_uniques, unrelateds],
        title=f"frequency of appearance in top k ranking\ngraph: (A: {id_a}, B: {id_b}), weight: (A: {wa}, B: {wb})",
        labels=["mutuals", "A_uniques",  "B_uniques", "unrelateds"],
        x_axis_title="k",
        y_axis_title="frequency",
        left=0,
        bottom=0,
        filename=filename
    )
    upload_to_imgbb(filename)


def exp_uiu_rwer_comparison():
    dataset = "SBM"
    # dataset = "REALWORLD"

    # strategy = "NWF"
    strategy = "dynamic-rw"
    # strategy = "unique-rw"

    COMPUTE_THE_VALUES = True
    transition_probability_ratio = 2
    # transition_probability_ratios = [1, 1/4, 1/2, 2, 4]

    if dataset == "SBM":
        id_a, id_b = "0.3", "0.1"
        ga = read_graph(f"tmp/graph/" + id_a + "-0.01-3-100-normalorder.gr")
        gb = read_graph(f"tmp/graph/" + id_b + "-0.01-3-100-normalorder.gr")
        RANKING_THRESHOLD = 300
    elif dataset == "REALWORLD":
        id_a, id_b = "Email-Enron", "CA-GrQc"
        ga = read_graph("graph/" + id_a + ".txt")
        gb = read_graph("graph/" + id_b + ".txt")
        RANKING_THRESHOLD = 500

    filename_separate = f"tmp/separate-{id_a}-{id_b}-{strategy}-{transition_probability_ratio}-rweras-uias-rwerbs-uibs.csv"
    filename_rwers = f"tmp/merged-{id_a}-{id_b}-{strategy}-{transition_probability_ratio}.csv"
    filename_ranking = f"tmp/rankings-{id_a}-{id_b}-{strategy}-{transition_probability_ratio}-rankings-mutuals-a_uniques-b_uniques-unrelateds.csv"

    if COMPUTE_THE_VALUES:
        appra = APPR(ga)
        apprb = APPR(gb)
        if strategy == "NWF":
            gm = merge_two_graphs(ga, gb, data=True)
        elif strategy == "dynamic-rw":
            gm = merge_two_graphs(ga, gb, data=False)
        elif strategy == "dynamic-rw":
            gm = merge_two_graphs(ga, gb, data=False)
        elif strategy == "unique-rw":
            gm = merge_two_graphs(ga, gb, data=False)

        apprm = APPR(gm)
        nodes_in_common = list(
            set([nd for nd in ga]).intersection([nd for nd in gb]))
        rwers = []
        uis = []
        rweras, rwerbs = [], []
        uias, uibs = [], []
        eta = ETA()
        seeds = random.sample(nodes_in_common, 10)
        # seeds = nodes_in_common
        a_uniques, mutuals, b_uniques, unrelateds, rankings = [
            0] * RANKING_THRESHOLD, [0] * RANKING_THRESHOLD, [0] * RANKING_THRESHOLD, [0] * RANKING_THRESHOLD, [0] * RANKING_THRESHOLD
        for i, seed in enumerate(seeds):
            ca = set(appra.compute_appr(seed))
            cb = set(apprb.compute_appr(seed))
            if strategy == "NWF":
                cm, rwera, rwerb = apprm.compute_appr_data(seed, ga, gb)
            elif strategy == "dynamic-rw":
                cm, rwera, rwerb, _, _ = apprm.compute_dynamic_appr(
                    seed, ga, gb, c=1, data=True)
            elif strategy == "backward-rw":
                cm, rwera, rwerb = apprm.compute_backward_appr(
                    seed, ga, gb, ca, cb, c=1, data=True)
            ui_a = unique_influence(ca, cb, cm)
            ui_b = unique_influence(cb, ca, cm)
            rwers.append(rwera - rwerb)
            rweras.append(rwera)
            rwerbs.append(rwerb)
            uias.append(ui_a)
            uibs.append(ui_b)
            uis.append(ui_a - ui_b)
            vm = apprm.get_appr_vec()
            for nd in gm:
                if nd not in vm:
                    vm[nd] = 0
            ndppr = [(k, v) for k, v in vm.items()]
            ndppr.sort(key=lambda x: x[1], reverse=True)
            if len(ndppr) < RANKING_THRESHOLD:
                for _ in range(RANKING_THRESHOLD - len(ndppr)):
                    ndppr.append((0, 0))
            a_unique, mutual, b_unique, unrelated, ranking = 0, 0, 0, 0, 1
            for j, (nd, _) in enumerate(ndppr[:RANKING_THRESHOLD]):
                if nd in ca and nd in cb:
                    mutual += 1
                elif nd in ca:
                    a_unique += 1
                elif nd in cb:
                    b_unique += 1
                else:
                    unrelated += 1
                ranking += 1
                a_uniques[j] += a_unique / len(gm)
                mutuals[j] += mutual / len(gm)
                b_uniques[j] += b_unique / len(gm)
                unrelateds[j] += unrelated / len(gm)
                rankings[j] += ranking
            print(
                "eta:",
                eta.eta((i+1) / len(seeds))
            )

        f = open(filename_separate, 'w')
        f.write(f"rweras,uias,rwerbs,uibs\n")
        for i in range(len(rweras)):
            f.write(f"{rweras[i]},{uias[i]},{rwerbs[i]},{uibs[i]}\n")
        f.close()

        f = open(filename_rwers, 'w')
        f.write(f"rwers,uis\n")
        for i in range(len(rwers)):
            f.write(f"{rwers[i]},{uis[i]}\n")
        f.close()

        for freqs in [mutuals, a_uniques, b_uniques, unrelateds, rankings]:
            for i in range(len(freqs)):
                freqs[i] /= len(seeds)
        f = open(filename_ranking, 'w')
        f.write(f"rankings,mutuals,a_uniques,b_uniques,unrelateds\n")
        for i in range(len(rankings)):
            f.write(
                f"{rankings[i]},{mutuals[i]},{a_uniques[i]},{b_uniques[i]},{unrelateds[i]}\n")
        f.close()

    rweras, uias, rwerbs, uibs = [], [], [], []
    f = open(filename_separate)
    for line in f.readlines()[1:]:
        rwera, uia, rwerb, uib = map(float, line.split(','))
        rweras.append(rwera)
        uias.append(uia)
        rwerbs.append(rwerb)
        uibs.append(uib)

    rwers, uis = [], []
    f = open(filename_rwers)
    for line in f.readlines()[1:]:
        rwer, ui = map(float, line.split(','))
        rwers.append(rwer)
        uis.append(ui)

    rankings, mutuals, a_uniques, b_uniques, unrelateds = [], [], [], [], []
    f = open(filename_ranking)
    for line in f.readlines()[1:]:
        ranking, mutual, a_unique, b_unique, unrelated = map(
            float, line.split(','))
        rankings.append(ranking)
        mutuals.append(mutual)
        a_uniques.append(a_unique)
        b_uniques.append(b_unique)
        unrelateds.append(unrelated)

    filename = f"tmp/separate-{id_a}-{id_b}.png"
    draw_scatter(
        [rweras, rwerbs],
        [uias, uibs],
        x_axis_title="total #RWers that pass the graph's edges",
        y_axis_title="UI",
        title="Total #RWers that pass the graph's edges affect UI.",
        labels=["Graph A: " + id_a, "Graph B: " + id_b],
        filename=filename,
    )
    upload_to_imgbb(filename)
    print(correl(rweras + rwerbs, uias + uibs))

    filename = f"tmp/merged-{id_a}-{id_b}.png"
    draw_scatter(
        [rwers],
        [uis],
        x_axis_title="RW_A - RW_B",
        y_axis_title="UI_A - UI_B",
        title="(RW_A - RW_B) and (UI_A - UI_B)",
        filename=filename,
    )
    upload_to_imgbb(filename)
    print(correl(rwers, uis))

    filename = "tmp/rankings.png"
    draw_chart(
        rankings,
        [mutuals, a_uniques, b_uniques, unrelateds],
        title=f"frequency of appearance in top k ranking\ngraph: (A: {id_a}, B: {id_b})",
        labels=["mutuals", "A_uniques",  "B_uniques", "unrelateds"],
        x_axis_title="k",
        y_axis_title="frequency",
        left=0,
        bottom=0,
        filename=filename
    )
    upload_to_imgbb(filename)


def exp_uiu_rwer_comparison_bw_strategies(
    dataset="SBM",
    transition_probability_ratio=1,
):
    # dataset = "REALWORLD"

    STRATEGIES = ["dynamic-rw"]

    if dataset == "SBM":
        id_a, id_b = "0.3", "0.1"
    elif dataset == "REALWORLD":
        id_a, id_b = "Email-Enron", "CA-GrQc"

    rwers, uis = [], []
    labels1, labels2 = [], []
    for strategy in STRATEGIES:
        filename = f"tmp/separate-{id_a}-{id_b}-{strategy}-{transition_probability_ratio}-rweras-uias-rwerbs-uibs.csv"

        rweras, uias, rwerbs, uibs = [], [], [], []
        f = open(filename)
        for line in f.readlines()[1:]:
            rwera, uia, rwerb, uib = map(float, line.split(','))
            rweras.append(rwera)
            uias.append(uia)
            rwerbs.append(rwerb)
            uibs.append(uib)
        rwers.append(rweras)
        rwers.append(rwerbs)
        uis.append(uias)
        uis.append(uibs)
        labels1.append(f'A strategy\n{strategy}')
        labels1.append(f'B strategy\n{strategy}')
        labels2.append(f'A strategy\n{strategy}')
        labels2.append(f'B strategy\n{strategy}')
    filename = "tmp/rwers.png"
    draw_boxplot(
        rwers,
        labels=labels1,
        y_axis_title="RWer",
        bottom=0,
        title=f"RWer difference (A: {id_a}, B: {id_b})\n left two: strategy None, right two: strategy dynamic-rw",
        filename=filename,
        # show_median=True,
    )
    upload_to_imgbb(filename)
    filename = "tmp/uis.png"
    draw_boxplot(
        uis,
        labels=labels2,
        y_axis_title="UI",
        bottom=0,
        title=f"UI difference (A: {id_a}, B: {id_b})\n left two: strategy None, right two: strategy dynamic-rw",
        filename=filename,
    )
    upload_to_imgbb(filename)


def draw_nhop(gm: nx.Graph, seed: int, hop: int, ga: nx.Graph = None, gb: nx.Graph = None, filename="tmp/graph.png"):
    if ga and seed in ga:
        ga_nhop = create_n_hop_graph(ga, seed, hop)
        # ga.add_node(seed)
    if gb and seed in gb:
        gb_nhop = create_n_hop_graph(gb, seed, hop)
        # gb.add_node(seed)
    gm_nhop = create_n_hop_graph(gm, seed, hop)
    nds = [nd for nd in gm_nhop]
    snds, snds2 = set(), set()
    for nd in nds:
        if ga and gb and nd in ga and nd in gb:
            pass
        elif ga and nd in ga:
            snds.add(nd)
        elif gb and nd in gb:
            snds2.add(nd)
        elif ga is None and gb is None:
            pass
        else:
            print("it can't be real")
            exit()
    print(gm.number_of_nodes(), len(snds), len(snds2))
    # for nd in nds:
    #     if specialnodes
    draw_graph(
        gm_nhop,
        special_node=seed,
        special_nodes=snds,
        special_nodes2=snds2,
        bold_edge=True,
        filename=filename,
    )


def exp_rwer_transition_ratio():
    # dataset = "SBM"
    dataset = "REALWORLD"
    # dataset = "REALWORLD2"

    strategy = "dynamic-rw"
    # strategy = "unique-rw"
    # strategy = "backward-rw"

    COMPUTE_THE_VALUES = True
    # COMPUTE_THE_VALUES = False

    if dataset == "SBM":
        id_a, id_b = "0.3", "0.1"
        ga = read_graph(f"tmp/graph/" + id_a + "-0.01-3-100-mixedorder.gr")
        gb = read_graph(f"tmp/graph/" + id_b + "-0.01-3-100-normalorder.gr")
    elif dataset == "REALWORLD":
        id_a, id_b = "Email-Enron", "CA-GrQc"
        ga = read_graph("graph/" + id_a + ".txt")
        gb = read_graph("graph/" + id_b + ".txt")
    elif dataset == "REALWORLD2":
        id_a, id_b = "socfb-Caltech36", "web-edu"
        ga = read_graph("graph/" + id_a + ".mtx")
        gb = read_graph("graph/" + id_b + ".mtx")
    transition_probability_ratios = [1/16 * 2 ** (i) for i in range(9)]

    if strategy == "dynamic-rw":
        rw_x_axis_title = "user-input ratio c"
        rw_y_axis_title = f"#RWer"
        rw_title = "Total #RWers that pass the graph's edges"
        ui_x_axis_title = "user-input ratio c"
        ui_y_axis_title = f"UI"
        ui_title = "UI"
        ratio_title = "resulting ratio of #Rwer and UI"
        ratio_x_axis_title = "user-input ratio c"
        ratio_y_axis_title = "ratio B/A"
        ratio_labels = ["#RWers", "UI"]
        error_title = "resulting error of #Rwer and UI"
        error_x_axis_title = "user-input ratio c"
        error_y_axis_title = "error (B/A) / c"
        error_labels = ["#RWers", "UI", "y = 1"]
    elif strategy == "unique-rw":
        rw_x_axis_title = "user-input ratio c"
        rw_y_axis_title = f"#RWer to uniques"
        rw_title = "Total #RWers to unique nodes"
        ui_x_axis_title = "user-input ratio c"
        ui_y_axis_title = f"UI"
        ui_title = "UI"
        ratio_title = "resulting ratio of #Rwer to unique nodes and UI"
        ratio_x_axis_title = "user-input ratio c"
        ratio_y_axis_title = "ratio B/A"
        ratio_labels = ["#RWers to unique nodes", "UI"]
        error_title = "resulting error of #Rwer to unique nodes and UI"
        error_x_axis_title = "user-input ratio c"
        error_y_axis_title = "error (B/A) / c"
        error_labels = ["#RWers to unique nodes", "UI", "y = 1"]
    elif strategy == "backward-rw":
        rw_x_axis_title = "user-input ratio c"
        rw_y_axis_title = f"Total PPR score of uniques"
        rw_title = "Total PPR score of uniques"
        ui_x_axis_title = "user-input ratio c"
        ui_y_axis_title = f"UI"
        ui_title = "UI"
        ratio_title = "resulting ratio of total unique PPR nodes and UI"
        ratio_x_axis_title = "user-input ratio c"
        ratio_y_axis_title = "ratio B/A"
        ratio_labels = ["total unique PPR", "UI"]
        error_title = "resulting error of total unique PPR nodes and UI"
        error_x_axis_title = "user-input ratio c"
        error_y_axis_title = "error (B/A) / c"
        error_labels = ["total unique PPR", "UI", "y = 1"]

    if COMPUTE_THE_VALUES:
        eta = ETA()
        for tpr_idx, transition_probability_ratio in enumerate(transition_probability_ratios):
            filename_separate = f"tmp/separate-{id_a}-{id_b}-{strategy}-{transition_probability_ratio}-rweras-uias-rwerbs-uibs.csv"
            filename_rwers = f"tmp/merged-{id_a}-{id_b}-{strategy}-{transition_probability_ratio}.csv"
            appra = APPR(ga)
            apprb = APPR(gb)
            gm = merge_two_graphs(ga, gb, data=False)
            apprm = APPR(gm)
            nodes_in_common = list(
                set([nd for nd in ga]).intersection([nd for nd in gb]))
            rwers = []
            uis = []
            rweras, rwerbs = [], []
            uias, uibs = [], []
            # seeds = random.sample(nodes_in_common, 1)
            # seeds = random.sample(nodes_in_common, 10)
            # print(seeds)
            # seeds = [266, 279, 215, 107,  292, 217, 195, 120, 275, 155]
            # seeds = [26092, 6266, 2614, 10637, 11152,
            #          16471, 20059, 4148, 1859, 14131]
            seeds = random.sample(nodes_in_common, 100)
            # seeds = [23303, 8157, 13319, 21287, 22421, 1599, 21998, 2556, 6364, 2851, 15188, 14308, 16563, 1563, 22725, 5060, 3068, 4493, 4748, 543, 2249, 9094, 15715, 21657, 11734, 914, 20554, 13026, 25554, 3412, 22023, 312, 13529, 4630, 18940, 22336, 10211, 24726, 20375, 19444, 18758, 13705, 8669, 7533, 23214, 24961, 5995, 17126, 10388,
            #          23264, 19215, 8728, 19575, 4301, 1695, 14067, 13859, 8054, 12409, 18941, 4144, 15422, 19739, 142, 1323, 23153, 6706, 21379, 6465, 11215, 4278, 3593, 11815, 2009, 19501, 5182, 5365, 187, 14985, 15933, 25510, 2621, 17179, 12599, 17464, 16068, 13404, 1840, 19568, 15708, 4642, 12720, 18283, 1966, 12187, 24943, 22738, 16433, 17189, 920]
            # seeds = random.sample(nodes_in_common, 1000)
            # seeds = nodes_in_common
            for i, seed in enumerate(seeds):
                ca = set(appra.compute_appr(seed))
                cb = set(apprb.compute_appr(seed))
                if strategy == "dynamic-rw":
                    cm, rwera, rwerb, _, _ = apprm.compute_dynamic_appr(
                        seed, ga, gb, c=transition_probability_ratio, data=True)
                elif strategy == "unique-rw":
                    cm, rwera, rwerb = apprm.compute_unique_appr(
                        seed, ga, gb, ca, cb, c=transition_probability_ratio, data=True)
                elif strategy == "backward-rw":
                    cm, rwera, rwerb = apprm.compute_backward_appr(
                        seed, ga, gb, ca, cb, c=transition_probability_ratio, data=True)
                cm = set(cm)
                two = ca.intersection(cb) - cm
                seven = cm - ca - cb
                # print(seed, len(two), len(seven))
                ui_a = unique_influence(ca, cb, cm)
                ui_b = unique_influence(cb, ca, cm)
                rwers.append(rwera - rwerb)
                rweras.append(rwera)
                rwerbs.append(rwerb)
                if transition_probability_ratio == 1 and rwerb / rwera > 2:
                    print(rwera, rwerb)
                uias.append(ui_a)
                uibs.append(ui_b)
                uis.append(ui_a - ui_b)
                vm = apprm.get_appr_vec()
                for nd in gm:
                    if nd not in vm:
                        vm[nd] = 0

            f = open(filename_separate, 'w')
            f.write(f"rweras,uias,rwerbs,uibs\n")
            for i in range(len(rweras)):
                f.write(f"{rweras[i]},{uias[i]},{rwerbs[i]},{uibs[i]}\n")
            f.close()

            f = open(filename_rwers, 'w')
            f.write(f"rwers,uis\n")
            for i in range(len(rwers)):
                f.write(f"{rwers[i]},{uis[i]}\n")
            f.close()
            print(
                "eta:",
                eta.eta((tpr_idx+1) / len(transition_probability_ratios))
            )

    exp_uiu_rwer_transition_ratio(
        dataset,
        transition_probability_ratios,
        strategy,
        rw_x_axis_title,
        rw_y_axis_title,
        rw_title,
        ui_x_axis_title,
        ui_y_axis_title,
        ui_title,
        ratio_title,
        ratio_x_axis_title,
        ratio_y_axis_title,
        ratio_labels,
        error_title,
        error_x_axis_title,
        error_y_axis_title,
        error_labels,
        send_to_slack=False,
    )


def exp_uiu_rwer_transition_ratio(
    dataset: str,
    transition_probability_ratios,
    strategy: str,
    rw_x_axis_title: str,
    rw_y_axis_title: str,
    rw_title: str,
    ui_x_axis_title: str,
    ui_y_axis_title: str,
    ui_title: str,
    ratio_title: str,
    ratio_x_axis_title: str,
    ratio_y_axis_title: str,
    ratio_labels: str,
    error_title: str,
    error_x_axis_title: str,
    error_y_axis_title: str,
    error_labels: str,
    send_to_slack: bool = True,
):
    if dataset == "SBM":
        id_a, id_b = "0.3", "0.1"
    elif dataset == "REALWORLD":
        id_a, id_b = "Email-Enron", "CA-GrQc"
    elif dataset == "REALWORLD2":
        id_a, id_b = "socfb-Caltech36", "web-edu"

    ave_rweras, ave_rwerbs, ave_uias, ave_uibs, ratio_rwers, ratio_uis = [], [], [], [], [], []
    all_rweras, all_rwerbs, all_uias, all_uibs, all_ratio_rwers, all_ratio_uis = [
    ], [], [], [], [], []
    for transition_probability_ratio in transition_probability_ratios:
        filename = f"tmp/separate-{id_a}-{id_b}-{strategy}-{transition_probability_ratio}-rweras-uias-rwerbs-uibs.csv"
        rweras, uias, rwerbs, uibs, ratio_rwer, ratio_ui = [], [], [], [], [], []
        f = open(filename)
        for line in f.readlines()[1:]:
            rwera, uia, rwerb, uib = map(float, line.split(','))
            rweras.append(rwera)
            uias.append(uia)
            rwerbs.append(rwerb)
            uibs.append(uib)
            if rwera != 0:
                ratio_rwer.append(rwerb / rwera)
            if uia != 0:
                ratio_ui.append(uib / uia)

        ave_rweras.append(statistics.median(rweras))
        ave_rwerbs.append(statistics.median(rwerbs))
        ave_uias.append(statistics.median(uias))
        ave_uibs.append(statistics.median(uibs))
        ratio_rwers.append(statistics.median(ratio_rwer))
        ratio_uis.append(statistics.median(ratio_ui))
        all_rweras.append(rweras)
        all_rwerbs.append(rwerbs)
        all_uias.append(uias)
        all_uibs.append(uibs)
        all_ratio_rwers.append(ratio_rwer)
        all_ratio_uis.append(ratio_ui)
    filename = "tmp/rwers.png"
    draw_chart(
        transition_probability_ratios,
        [ave_rweras, ave_rwerbs],
        labels=[f"A: {id_a}", f"B: {id_b}"],
        title=f"{rw_title} (A: {id_a}, B: {id_b}, strategy: {strategy})",
        x_axis_title=rw_x_axis_title,
        y_axis_title=rw_y_axis_title,
        left=transition_probability_ratios[0],
        bottom=0,
        # top=1,
        xscale="log",
        filename=filename
    )
    image_url = upload_to_imgbb(filename)
    if send_to_slack:
        notify_slack(
            title=f"{rw_title} (A: {id_a}, B: {id_b}, strategy: {strategy})",
            result=image_url
        )

    filename = "tmp/uis.png"
    draw_chart(
        transition_probability_ratios,
        [ave_uias, ave_uibs],
        labels=[f"A: {id_a}", f"B: {id_b}"],
        title=f"{ui_title} (A: {id_a}, B: {id_b}, strategy: {strategy})",
        x_axis_title=ui_x_axis_title,
        y_axis_title=ui_y_axis_title,
        left=transition_probability_ratios[0],
        bottom=0,
        top=1,
        xscale="log",
        filename=filename
    )
    image_url = upload_to_imgbb(filename)
    if send_to_slack:
        notify_slack(
            title=f"{ui_title} (A: {id_a}, B: {id_b}, strategy: {strategy})",
            result=image_url
        )

    filename = "tmp/ratios.png"
    draw_chart(
        transition_probability_ratios,
        [ratio_rwers, ratio_uis],
        # labels=[f"#RWers", f"UI"],
        labels=ratio_labels,
        # title=f"ratio of #RWers and UI (A: {id_a}, B: {id_b}, strategy: {strategy})",
        title=f"{ratio_title} (A: {id_a}, B: {id_b}, strategy: {strategy})",
        x_axis_title=ratio_x_axis_title,
        y_axis_title=ratio_y_axis_title,
        left=transition_probability_ratios[0],
        bottom=10 ** (-1),
        # top=1,
        xscale="log",
        yscale="log",
        filename=filename
    )
    image_url = upload_to_imgbb(filename)
    if send_to_slack:
        notify_slack(
            title=f"{ratio_title} (A: {id_a}, B: {id_b}, strategy: {strategy})",
            result=image_url
        )

    filename = "tmp/errors.png"
    error_rwers = [ratio_rwers[i] / transition_probability_ratios[i]
                   for i, _ in enumerate(ratio_rwers)]
    error_uis = [ratio_uis[i] / transition_probability_ratios[i]
                 for i, _ in enumerate(ratio_uis)]
    ideals = [1] * len(ratio_uis)
    draw_chart(
        transition_probability_ratios,
        [error_rwers, error_uis, ideals],
        labels=error_labels,
        # title=f"ratio of #RWers and UI (A: {id_a}, B: {id_b}, strategy: {strategy})",
        title=f"{error_title} (A: {id_a}, B: {id_b}, strategy: {strategy})",
        x_axis_title=error_x_axis_title,
        y_axis_title=error_y_axis_title,
        left=transition_probability_ratios[0],
        bottom=0,
        # top=1,
        xscale="log",
        filename=filename
    )
    image_url = upload_to_imgbb(filename)
    if send_to_slack:
        notify_slack(
            title=f"{ratio_title} (A: {id_a}, B: {id_b}, strategy: {strategy})",
            result=image_url
        )

    filename = "tmp/box.png"
    for i, ratio_rwers in enumerate(all_ratio_rwers):
        for j, _ in enumerate(ratio_rwers):
            ratio_rwers[j] /= transition_probability_ratios[i]
    draw_boxplot(
        all_ratio_rwers,
        title=f"error of ratio of #RWers (A: {id_a}, B: {id_b}, strategy: {strategy})",
        x_axis_title=error_x_axis_title,
        y_axis_title=error_y_axis_title,
        labels=list(map(str, transition_probability_ratios)),
        bottom=0,
        # showfliers=True,
        filename=filename,
    )
    image_url = upload_to_imgbb(filename)
    if send_to_slack:
        notify_slack(
            title=f"{ratio_title} (A: {id_a}, B: {id_b}, strategy: {strategy})",
            result=image_url
        )


def output_merged_clusters():
    PATHS = [
        # ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        # ("graph/0.1-0.01-3-100-normalorder.gr",
        #  "graph/0.3-0.01-3-100-mixedorder.gr"),
        # ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        # ("graph/Email-Enron.txt", "graph/web-edu.mtx"),
        # ("graph/aucs-lunch.gr", "graph/aucs-facebook.gr"),
        ("graph/Airports-Lufthansa.gr", "graph/Airports-Ryanair.gr"),
        # ("graph/dkpol-ff.gr", "graph/dkpol-Re.gr"),
        ("graph/Rattus-DI.gr",
         "graph/Rattus-PA.gr"),
    ]
    STRATEGIES = [
        # "NWF",
        # "WAPPR",
        # "ClassicRW",
        # "RelaxedRW",
        # "WAPPRS",
        # "ML-LCD",

        # OPTIONS
        "WAPPRS-0.125",
        "WAPPRS-0.25",
        "WAPPRS-0.5",
        "WAPPRS-1",
        "WAPPRS-2",
        "WAPPRS-4",
        "WAPPRS-8",

        # OLD
        # "allocation-rw",
        # "dynamic-rw",
        # "OneBased",
        # "AllocationRW",
        # "AllocationRW-ZB",
        # "DynamicRW",
        # "DynamicRW-ZB",
    ]
    SKIP_IF_EXISTS = True
    NUM_PLOTS = 21
    RATIOS = [1 / LARGE] + [(i + 1) / (NUM_PLOTS + 1)
                            for i in range(NUM_PLOTS)] + [1 - 1 / LARGE]
    RATIOS_FOR_ML_LCD = [-1 + 1 / LARGE] + \
        [i / 10 for i in range(-10, 11)] + [1 - 1 / LARGE]

    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)

        nodes_in_common = list(
            set([nd for nd in ga]).intersection([nd for nd in gb]))
        NUM_SAMPLES = None
        if NUM_SAMPLES:
            nodes_in_common = random.sample(nodes_in_common, NUM_SAMPLES)

        try:
            dir = f"cluster/{ganame}-{gbname}"
            os.makedirs(dir)
        except FileExistsError:
            pass
        try:
            dir_top = f"cluster/{ganame}-{gbname}-top"
            os.makedirs(dir_top)
        except FileExistsError:
            pass
        try:
            dir_conductance = f"cluster/{ganame}-{gbname}-conductance"
            os.makedirs(dir_conductance)
        except FileExistsError:
            pass
        eta = ETA()
        for i, strategy in enumerate(STRATEGIES):
            for j, r in enumerate(RATIOS):
                path = f"cluster/{ganame}-{gbname}/{strategy}-{r}.txt"
                if SKIP_IF_EXISTS and os.path.isfile(path):
                    continue
                f = open(path, "w")

                path_top = f"cluster/{ganame}-{gbname}-top/{strategy}-{r}.txt"
                f_top = open(path_top, "w")

                if strategy != "ML-LCD":
                    path_conductance = f"cluster/{ganame}-{gbname}-conductance/{strategy}-{r}.txt"
                    f_conductance = open(path_conductance, "w")

                if strategy == "WAPPR":
                    nga, ngb = normalize_edge_weight(ga, gb, r)
                    gm = merge_two_graphs(nga, ngb, data=True)
                    apprm = APPR(gm)
                # elif strategy == "AllocationRW":
                #     gm = convert_two_graphs_to_digraph(ga, gb, ratio)
                #     apprm = DAPPR(gm)
                # elif strategy == "AllocationRW-ZB":
                #     gm = convert_two_graphs_to_digraph(
                #         ga, gb, ratio, one_based=False)
                #     apprm = DAPPR(gm)
                elif strategy in ["ClassicRW", "RelaxedRW"]:
                    gm = merge_two_graphs(ga, gb, data=False)
                    apprm = APPR(gm, ga, gb)
                elif strategy == "ML-LCD":
                    mllcd = MLLCD([ga, gb], RATIOS_FOR_ML_LCD[j])
                # elif strategy == "OneBased":
                #     gm = convert_two_graphs_to_digraph_one_based(
                #         ga, gb, ratio)
                #     apprm = DAPPR(gm)
                elif strategy == "WAPPRS":
                    gm = merge_two_graphs_with_supernode(ga, gb, r)
                    apprm = APPR(gm)
                elif "WAPPRS-" in strategy:
                    gamma_index = strategy.find("-") + 1
                    gamma = float(strategy[gamma_index:])
                    gm = merge_two_graphs_with_supernode(ga, gb, r, gamma)
                    apprm = APPR(gm)
                else:
                    gm = merge_two_graphs(ga, gb, data=False)
                    apprm = APPR(gm)
                for seed in nodes_in_common:
                    if strategy in ["NWF", "WAPPR", "AllocationRW", "AllocationRW-ZB", "OneBased"]:
                        cm = apprm.compute_appr(seed)
                    # elif strategy == "allocation-rw":
                    #     cm = apprm.compute_allocation_appr(
                    #         seed, ga, gb, c=ratio)
                    # elif strategy == "dynamic-rw":
                    #     cm = apprm.compute_dynamic_appr(seed, ga, gb, c=ratio)
                    # elif strategy == "DynamicRW":
                    #     cm = apprm.compute_dynamic_weighting_appr(
                    #         seed, ga, gb, c=ratio)
                    # elif strategy == "DynamicRW-ZB":
                    #     cm = apprm.compute_dynamic_weighting_appr(
                    #         seed, ga, gb, c=ratio, one_based=False)
                    elif strategy == "ClassicRW":
                        cm = apprm.compute_aclcut_c_appr(seed, omega=r)
                    elif strategy == "RelaxedRW":
                        cm = apprm.compute_aclcut_r_appr(seed, r=r)
                    elif strategy == "ML-LCD":
                        cm = mllcd.compute_mllcd(seed)
                        node_in_order = cm
                    if "WAPPRS" in strategy:
                        cm = apprm.compute_appr_with_supernode(seed)
                    cm.sort()
                    line = f'{seed}'
                    for nd in cm:
                        line += f' {nd}'
                    line += '\n'
                    f.write(line)
                    if strategy != "ML-LCD":
                        node_in_order = apprm.get_node_in_order()
                    line = f'{seed}'
                    for nd in node_in_order:
                        line += f' {nd}'
                    line += '\n'
                    f_top.write(line)

                    if strategy == "ML-LCD":
                        continue

                    conductances = apprm.get_cond_profile()
                    line = f'{seed}'
                    for cond in conductances:
                        line += f' {cond}'
                    line += '\n'
                    f_conductance.write(line)

                f.close()
                print(
                    "ETA:",
                    eta.eta(
                        (j + 1 + i * len(RATIOS)) /
                        (len(RATIOS) * len(STRATEGIES))
                    )
                )
        # notify_slack(
        #     f"output_merged_clusters for ({ganame}, {gbname}) ended",
        #     f"#samples: {NUM_SAMPLES}, total_time: {eta.total_time()}"
        # )


def create_newinfo_figs():
    PATHS = [
        # ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        # ("graph/0.1-0.01-3-100-normalorder.gr",
        #  "graph/0.3-0.01-3-100-mixedorder.gr"),
        # ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        # ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        # ("graph/Email-Enron.txt", "graph/web-edu.mtx"),
        ("graph/dkpol-ff.gr", "graph/dkpol-Re.gr"),
        ("graph/Rattus-DI.gr",
         "graph/Rattus-PA.gr"),
    ]
    STRATEGIES = [
        "NWF",
        "WAPPR",
        # "allocation-rw",
        # "dynamic-rw",
        "AllocationRW",
        "AllocationRW-ZB",
        "DynamicRW",
        "DynamicRW-ZB",
    ]
    RATIOS = [1/1024 * 2 ** (i) for i in range(21)]
    # RATIOS = [1/16 * 2 ** (i) for i in range(9)]
    # STAT = "average"
    STAT = "median"
    WITH_ERRORBARS = True
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        cas = read_clusters(f"cluster/{ganame}/appr.txt")
        cbs = read_clusters(f"cluster/{gbname}/appr.txt")

        list_hiddens, list_fakes, list_a_trues, list_b_trues, list_csizes = [], [], [], [], []
        list_errorbars_hiddens, list_errorbars_fakes, list_errorbars_a_trues, list_errorbars_b_trues, list_errorbars_csizes = [], [], [], [], []
        for strategy in STRATEGIES:
            hiddens, fakes, a_trues, b_trues, csizes = [], [], [], [], []
            hiddens_error_q1, hiddens_error_q3 = [], []
            fakes_error_q1, fakes_error_q3 = [], []
            a_trues_error_q1, a_trues_error_q3 = [], []
            b_trues_error_q1, b_trues_error_q3 = [], []
            csizes_error_q1, csizes_error_q3 = [], []
            for ratio in RATIOS:
                cms = read_clusters(
                    f"cluster/{ganame}-{gbname}/{strategy}-{ratio}.txt")
                hs, fs, a_s, b_s, csz = [], [], [], [], []
                for seed, cm in cms.items():
                    ca = cas[seed]
                    cb = cbs[seed]
                    a_uniques = ca - cb
                    b_uniques = cb - ca
                    intersec = ca.intersection(cb)
                    hidden = len(cm - ca - cb) / len(cm)
                    # hidden = len(cm - ca - cb)
                    hs.append(hidden)
                    try:
                        fake = len(intersec - cm) / len(intersec)
                        # fake = len(intersec - cm)
                        fs.append(fake)
                    except ZeroDivisionError:
                        pass
                    try:
                        # a_true = len(a_uniques.intersection(cm)) / \
                        #     len(a_uniques)
                        a_true = len(a_uniques.intersection(cm)) / len(cm)
                        # a_true = len(a_uniques.intersection(cm))
                        a_s.append(a_true)
                    except ZeroDivisionError:
                        pass
                    try:
                        # b_true = len(b_uniques.intersection(cm)) / \
                        #     len(b_uniques)
                        b_true = len(b_uniques.intersection(cm)) / len(cm)
                        # b_true = len(b_uniques.intersection(cm))
                        b_s.append(b_true)
                    except ZeroDivisionError:
                        pass
                    csz.append(len(cm))
                if STAT == "median":
                    hidden_q1, hidden_med, hidden_q3 = compute_q1_med_q3(hs)
                    fake_q1, fake_med, fake_q3 = compute_q1_med_q3(fs)
                    a_true_q1, a_true_med, a_true_q3 = compute_q1_med_q3(a_s)
                    b_true_q1, b_true_med, b_true_q3 = compute_q1_med_q3(b_s)
                    csize_q1, csize_med, csize_q3 = compute_q1_med_q3(csz)
                    hiddens.append(hidden_med)
                    fakes.append(fake_med)
                    a_trues.append(a_true_med)
                    b_trues.append(b_true_med)
                    csizes.append(csize_med)
                    hiddens_error_q1.append(hidden_med - hidden_q1)
                    fakes_error_q1.append(fake_med - fake_q1)
                    a_trues_error_q1.append(a_true_med - a_true_q1)
                    b_trues_error_q1.append(b_true_med - b_true_q1)
                    csizes_error_q1.append(csize_med - csize_q1)
                    hiddens_error_q3.append(hidden_q3 - hidden_med)
                    fakes_error_q3.append(fake_q3 - fake_med)
                    a_trues_error_q3.append(a_true_q3 - a_true_med)
                    b_trues_error_q3.append(b_true_q3 - b_true_med)
                    csizes_error_q3.append(csize_q3 - csize_med)
                elif STAT == "average":
                    hiddens.append(sum(hs) / len(hs))
                    fakes.append(sum(fs) / len(fs))
                    a_trues.append(sum(a_s) / len(a_s))
                    b_trues.append(sum(b_s) / len(b_s))
                    csizes.append(sum(csz) / len(csz))
            list_hiddens.append(hiddens)
            list_fakes.append(fakes)
            list_a_trues.append(a_trues)
            list_b_trues.append(b_trues)
            list_csizes.append(csizes)
            list_errorbars_hiddens.append([hiddens_error_q1, hiddens_error_q3])
            list_errorbars_fakes.append([fakes_error_q1, fakes_error_q3])
            list_errorbars_a_trues.append([a_trues_error_q1, a_trues_error_q3])
            list_errorbars_b_trues.append([b_trues_error_q1, b_trues_error_q3])
            list_errorbars_csizes.append([csizes_error_q1, csizes_error_q3])

        filename_hidden = f"tmp/hidden-{ganame}-{gbname}-{STAT}.png"
        draw_chart(
            [RATIOS] * len(STRATEGIES),
            list_hiddens,
            list_errorbars=list_errorbars_hiddens if STAT == "median" and WITH_ERRORBARS else None,
            labels=STRATEGIES,
            title=f"fraction of hiddens (A: {ganame}, B: {gbname}, {STAT})",
            x_axis_title="user-input ratio c",
            y_axis_title="fraction of hidden nodes",
            left=RATIOS[0] * 0.9,
            right=RATIOS[-1]*1.1,
            top=max(1, max([max(v) for v in list_hiddens])) * 1.01,
            bottom=-0.01,
            xscale="log",
            loc="upper right",
            filename=filename_hidden,
            print_filename=False,
        )

        filename_fake = f"tmp/fake-{ganame}-{gbname}-{STAT}.png"
        draw_chart(
            [RATIOS] * len(STRATEGIES),
            list_fakes,
            list_errorbars=list_errorbars_fakes if STAT == "median" and WITH_ERRORBARS else None,
            labels=STRATEGIES,
            title=f"fraction of fakes (A: {ganame}, B: {gbname}, {STAT})",
            x_axis_title="user-input ratio c",
            y_axis_title="fraction of fake nodes",
            left=RATIOS[0] * 0.9,
            right=RATIOS[-1]*1.1,
            top=max(1, max([max(v) for v in list_fakes])) * 1.01,
            bottom=-0.01,
            xscale="log",
            loc="upper right",
            filename=filename_fake,
            print_filename=False,
        )

        filename_a_true = f"tmp/a_true-{ganame}-{gbname}-{STAT}.png"
        draw_chart(
            [RATIOS] * len(STRATEGIES),
            list_a_trues,
            list_errorbars=list_errorbars_a_trues if STAT == "median" and WITH_ERRORBARS else None,
            labels=STRATEGIES,
            title=f"fraction of a_trues (A: {ganame}, B: {gbname}, {STAT})",
            x_axis_title="user-input ratio c",
            y_axis_title="fraction of a_true nodes",
            left=RATIOS[0] * 0.9,
            right=RATIOS[-1]*1.1,
            top=max(1, max([max(v) for v in list_a_trues])) * 1.01,
            bottom=-0.01,
            xscale="log",
            loc="upper right",
            filename=filename_a_true,
            print_filename=False,
        )

        filename_b_true = f"tmp/b_true-{ganame}-{gbname}-{STAT}.png"
        draw_chart(
            [RATIOS] * len(STRATEGIES),
            list_b_trues,
            list_errorbars=list_errorbars_b_trues if STAT == "median" and WITH_ERRORBARS else None,
            labels=STRATEGIES,
            title=f"fraction of b_trues (A: {ganame}, B: {gbname}, {STAT})",
            x_axis_title="user-input ratio c",
            y_axis_title="fraction of b_true nodes",
            left=RATIOS[0] * 0.9,
            right=RATIOS[-1]*1.1,
            top=max(1, max([max(v) for v in list_b_trues])) * 1.01,
            bottom=-0.01,
            xscale="log",
            loc="upper right",
            filename=filename_b_true,
            print_filename=False,
        )

        filename = concatanate_images(
            [filename_hidden, filename_fake, filename_a_true, filename_b_true],
            f"tmp/newinfo-{ganame}-{gbname}-{STAT}",
            num_x=2,
            num_y=2,
            print_filename=False
        )
        # upload_to_imgbb(filename)

        filename_csizes = f"tmp/csize-{ganame}-{gbname}-{STAT}.png"
        draw_chart(
            [RATIOS] * len(STRATEGIES),
            list_csizes,
            list_errorbars=list_errorbars_csizes if STAT == "median" and WITH_ERRORBARS else None,
            labels=STRATEGIES,
            title=f"cluster size (A: {ganame}, B: {gbname}, {STAT})",
            x_axis_title="user-input ratio c",
            y_axis_title="cluster size",
            left=RATIOS[0] * 0.9,
            right=RATIOS[-1]*1.1,
            top=None,
            bottom=-0.01,
            xscale="log",
            loc="upper right",
            filename=filename_csizes,
            print_filename=False,
        )
        # upload_to_imgbb(filename_csizes)


def create_hidden_info_figs():
    PATHS = [
        ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        ("graph/0.1-0.01-3-100-normalorder.gr",
         "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        ("graph/Email-Enron.txt", "graph/web-edu.mtx"),
        # # ("graph/com-amazon.ungraph.txt", "graph/com-dblp.ungraph.txt")
    ]
    STRATEGIES = [
        "NWF",
        "WAPPR",
        "allocation-rw",
        "dynamic-rw",
        "AllocationRW",
    ]
    RATIOS = [1/1024 * 2 ** (i) for i in range(21)]
    # RATIOS = [1/16 * 2 ** (i) for i in range(9)]
    # STAT = "average"
    STAT = "median"
    WITH_ERRORBARS = True
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)
        cas = read_clusters(f"cluster/{ganame}/appr.txt")
        cbs = read_clusters(f"cluster/{gbname}/appr.txt")

        list_only = []
        list_errorbars_only = []
        for strategy in STRATEGIES:
            only_in_as, only_in_bs, boths = [], [], []
            only_in_as_error_q1, only_in_as_error_q3 = [], []
            only_in_bs_error_q1, only_in_bs_error_q3 = [], []
            boths_error_q1, boths_error_q3 = [], []
            for ratio in RATIOS:
                cms = read_clusters(
                    f"cluster/{ganame}-{gbname}/{strategy}-{ratio}.txt")
                hs, fs, bs = [], [], []
                for seed, cm in cms.items():
                    ca = cas[seed]
                    cb = cbs[seed]
                    hidden = cm - ca - cb
                    only_in_a, only_in_b, both = 0, 0, 0
                    for nd in hidden:
                        if nd in ga and nd not in gb:
                            only_in_a += 1
                        elif nd not in ga and nd in gb:
                            only_in_b += 1
                        else:
                            both += 1
                    try:
                        hs.append(only_in_a / len(hidden))
                    except ZeroDivisionError:
                        pass
                    try:
                        fs.append(only_in_b / len(hidden))
                    except ZeroDivisionError:
                        pass
                    try:
                        bs.append(both / len(hidden))
                    except ZeroDivisionError:
                        pass
                if STAT == "median":
                    only_in_a_q1, only_in_a_med, only_in_a_q3 = compute_q1_med_q3(
                        hs)
                    only_in_b_q1, only_in_b_med, only_in_b_q3 = compute_q1_med_q3(
                        fs)
                    both_q1, both_med, both_q3 = compute_q1_med_q3(
                        bs)
                    only_in_as.append(only_in_a_med)
                    only_in_bs.append(only_in_b_med)
                    boths.append(both_med)
                    only_in_as_error_q1.append(only_in_a_med - only_in_a_q1)
                    only_in_bs_error_q1.append(only_in_b_med - only_in_b_q1)
                    boths_error_q1.append(both_med - both_q1)
                    only_in_as_error_q3.append(only_in_a_q3 - only_in_a_med)
                    only_in_bs_error_q3.append(only_in_b_q3 - only_in_b_med)
                    boths_error_q3.append(both_q3 - both_med)
            list_only.append(only_in_as)
            list_only.append(only_in_bs)
            list_only.append(boths)
            list_errorbars_only.append(
                [only_in_as_error_q1, only_in_as_error_q3])
            list_errorbars_only.append(
                [only_in_bs_error_q1, only_in_bs_error_q3])
            list_errorbars_only.append(
                [boths_error_q1, boths_error_q3])

        filename_only_in_a = f"tmp/only-{ganame}-{gbname}-{STAT}.png"
        draw_chart(
            RATIOS,
            list_only,
            list_errorbars=list_errorbars_only if STAT == "median" and WITH_ERRORBARS else None,
            labels=["only in A", "only in B", "both in A and in B"],
            title=f"fraction of only_in_as (A: {ganame}, B: {gbname}, {STAT})",
            x_axis_title="user-input ratio c",
            y_axis_title="fraction of only_in_a nodes",
            left=RATIOS[0] * 0.9,
            right=RATIOS[-1]*1.1,
            top=max(1, max([max(v) for v in list_only])) * 1.01,
            bottom=-0.01,
            xscale="log",
            loc="upper right",
            filename=filename_only_in_a,
            print_filename=False,
        )


def compare_overlap_bw_strategies():
    PATHS = [
        ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        ("graph/0.1-0.01-3-100-normalorder.gr",
         "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        ("graph/Email-Enron.txt", "graph/web-edu.mtx",)
    ]
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        cas = read_clusters(f"cluster/{ganame}/appr.txt")
        cbs = read_clusters(f"cluster/{gbname}/appr.txt")
        STRATEGIES = ["NWF", "WAPPR", "allocation-rw",
                      "dynamic-rw"]
        RATIOS = [1/16 * 2 ** (i) for i in range(9)]
        # RATIOS = [0.0625, 0.08838834764831845, 0.125, 0.1767766952966369, 0.25, 0.3535533905932738, 0.5, 0.7071067811865476,
        #           1.0, 1.4142135623730951, 2.0, 2.8284271247461903, 4.0, 5.656854249492381, 8.0, 11.313708498984761, 16.0]
        STAT = "median"
        # STAT = "average"

        seeds = None
        filenames = []
        for ratio in RATIOS:
            hiddens, fakes, a_trues, b_trues = {}, {}, {}, {}
            similarity_hidden = {strategy: {strategy: 0 for strategy in STRATEGIES}
                                 for strategy in STRATEGIES}
            similarity_fake = {strategy: {strategy: 0 for strategy in STRATEGIES}
                               for strategy in STRATEGIES}
            similarity_a_true = {strategy: {strategy: 0 for strategy in STRATEGIES}
                                 for strategy in STRATEGIES}
            similarity_b_true = {strategy: {strategy: 0 for strategy in STRATEGIES}
                                 for strategy in STRATEGIES}
            for strategy in STRATEGIES:
                hiddens[strategy] = {}
                fakes[strategy] = {}
                a_trues[strategy] = {}
                b_trues[strategy] = {}
                cms = read_clusters(
                    f"cluster/{ganame}-{gbname}/{strategy}-{ratio}.txt")
                if seeds is None:
                    seeds = list(cms.keys())
                for seed, cm in cms.items():
                    ca = cas[seed]
                    cb = cbs[seed]
                    a_uniques = ca - cb
                    b_uniques = cb - ca
                    intersec = ca.intersection(cb)
                    hidden = cm - ca - cb
                    fake = intersec - cm
                    a_true = a_uniques.intersection(cm)
                    b_true = b_uniques.intersection(cm)
                    hiddens[strategy][seed] = hidden
                    fakes[strategy][seed] = fake
                    a_trues[strategy][seed] = a_true
                    b_trues[strategy][seed] = b_true
            for strategy1 in STRATEGIES:
                for strategy2 in STRATEGIES:
                    if strategy1 == strategy2:
                        similarity_hidden[strategy1][strategy2] = 1
                        similarity_fake[strategy1][strategy2] = 1
                        similarity_a_true[strategy1][strategy2] = 1
                        similarity_b_true[strategy1][strategy2] = 1
                        continue
                    sims_hidden = []
                    sims_fake = []
                    sims_a_true = []
                    sims_b_true = []
                    for seed in seeds:
                        hidden1 = hiddens[strategy1][seed]
                        hidden2 = hiddens[strategy2][seed]
                        fake1 = fakes[strategy1][seed]
                        fake2 = fakes[strategy2][seed]
                        a_true1 = a_trues[strategy1][seed]
                        a_true2 = a_trues[strategy2][seed]
                        b_true1 = b_trues[strategy1][seed]
                        b_true2 = b_trues[strategy2][seed]
                        try:
                            sim = len(hidden1.intersection(
                                hidden2)) / len(hidden2)
                            sims_hidden.append(sim)
                        except ZeroDivisionError:
                            pass
                        try:
                            sim = len(fake1.intersection(fake2)) / len(fake2)
                            sims_fake.append(sim)
                        except ZeroDivisionError:
                            pass
                        try:
                            sim = len(a_true1.intersection(
                                a_true2)) / len(a_true2)
                            sims_a_true.append(sim)
                        except ZeroDivisionError:
                            pass
                        try:
                            sim = len(b_true1.intersection(
                                b_true2)) / len(b_true2)
                            sims_b_true.append(sim)
                        except ZeroDivisionError:
                            pass
                    if STAT == "median":
                        similarity_hidden[strategy1][strategy2] = statistics.median(
                            sims_hidden)
                        similarity_fake[strategy1][strategy2] = statistics.median(
                            sims_fake)
                        similarity_a_true[strategy1][strategy2] = statistics.median(
                            sims_a_true)
                        similarity_b_true[strategy1][strategy2] = statistics.median(
                            sims_b_true)
                    elif STAT == "average":
                        similarity_hidden[strategy1][strategy2] = sum(
                            sims_hidden) / len(sims_hidden)
                        similarity_fake[strategy1][strategy2] = sum(
                            sims_fake) / len(sims_fake)
                        similarity_a_true[strategy1][strategy2] = sum(
                            sims_a_true) / len(sims_a_true)
                        similarity_b_true[strategy1][strategy2] = sum(
                            sims_b_true) / len(sims_b_true)
            filename = f"tmp/similarity-{ganame}-{gbname}-{ratio}.png"
            filenames.append(filename)
            draw_heatmap(
                similarity_hidden,
                title=f"similarity (c = {ratio}, {STAT})\nA: {ganame}, B: {gbname}",
                filename=filename
            )

        imgname = concatanate_images(
            filenames, "tmp/similarity", 3, 3, print_filename=False)
        upload_to_imgbb(imgname)


def compare_diff_datasets(
        paths=[
            ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
            ("graph/0.1-0.01-3-100-normalorder.gr",
             "graph/0.3-0.01-3-100-mixedorder.gr"),
            ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
            ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
            ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
            ("graph/Email-Enron.txt", "graph/web-edu.mtx"),
        ],
        # metric_sim="similarity",
        metrics=[
            "hidden",
            "fake",
            "a_true",
            "b_true",
            "csize",
            # "only",
            # "rwers",
            # "ratios",
            # "errors",
        ],
        stat="median",
):
    # filenames_sim = []
    metric2filenames = defaultdict(list)
    for ga_path, gb_path in paths:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        # filenames_sim.append(f"tmp/{metric_sim}-{ganame}-{gbname}-{1.0}.png")
        for metric in metrics:
            metric2filenames[metric].append(
                f"tmp/{metric}-{ganame}-{gbname}-{stat}.png")
    # filename_sim = concatanate_images(filenames_sim, f'tmp/{metric_sim}')
    # upload_to_imgbb(filename_sim)

    for metric in metrics:
        filename = concatanate_images(
            metric2filenames[metric],
            f"tmp/{metric}",
            print_filename=False,
        )
        upload_to_imgbb(filename)


def compare_ppr_methods():
    # ga_path, gb_path = "graph/Email-Enron.txt", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/0.1-0.01-3-100-normalorder.gr", "graph/0.3-0.01-3-100-mixedorder.gr"
    # ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/web-edu.mtx", "graph/CA-GrQc.txt"
    ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/Email-Enron.txt", "graph/web-edu.mtx"
    ga = read_graph(ga_path)
    from ppr import ppr, manual_ppr, appr
    diff_old = []
    diff_new = []
    eta = ETA()
    nds = random.sample([nd for nd in ga], 100)
    for i, nd in enumerate(nds):
        tr = ppr(ga, nd)
        old_appr = appr(ga, nd)
        new_appr = manual_ppr(ga, nd)
        sum_diff = 0
        for nd in nds:
            sum_diff += abs(old_appr[nd] - tr[nd])
        diff_old.append(sum_diff / len(ga))
        sum_diff = 0
        for nd in nds:
            sum_diff += abs(new_appr[nd] - tr[nd])
        diff_new.append(sum_diff / len(ga))
        print(eta.eta((i + 1) / len(nds)))
    fname = "tmp/tmp.png"
    draw_boxplot(
        [diff_old, diff_new],
        labels=["old", "new"],
        y_axis_title="difference from true PPR score",
        filename=fname
    )
    upload_to_imgbb(fname)
    pass


def check_hidden_with_seed():
    # ga_path, gb_path = "graph/Email-Enron.txt", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/0.1-0.01-3-100-normalorder.gr", "graph/0.3-0.01-3-100-mixedorder.gr"
    # ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/web-edu.mtx", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/Email-Enron.txt", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/aucs-lunch.gr", "graph/aucs-facebook.gr"
    # ga_path, gb_path = "graph/Airports-Lufthansa.gr", "graph/Airports-Ryanair.gr"
    # ga_path, gb_path = "graph/dkpol-ff.gr", "graph/dkpol-Re.gr"
    ga_path, gb_path = "graph/Rattus-DI.gr", "graph/Rattus-PA.gr"
    ganame = get_gname(ga_path)
    gbname = get_gname(gb_path)
    ga = read_graph(ga_path)
    gb = read_graph(gb_path)
    # STRATEGY = "WAPPR"
    # STRATEGY = "allocation-rw"
    # STRATEGY = "AllocationRW"
    STRATEGY = "AllocationRW-ZB"
    # STRATEGY = "DynamicRW"
    # RATIOS = [0.0009765625, ]
    # RATIOS = [16.0, 32.0, 64.0]
    # RATIOS = [1/1024, 1, 1024]
    RATIOS = [1/1024]
    # RATIOS = [8, 16, 32]
    # RATIOS = [1/1024 * 2 ** (i) for i in range(21)]
    DEGREE_NORMALIZED = True
    for ratio in RATIOS:
        print("- ratio: ", ratio, sep='')
        if STRATEGY == "WAPPR":
            nga, ngb = normalize_edge_weight(ga, gb, ratio)
            gm = merge_two_graphs(nga, ngb, data=True)
        elif STRATEGY in ["AllocationRW"]:
            gm = convert_two_graphs_to_digraph(ga, gb, ratio)
        elif STRATEGY in ["AllocationRW-ZB"]:
            gm = convert_two_graphs_to_digraph(ga, gb, ratio, one_based=False)
        else:
            gm = merge_two_graphs(ga, gb, data=False)
        appra = APPR(ga)
        apprb = APPR(gb)
        if STRATEGY in ["AllocationRW", "AllocationRW-ZB"]:
            apprm = DAPPR(gm)
        else:
            apprm = APPR(gm)
        # print(list(
        #     set([nd for nd in ga]).intersection([nd for nd in gb])))
        if len(sys.argv) < 2:
            seed = 1264
        else:
            seed = int(sys.argv[1])
        ca = appra.compute_appr(seed, degree_normalized=DEGREE_NORMALIZED)
        da = nx.single_source_dijkstra_path_length(ga, seed, weight=None)
        ppr_vec_a = appra.get_appr_vec()

        if DEGREE_NORMALIZED:
            wdeg_as = dict()
            for k in ppr_vec_a.keys():
                try:
                    wdeg_as[k] = sum([ga[k][nbr]['weight']
                                     for nbr in ga.neighbors(k)])
                except KeyError:
                    wdeg_as[k] = ga.degree(k)
            ppr_a = [k for k, _ in sorted(
                [(k, v / wdeg_as[k]) for k, v in ppr_vec_a.items()], key=lambda x:x[1], reverse=True)]
        else:
            ppr_a = [k for k, _ in sorted(
                [(k, v) for k, v in ppr_vec_a.items()], key=lambda x:x[1], reverse=True)]
        cb = apprb.compute_appr(seed, DEGREE_NORMALIZED)
        db = nx.single_source_dijkstra_path_length(gb, seed, weight=None)
        ppr_vec_b = apprb.get_appr_vec()
        if DEGREE_NORMALIZED:
            wdeg_bs = dict()
            for k in ppr_vec_b.keys():
                try:
                    wdeg_bs[k] = sum([gb[k][nbr]['weight']
                                     for nbr in gb.neighbors(k)])
                except KeyError:
                    wdeg_bs[k] = gb.degree(k)
            ppr_b = [k for k, _ in sorted(
                [(k, v / wdeg_bs[k]) for k, v in ppr_vec_b.items()], key=lambda x:x[1], reverse=True)]
        else:
            ppr_b = [k for k, _ in sorted(
                [(k, v) for k, v in ppr_vec_b.items()], key=lambda x:x[1], reverse=True)]
        if STRATEGY in ["NWF", "WAPPR", "AllocationRW", "AllocationRW-ZB"]:
            cm = apprm.compute_appr(seed, DEGREE_NORMALIZED)
        elif STRATEGY == "allocation-rw":
            cm = apprm.compute_allocation_appr(
                seed, ga, gb, c=ratio, degree_normalized=DEGREE_NORMALIZED)
        elif STRATEGY == "DynamicRW":
            cm = apprm.compute_dynamic_weighting_appr(
                seed, ga, gb, c=ratio, degree_normalized=DEGREE_NORMALIZED)
        dm = nx.single_source_dijkstra_path_length(gm, seed, weight=None)
        ppr_vec_m = apprm.get_appr_vec()
        if DEGREE_NORMALIZED:
            wdeg_ms = dict()
            for k in ppr_vec_m.keys():
                try:
                    wdeg_ms[k] = sum([gm[k][nbr]['weight'] + gm[nbr][k]['weight']
                                     for nbr in gm.neighbors(k)])
                except KeyError:
                    wdeg_ms[k] = gm.in_degree(k) + gm.out_degree(k)
            ppr_m = [k for k, _ in sorted(
                [(k, v / wdeg_ms[k]) for k, v in ppr_vec_m.items()], key=lambda x:x[1], reverse=True)]
        else:
            ppr_m = [k for k, _ in sorted(
                [(k, v) for k, v in ppr_vec_m.items()], key=lambda x:x[1], reverse=True)]
        if DEGREE_NORMALIZED:
            ppr_label_ab, ppr_label_a, ppr_label_b = "DNPPR in A+B", "DNPPR in A", "DNPPR in B",
        else:
            ppr_label_ab, ppr_label_a, ppr_label_b = "PPR in A+B", "PPR in A", "PPR in B",
        labels = [
            "rank",
            # f"cm (hidden: { round(len(hidden) / len(cm) * 100, 1)}%)",
            f"node",
            "rank in A+B", "rank in A", "rank in B",
            "conductance",
            ppr_label_ab, ppr_label_a, ppr_label_b,
            "dist in A+B", "dist in A", "dist in B",
            "deg in A+B", "deg in A", "deg in B",
            # "#paths in A+B", "#paths in A", "#paths in B"
        ]
        # print(seed in ga, seed in gb, file=sys.stderr)
        for cluster in [
            "A",
            "B",
            "A+B"
        ]:
            if cluster == "A":
                # c = ca
                c = appra.get_node_in_order()
                conds = appra.get_cond_profile()
            elif cluster == "B":
                # c = cb
                c = apprb.get_node_in_order()
                conds = apprb.get_cond_profile()
            elif cluster == "A+B":
                # c = cm
                c = apprm.get_node_in_order()
                conds = apprm.get_cond_profile()

            records = []
            rank = 0
            for nd in c:
                rank += 1
                # if CLUSTER == "A+B" and (nd in ca or nd in cb):
                #     continue
                ndm = f'{nd}'
                if DEGREE_NORMALIZED:
                    try:
                        wdegm = sum([gm[nd][nbr]['weight'] + gm[nbr][nd]['weight']
                                     for nbr in gm.neighbors(nd)])
                        score_m = ppr_vec_m[nd] / wdegm
                    except AttributeError:
                        score_m = ppr_vec_m[nd] / \
                            gm.degree(nd, weight='weight')
                else:
                    score_m = ppr_vec_m[nd]
                    score_m = 0
                try:
                    if DEGREE_NORMALIZED:
                        score_a = ppr_vec_a[nd] / sum([ga[nd][nbr]['weight']
                                                      for nbr in ga.neighbors(nd)])
                    else:
                        score_a = ppr_vec_a[nd]
                except:
                    score_a = 0
                try:
                    if DEGREE_NORMALIZED:
                        score_b = ppr_vec_b[nd] / sum([gb[nd][nbr]['weight']
                                                      for nbr in gb.neighbors(nd)])
                    else:
                        score_b = ppr_vec_b[nd]
                except:
                    score_b = 0
                dist_m = dm[nd]
                if nd in da:
                    dist_a = da[nd]
                elif nd in ga:
                    dist_a = "inf"
                else:
                    dist_a = "not exists"
                if nd in db:
                    dist_b = db[nd]
                elif nd in gb:
                    dist_b = "inf"
                else:
                    dist_b = "not exists"
                try:
                    rank_in_a = ppr_a.index(nd) + 1
                    if rank_in_a > len(ca):
                        rank_in_a = f'{rank_in_a} (out)'
                except:
                    rank_in_a = "out"
                try:
                    rank_in_b = ppr_b.index(nd) + 1
                    if rank_in_b > len(cb):
                        rank_in_b = f'{rank_in_b} (out)'
                except:
                    rank_in_b = "out"
                try:
                    rank_in_m = ppr_m.index(nd) + 1
                    if rank_in_m > len(cm):
                        rank_in_m = f'{rank_in_m} (out)'
                except:
                    rank_in_m = "out"
                try:
                    dega = ga.degree(nd)
                except:
                    dega = 0
                try:
                    degb = gb.degree(nd)
                except:
                    degb = 0
                degm = gm.degree(nd, weight='weight')
                # try:
                #     num_paths_a = len(
                #         [_ for _ in nx.all_shortest_paths(ga, seed, nd, weight=None)])
                # except:
                #     num_paths_a = 0
                # try:
                #     num_paths_b = len(
                #         [_ for _ in nx.all_shortest_paths(gb, seed, nd, weight=None)])
                # except:
                #     num_paths_b = 0
                # try:
                #     all_shortest_paths = [
                #         _ for _ in nx.all_shortest_paths(gm, seed, nd, weight=None)]
                #     num_paths_m = len(all_shortest_paths)
                # except:
                #     num_paths_m = 0
                # if dist_a != dist_m and dist_b != dist_m:
                #     continue
                records.append([
                    f"{rank} / {len(c)}", ndm, rank_in_m, rank_in_a, rank_in_b,
                    conds[rank - 1],
                    float(f"{score_m:.03g}"), float(
                        f"{score_a:.03g}"), float(f"{score_b:.03g}"),
                    dist_m, dist_a, dist_b,
                    degm, dega, degb,
                    # num_paths_m, num_paths_a, num_paths_b,
                ])
            # records.sort(key=lambda x: x[11])
            # export_to_csv(records, labels)
            # export_table(records, labels)
            print(f'    - {cluster}: ', end='')
            export_to_google_sheets(
                records,
                labels,
                title=f"cluster in {cluster} (A: {ganame}, B: {gbname}, seed: {seed}, c: {ratio}, strategy: {STRATEGY}, DN: {DEGREE_NORMALIZED})",
                template_id="17FNHpVoc6XbzqxSm3HQ4gy81q3-mDDmPHx3uC0s8QGE"
            )
            # print("fraction of hidden:", len(hidden)/len(cm))


def check_hidden():
    PATHS = [
        ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        ("graph/0.1-0.01-3-100-normalorder.gr",
         "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        ("graph/Email-Enron.txt", "graph/web-edu.mtx")
    ]
    STRATEGY = "NWF"
    # STRATEGY = "allocation-rw"
    RATIO = 1.0
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)
        if STRATEGY == "WAPPR":
            nga, ngb = normalize_edge_weight(ga, gb, RATIO)
            gm = merge_two_graphs(nga, ngb, data=True)
        else:
            gm = merge_two_graphs(ga, gb, False)

        COMPUTE = True
        if COMPUTE:
            cas = read_clusters(f"cluster/{ganame}/appr.txt")
            cbs = read_clusters(f"cluster/{gbname}/appr.txt")
            cms = read_clusters(
                f"cluster/{ganame}-{gbname}/{STRATEGY}-{RATIO}.txt")
            nodes_in_common = list(
                set([nd for nd in ga]).intersection([nd for nd in gb]))
            # nodes_in_common = random.sample(nodes_in_common, 100)
            eta = ETA()
            all_cnt = 0
            zeros = []
            list_counter_tuple_orig = []
            list_counter_tuple_merged = []
            list_counter_tuple_diff = []
            for i, seed in enumerate(nodes_in_common):
                counter_orig = defaultdict(int)
                counter_merged = defaultdict(int)
                counter_diff = defaultdict(int)
                ca = cas[seed]
                das = nx.single_source_dijkstra_path_length(ga, seed)
                cb = cbs[seed]
                dbs = nx.single_source_dijkstra_path_length(gb, seed)
                cm = cms[seed]
                dms = nx.single_source_dijkstra_path_length(gm, seed)
                hidden = set(cm) - set(ca) - set(cb)
                # for nd in hidden:
                for nd in cm:
                    da = das[nd] if nd in das else INF
                    db = dbs[nd] if nd in dbs else INF
                    dm = dms[nd] if nd in dms else INF
                    counter_orig[min(da, db)] += 1
                    counter_merged[dm] += 1
                    if min(da, db) == INF:
                        counter_diff[INF] += 1
                    else:
                        counter_diff[min(da, db) - dm] += 1
                        if min(da, db) - dm == 0:
                            zeros.append((seed, nd))
                    all_cnt += 1
                print("ETA:", eta.eta((i + 1) / len(nodes_in_common)))
                counter_tuple_orig = sorted(
                    [(k, v) for k, v in counter_orig.items()])
                counter_tuple_merged = sorted(
                    [(k, v) for k, v in counter_merged.items()])
                counter_tuple_diff = sorted(
                    [(k, v) for k, v in counter_diff.items()])
                counter_tuple_orig = [seed] + \
                    [j for sub in counter_tuple_orig for j in sub]
                counter_tuple_merged = [seed] + \
                    [j for sub in counter_tuple_merged for j in sub]
                counter_tuple_diff = [seed] + \
                    [j for sub in counter_tuple_diff for j in sub]
                list_counter_tuple_orig.append(counter_tuple_orig)
                list_counter_tuple_merged.append(counter_tuple_merged)
                list_counter_tuple_diff.append(counter_tuple_diff)
            export_to_simple_file(
                zeros, f"tmp/zeros-{ganame}-{gbname}-{STRATEGY}.txt")
            export_to_simple_file(list_counter_tuple_orig,
                                  f"tmp/orig-{ganame}-{gbname}-{STRATEGY}.txt")
            export_to_simple_file(list_counter_tuple_merged,
                                  f"tmp/merged-{ganame}-{gbname}-{STRATEGY}.txt")
            export_to_simple_file(list_counter_tuple_diff,
                                  f"tmp/diff-{ganame}-{gbname}-{STRATEGY}.txt")

        def convert_to_percentage(list_counter_tuple: List[List]):

            dist2sum_percentages = defaultdict(int)
            num_seeds_with_hidden = 0
            for line in list_counter_tuple:
                seed = line[0]
                if len(line) == 1:
                    continue
                num_seeds_with_hidden += 1
                num_dists = len(line) // 2
                total_count = 0
                for i in range(num_dists):
                    count = line[i * 2 + 2]
                    total_count += count
                for i in range(num_dists):
                    dist = line[i * 2 + 1]
                    count = line[i * 2 + 2]
                    dist2sum_percentages[dist] += count / total_count
            for k in dist2sum_percentages:
                dist2sum_percentages[k] /= num_seeds_with_hidden
            return sorted([(k, v) for k, v in dist2sum_percentages.items()])

            # # if you want to get an agerage over all hidden nodes
            # # instead of percentages for all seed nodes
            # # old data (e.g., https://github.com/TeraokaKanekoLab/thenter-journal/blob/master/2022/10/1019.md#allocation-rw-探索)
            # dist2cnt = defaultdict(int)
            # for line in list_counter_tuple:
            #     if len(line) == 1:
            #         print(line)
            #         continue
            #     seed = line[0]
            #     for i in range(len(line) // 2):
            #         dist = line[i * 2 + 1]
            #         count = line[i * 2 + 2]
            #         dist2cnt[dist] += count
            # return sorted([(k, v) for k, v in dist2cnt.items()])

        DIST = True
        if DIST:
            list_counter_tuple_orig = read_from_simple_file(
                f"tmp/orig-{ganame}-{gbname}-{STRATEGY}.txt")
            list_counter_tuple_merged = read_from_simple_file(
                f"tmp/merged-{ganame}-{gbname}-{STRATEGY}.txt")
            list_counter_tuple_diff = read_from_simple_file(
                f"tmp/diff-{ganame}-{gbname}-{STRATEGY}.txt")

            counter_tuple_orig = convert_to_percentage(list_counter_tuple_orig)
            counter_tuple_merged = convert_to_percentage(
                list_counter_tuple_merged)
            counter_tuple_diff = convert_to_percentage(list_counter_tuple_diff)
            for l in [counter_tuple_orig, counter_tuple_diff]:
                sum_cnt = sum([cnt for _, cnt in l])
                limit = len(l)
                for i, len_cnt in enumerate(l[-2::-1]):
                    if len_cnt[1] / sum_cnt * 100 >= 3:
                        limit = len(l) - i - 1
                        break
                if len(l) > limit + 2:
                    sum_cnt = 0
                    for i in range(limit, len(l) - 1):
                        sum_cnt += l[i][1]
                    l[limit] = (f">{l[limit - 1][0]}", sum_cnt)
                    l[limit + 1] = ("inf", l[-1][1])
                    del l[limit+2:]
            labels_orig = [v[0] for v in counter_tuple_orig]
            data_orig = [v[1] for v in counter_tuple_orig]
            labels_merged = [v[0] for v in counter_tuple_merged]
            data_merged = [v[1] for v in counter_tuple_merged]
            labels_diff = [v[0] for v in counter_tuple_diff]
            data_diff = [v[1] for v in counter_tuple_diff]
            fname1 = f"tmp/orig-{ganame}-{gbname}-{STRATEGY}.png"
            draw_pie_chart(
                data_orig,
                labels_orig,
                title=f"min distance (in A or in B) of hidden nodes from seed node\n(A: {ganame}, B: {gbname}, strategy: {STRATEGY})",
                filename=fname1
            )
            upload_to_imgbb(fname1)
            fname2 = f"tmp/merged-{ganame}-{gbname}-{STRATEGY}.png"
            draw_pie_chart(
                data_merged,
                labels_merged,
                title=f"distance (in A+B) of hidden nodes from seed node\n(A: {ganame}, B: {gbname}, strategy: {STRATEGY})",
                filename=fname2
            )
            upload_to_imgbb(fname2)
            fname3 = f"tmp/diff-{ganame}-{gbname}-{STRATEGY}.png"
            draw_pie_chart(
                data_diff,
                labels_diff,
                title=f"dist min(in A, in B) - dist in A+B \n(A: {ganame}, B: {gbname}, strategy: {STRATEGY})",
                filename=fname3
            )
            upload_to_imgbb(fname3)

        ZEROS = False
        if ZEROS:
            zeros = read_from_simple_file(
                f"tmp/zeros-{ganame}-{gbname}-{STRATEGY}.txt")
            seed2nd = defaultdict(set)
            for seed, nd in zeros:
                seed2nd[seed].add(nd)
            appra = APPR(ga)
            apprb = APPR(gb)
            apprm = APPR(gm)
            cnt_in = 0
            cnt_out = 0
            seed2cnt = []
            cnt_same_num_paths, cnt_diff_num_paths = 0, 0
            eta = ETA()
            cnt_done = 0
            for seed in seed2nd.keys():
                seed_cnt = 0
                ca = appra.compute_appr(seed)
                ca_size = len(ca)
                appr_vec_a = sorted([(k, v) for k, v in appra.get_appr_vec(
                ).items()], key=lambda x: x[1], reverse=True)
                cb = apprb.compute_appr(seed)
                cb_size = len(cb)
                if STRATEGY in ["NWF", "WAPPR"]:
                    cm = apprm.compute_appr(seed)
                elif STRATEGY == "allocation-rw":
                    cm = apprm.compute_allocation_appr(seed, ga, gb, RATIO)
                cm_size = len(cm)
                das = nx.single_source_dijkstra_path_length(ga, seed)
                dbs = nx.single_source_dijkstra_path_length(gb, seed)
                dms = nx.single_source_dijkstra_path_length(gm, seed)
                for nd in seed2nd[seed]:
                    try:
                        da = das[nd]
                    except:
                        da = INF
                    try:
                        db = dbs[nd]
                    except:
                        db = INF
                    dm = dms[nd]
                    idx = cm.index(nd)
                    if da == db == dm:
                        csize = min(ca_size, cb_size)
                    elif da == dm:
                        csize = ca_size
                    elif db == dm:
                        csize = cb_size
                    # if csize > min(ca_size, cb_size):
                    #     print("probelm", csize, ca_size, cb_size, da, db, dm)
                    if idx < csize:
                        cnt_in += 1
                        for i, (k, v) in enumerate(appr_vec_a):
                            if k == nd:
                                rank = i + 1
                                break
                        else:
                            rank = "out"
                        try:
                            num_paths_a = len(
                                [_ for _ in nx.all_shortest_paths(ga, seed, nd)])
                        except:
                            num_paths_a = 0
                        try:
                            num_paths_b = len(
                                [_ for _ in nx.all_shortest_paths(gb, seed, nd)])
                        except:
                            num_paths_b = 0
                        try:
                            num_paths_m = len(
                                [_ for _ in nx.all_shortest_paths(gm, seed, nd)])
                        except:
                            num_paths_m = 0
                        if da == dm:
                            if num_paths_a == num_paths_m:
                                cnt_same_num_paths += 1
                            else:
                                cnt_diff_num_paths += 1
                        elif db == dm:
                            if num_paths_b == num_paths_m:
                                cnt_same_num_paths += 1
                            else:
                                cnt_diff_num_paths += 1
                            # print("seed: ", seed, ", nd: ", nd,
                            #       ", old_rank: ", rank, ", new_rank: ", idx + 1, ", old_cluster_size: ", csize,
                            #       ", da: ", da, ", db: ", db, ", dm: ", dm,
                            #       ", ca_size: ", ca_size, ", cb_size: ", cb_size, ", cm_size: ", cm_size, ", csize: ", csize,
                            #       #   ", num_paths_a: ", num_paths_a, ", num_paths_b: ", num_paths_b, ", num_paths_m: ", num_paths_m,
                            #       sep='')
                        seed_cnt += 1
                    else:
                        cnt_out += 1
                    cnt_done += 1
                seed2cnt.append((seed, seed_cnt))
                print("seed: ", seed, ", cnt: ", seed_cnt, ", ETA: ",
                      eta.eta(cnt_done / len(zeros)), sep='')
            seed2cnt.sort(key=lambda x: x[1], reverse=True)
            print(seed2cnt[:10])
            data = [cnt_in, cnt_out]
            data2 = [cnt_same_num_paths, cnt_diff_num_paths]
            labels = ["under", "above"]
            fname = f"tmp/inandout-{ganame}-{gbname}-{STRATEGY}.png"
            draw_pie_chart(
                data,
                labels,
                title=f"Whether the new ranking is under the old cluster size \n(A: {ganame}, B: {gbname}, strategy: {STRATEGY})",
                filename=fname
            )
            upload_to_imgbb(fname)
            fname = f"tmp/num_paths-{ganame}-{gbname}-{STRATEGY}.png"
            draw_pie_chart(
                data2,
                ["same", "different"],
                title=f"Whether #shortest paths are same in A+B and (A or B) \n(A: {ganame}, B: {gbname}, strategy: {STRATEGY})",
                filename=fname
            )
            image_url = upload_to_imgbb(fname)
            send_to_slack = False
            if send_to_slack:
                notify_slack(
                    title=f"{fname} (A: {ganame}, B: {gbname}, strategy: {None})",
                    result=image_url,
                    with_mention=True,
                )


def check_hidden2():
    PATHS = [
        ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        ("graph/0.1-0.01-3-100-normalorder.gr",
         "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        ("graph/Email-Enron.txt", "graph/web-edu.mtx")
    ]
    STRATEGIES = ["NWF", "allocation-rw", "dynamic-rw"]
    RATIO = 1.0
    NUM_SAMPLES = 1000
    filenames1 = []
    filenames2 = []
    filenames3 = []
    filenames4 = []
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)
        list_ratio_ppr_shortened = []
        list_ratio_shortened_in_cluster = []
        nodes_in_common = list(
            set([nd for nd in ga]).intersection([nd for nd in gb]))
        if NUM_SAMPLES and len(nodes_in_common) > NUM_SAMPLES:
            nodes_in_common = random.sample(nodes_in_common, NUM_SAMPLES)

        for strategy in STRATEGIES:
            if strategy == "WAPPR":
                nga, ngb = normalize_edge_weight(ga, gb, RATIO)
                gm = merge_two_graphs(nga, ngb, data=True)
            else:
                gm = merge_two_graphs(ga, gb, False)
            appra = APPR(ga)
            apprb = APPR(gb)
            apprm = APPR(gm)
            eta = ETA()
            ratio_hiddens = []
            ratio_not_hiddens = []
            ratio_ppr_shortened = []
            ratio_shortened_in_cluster = []
            for i, seed in enumerate(nodes_in_common):
                ca = appra.compute_appr(seed)
                das = nx.single_source_dijkstra_path_length(ga, seed)
                cb = apprb.compute_appr(seed)
                dbs = nx.single_source_dijkstra_path_length(gb, seed)
                if strategy == "NWF":
                    cm = set(apprm.compute_appr(seed))
                elif strategy == "allocation-rw":
                    cm = set(apprm.compute_allocation_appr(
                        seed, ga, gb, RATIO))
                elif strategy == "dynamic-rw":
                    cm = set(apprm.compute_dynamic_appr(
                        seed, ga, gb, RATIO))
                dms = nx.single_source_dijkstra_path_length(gm, seed)
                appr_vec = apprm.get_appr_vec()
                hidden = set(cm) - set(ca) - set(cb)
                cnt_shortened = 0
                cnt_hidden = 0
                cnt_not_hidden = 0
                ppr_shortened = 0
                ppr_not_shortened = 0
                shortened_in_cluster = 0
                shortened_not_in_cluster = 0
                for nd in gm:
                    da = das[nd] if nd in das else INF
                    db = dbs[nd] if nd in dbs else INF
                    dm = dms[nd] if nd in dms else INF
                    shortened = not (min(da, db) == dm)
                    if shortened:
                        ppr_shortened += appr_vec[nd]
                    else:
                        ppr_not_shortened += appr_vec[nd]
                    if shortened and nd in cm:
                        cnt_shortened += 1
                        if nd in hidden:
                            cnt_hidden += 1
                        else:
                            cnt_not_hidden += 1
                    if shortened:
                        if nd in cm:
                            shortened_in_cluster += 1
                        else:
                            shortened_not_in_cluster += 1
                ratio_ppr_shortened.append(
                    ppr_shortened / (ppr_shortened + ppr_not_shortened))
                ratio_shortened_in_cluster.append(
                    shortened_in_cluster / (shortened_in_cluster + shortened_not_in_cluster) / len(cm))
                if cnt_shortened == 0:
                    continue
                ratio_hiddens.append(cnt_hidden / cnt_shortened)
                ratio_not_hiddens.append(cnt_not_hidden / cnt_shortened)
                print("ETA:", eta.eta((i + 1) / len(nodes_in_common)))
            # print(sum(ratio_hiddens) / len(ratio_hiddens))
            # print(sum(ratio_not_hiddens) / len(ratio_not_hiddens))
            list_ratio_ppr_shortened.append(ratio_ppr_shortened)
            list_ratio_shortened_in_cluster.append(ratio_shortened_in_cluster)
        result = []
        ratio_ppr_shortened1 = list_ratio_ppr_shortened[0]
        ratio_ppr_shortened2 = list_ratio_ppr_shortened[1]
        for i in range(len(ratio_ppr_shortened1)):
            try:
                ratio = ratio_ppr_shortened2[i] / ratio_ppr_shortened1[i]
            except ZeroDivisionError:
                continue
            result.append(ratio)
        filename1 = f"tmp/ratio_ppr_shortened-compare-{ganame}-{gbname}-{STRATEGIES[0]}-{STRATEGIES[1]}.png"
        draw_boxplot(
            list_ratio_ppr_shortened,
            labels=STRATEGIES,
            title=f"PPR of distance-shortened nodes\n(A: {ganame}, B: {gbname})",
            y_axis_title="sum of PPR",
            bottom=0,
            filename=filename1
        )
        filename2 = f"tmp/ratio_ppr_shortened-ratio-{ganame}-{gbname}-{STRATEGIES[0]}-{STRATEGIES[1]}.png"
        draw_boxplot(
            result,
            title=f"ratio of PPR of distance-shortened nodes ({STRATEGIES[1]} / {STRATEGIES[0]})\n(A: {ganame}, B: {gbname})",
            y_axis_title="ratio of PPR",
            filename=filename2
        )
        filenames1.append(filename1)
        filenames2.append(filename2)

        result = []
        ratio_shortened_in_cluster1 = list_ratio_shortened_in_cluster[0]
        ratio_shortened_in_cluster2 = list_ratio_shortened_in_cluster[1]
        for i in range(len(ratio_shortened_in_cluster1)):
            try:
                ratio = ratio_shortened_in_cluster2[i] / \
                    ratio_shortened_in_cluster1[i]
            except ZeroDivisionError:
                continue
            result.append(ratio)
        filename3 = f"tmp/ratio_shortened_in_cluster-compare-{ganame}-{gbname}-{STRATEGIES[0]}-{STRATEGIES[1]}.png"
        draw_boxplot(
            list_ratio_shortened_in_cluster,
            labels=STRATEGIES,
            title=f"%appearance in cluster over all distance-shortened nodes\n(A: {ganame}, B: {gbname})",
            y_axis_title="%appearance",
            bottom=0,
            filename=filename3
        )
        filename4 = f"tmp/ratio_shortened_in_cluster-ratio-{ganame}-{gbname}-{STRATEGIES[0]}-{STRATEGIES[1]}.png"
        draw_boxplot(
            result,
            title=f"ratio of %appearance in cluster of distance-shortened nodes ({STRATEGIES[1]} / {STRATEGIES[0]})\n(A: {ganame}, B: {gbname})",
            y_axis_title="ratio of %appearance",
            filename=filename4
        )
        filenames3.append(filename3)
        filenames4.append(filename4)
    fname = concatanate_images(filenames1, "tmp/compare")
    upload_to_imgbb(fname)
    fname = concatanate_images(filenames2, "tmp/compare-ratio")
    upload_to_imgbb(fname)
    fname = concatanate_images(filenames3, "tmp/in-cluster")
    upload_to_imgbb(fname)
    fname = concatanate_images(filenames4, "tmp/in-cluster-ratio")
    upload_to_imgbb(fname)


def compare_hiddens_am_datasets():
    PATHS = [
        ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        ("graph/0.1-0.01-3-100-normalorder.gr",
         "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        ("graph/Email-Enron.txt", "graph/web-edu.mtx")
    ]
    # STRATEGY = "NWF"
    STRATEGY = "allocation-rw"
    filenames1, filenames2, filenames3, filenames4 = [], [], [], []
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        fname1 = f"tmp/orig-{ganame}-{gbname}-{STRATEGY}.png"
        fname2 = f"tmp/merged-{ganame}-{gbname}-{STRATEGY}.png"
        fname3 = f"tmp/diff-{ganame}-{gbname}-{STRATEGY}.png"
        # fname4 = f"tmp/inandout-{ganame}-{gbname}-{STRATEGY}.png"
        filenames1.append(fname1)
        filenames2.append(fname2)
        filenames3.append(fname3)
        # filenames4.append(fname4)
    filename = concatanate_images(filenames1, "tmp/image")
    upload_to_imgbb(filename)
    filename = concatanate_images(filenames2, "tmp/image")
    upload_to_imgbb(filename)
    filename = concatanate_images(filenames3, "tmp/image")
    upload_to_imgbb(filename)
    # filename = concatanate_images(filenames4, "tmp/image")
    # upload_to_imgbb(filename)
    pass


def check_median_and_draw_box():
    # ga_path, gb_path = "graph/Email-Enron.txt", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/0.1-0.01-3-100-normalorder.gr", "graph/0.3-0.01-3-100-mixedorder.gr"
    # ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/web-edu.mtx", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/Email-Enron.txt", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"
    ga_path, gb_path = "graph/Rattus-DI.gr", "graph/Rattus-PA.gr"
    ga = read_graph(ga_path)
    gb = read_graph(gb_path)
    STRATEGY = "WAPPR"
    if STRATEGY == "WAPPR":
        nga, ngb = normalize_edge_weight(ga, gb, 16)
        gm = merge_two_graphs(nga, ngb, data=True)
    else:
        gm = merge_two_graphs(ga, gb, data=False)
    appra = APPR(ga)
    apprb = APPR(gb)
    apprm = APPR(gm)
    nodes_in_common = list(
        set([nd for nd in ga]).intersection([nd for nd in gb]))
    # nodes_in_common = random.sample(nodes_in_common, 101)
    data = []
    for seed in nodes_in_common:
        ca = set(appra.compute_appr(seed))
        cb = set(apprb.compute_appr(seed))
        if STRATEGY in ["NWF", "WAPPR"]:
            cm = set(apprm.compute_appr(seed))
        elif STRATEGY == "allocation-rw":
            cm = set(apprm.compute_allocation_appr(seed, ga, gb, c=1))
        hidden = cm - ca - cb
        data.append((len(hidden) / len(cm), seed))
    draw_boxplot([[x[0] for x in data]])
    print("median:", statistics.median(data))
    data.sort()
    print("3Q:", data[len(data)//4 * 3])
    pass


def draw_subgraph_around_specific_node():
    # ga_path, gb_path = "graph/Email-Enron.txt", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/0.1-0.01-3-100-normalorder.gr", "graph/0.3-0.01-3-100-mixedorder.gr"
    ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/web-edu.mtx", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/Email-Enron.txt", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"
    ganame = get_gname(ga_path)
    gbname = get_gname(gb_path)
    ga = read_graph(ga_path)
    gb = read_graph(gb_path)
    # gm = merge_two_graphs(ga, gb, data=False)
    gm = convert_two_graphs_to_digraph(ga, gb, 1)
    SEED = 2
    fnamea = "tmp/grapha.png"
    fnameb = "tmp/graphb.png"
    fnamem = "tmp/graphm.png"
    fnames = []
    print(SEED in ga, SEED in gb)
    try:
        draw_nhop(ga, SEED, 1, filename=fnamea)
        fnames.append(fnamea)
    except nx.exception.NetworkXError:
        print("A skipped")
        pass
    try:
        draw_nhop(gb, SEED, 1, filename=fnameb)
        fnames.append(fnameb)
    except nx.exception.NetworkXError:
        print("B skipped")
        pass
    draw_nhop(gm, SEED, 1, ga, gb, filename=fnamem)
    fnames.append(fnamem)
    fname = concatanate_images(
        fnames, "tmp/image", 3, 1, False)
    upload_to_imgbb(fname)


def count_deg_zero_nodes():
    # ga_path, gb_path = "graph/Email-Enron.txt", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/0.1-0.01-3-100-normalorder.gr", "graph/0.3-0.01-3-100-mixedorder.gr"
    # ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/web-edu.mtx", "graph/CA-GrQc.txt"
    ga_path, gb_path = "graph/Email-Enron.txt", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"
    ganame = get_gname(ga_path)
    gbname = get_gname(gb_path)
    ga = read_graph(ga_path)
    gb = read_graph(gb_path)
    ratio = 16
    STRATEGY = "WAPPR"
    if STRATEGY == "WAPPR":
        nga, ngb = normalize_edge_weight(ga, gb, ratio)
        gm = merge_two_graphs(nga, ngb, data=True)
    else:
        gm = merge_two_graphs(nga, ngb, False)

    appra = APPR(ga)
    apprb = APPR(gb)
    apprm = APPR(gm)
    nodes_in_common = list(
        set([nd for nd in ga]).intersection([nd for nd in gb]))
    cnt_only_a = 0
    cnt_only_b = 0
    cnt_both_a_b = 0
    others = 0
    for i, seed in enumerate(nodes_in_common):
        ca = appra.compute_appr(seed)
        cb = apprb.compute_appr(seed)
        cm = apprm.compute_appr(seed)
        hidden = set(cm) - set(ca) - set(cb)
        for nd in hidden:
            if nd in ga and nd in gb:
                cnt_both_a_b += 1
            elif nd in ga:
                cnt_only_a += 1
            elif nd in gb:
                cnt_only_b += 1
            else:
                others += 1
    data = [cnt_only_a, cnt_only_b, cnt_both_a_b]
    labels = ["only in A", "only in B", "both in A and in B"]
    sum_all = sum(data)
    for d in data:
        print(d / sum_all)
    print("others:", others)
    draw_pie_chart(
        data,
        labels,
        title=f"all hidden nodes are in A? B? or both?\n(A: {ganame}, B: {gbname}, c: {ratio})",
    )
    pass


def output_unique_nodes_am_datasets():
    paths = [
        "graph/Email-Enron.txt",
        "graph/CA-GrQc.txt",
        "graph/web-edu.mtx",
        "graph/socfb-Caltech36.mtx",
    ]
    labels = [""]
    records = []
    for ga_path in paths:
        ganame = get_gname(ga_path)
        ga = read_graph(ga_path)
        labels.append(f'{ganame} ({ga.number_of_nodes()})')
        record = [f'{ganame} ({ga.number_of_nodes()})']
        for gb_path in paths:
            gbname = get_gname(gb_path)
            gb = read_graph(gb_path)
            nodes_in_common = list(
                set([nd for nd in ga]).intersection([nd for nd in gb]))
            record.append((len(ga) - len(nodes_in_common)))
        records.append(record)

    export_table(records, labels)


def compute_dist_matrix():
    PATHS = [
        # ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        ("graph/0.1-0.01-3-100-normalorder.gr",
         "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        # ("graph/Email-Enron.txt", "graph/web-edu.mtx")
    ]
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        gname = f'{ganame}-{gbname}'
        ga = read_graph(ga_path)
        gb = read_graph(ga_path)
        gm = merge_two_graphs(ga, gb, False)
        nodes_in_common = sorted(list(
            set([nd for nd in ga]).intersection([nd for nd in gb])))
        nds = [nd for nd in gm]
        labels = ["src\\dest"] + nodes_in_common
        records = []
        eta = ETA()
        for i, src in enumerate(nodes_in_common):
            record = []
            lengths = nx.single_source_dijkstra_path_length(gm, src)
            record.append(src)
            for dst in nds:
                try:
                    record.append(lengths[dst])
                except KeyError:
                    record.append("inf")
            records.append(record)
            print(eta.eta((i+1)/len(nds)))
        export_to_csv(records, labels, f"tmp/dist/{gname}.txt")


def compare_hidden_to_none_with_seed():
    ga_path, gb_path = "graph/Email-Enron.txt", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/0.1-0.01-3-100-normalorder.gr", "graph/0.3-0.01-3-100-mixedorder.gr"
    # ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/web-edu.mtx", "graph/CA-GrQc.txt"
    # ga_path, gb_path = "graph/Email-Enron.txt", "graph/web-edu.mtx"
    # ga_path, gb_path = "graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"
    ganame = get_gname(ga_path)
    gbname = get_gname(gb_path)
    ga = read_graph(ga_path)
    gb = read_graph(gb_path)

    # STRATEGY1 = "allocation-rw"
    # STRATEGY2 = "NWF"
    STRATEGY1 = "NWF"
    STRATEGY2 = "allocation-rw"
    RATIO = 1

    if STRATEGY1 == "WAPPR" or STRATEGY2 == "WAPPR":
        print("strategy weight is currently unavailable")
        exit()

    gm = merge_two_graphs(ga, gb, data=False)
    appra = APPR(ga)
    apprb = APPR(gb)
    apprm1 = APPR(gm)
    apprm2 = APPR(gm)
    # print(list(
    #     set([nd for nd in ga]).intersection([nd for nd in gb])))
    if len(sys.argv) < 2:
        nodes_in_common = list(
            set([nd for nd in ga]).intersection([nd for nd in gb]))
        seed = random.choice(nodes_in_common)
    else:
        seed = int(sys.argv[1])

    ca = appra.compute_appr(seed)
    da = nx.single_source_dijkstra_path_length(ga, seed, weight=None)
    ppr_vec_a = appra.get_appr_vec()
    ppr_a = [k for k, _ in sorted(
        [(k, v / ga.degree(k, weight='weight')) for k, v in ppr_vec_a.items()], key=lambda x:x[1], reverse=True)]
    cb = apprb.compute_appr(seed)
    db = nx.single_source_dijkstra_path_length(gb, seed, weight=None)
    ppr_vec_b = apprb.get_appr_vec()
    ppr_b = [k for k, _ in sorted(
        [(k, v / gb.degree(k, weight='weight')) for k, v in ppr_vec_b.items()], key=lambda x:x[1], reverse=True)]
    if STRATEGY1 == "NWF":
        cm1 = apprm1.compute_appr(seed)
    elif STRATEGY1 == "allocation-rw":
        cm1 = apprm1.compute_allocation_appr(
            seed, ga, gb, c=RATIO)
    if STRATEGY2 == "NWF":
        cm2 = apprm2.compute_appr(seed)
    elif STRATEGY2 == "allocation-rw":
        cm2 = apprm2.compute_allocation_appr(
            seed, ga, gb, c=RATIO)
    dm = nx.single_source_dijkstra_path_length(gm, seed, weight=None)
    ppr_vec_m1 = apprm1.get_appr_vec()
    ppr_vec_m2 = apprm2.get_appr_vec()
    ppr_m1 = [k for k, _ in sorted(
        [(k, v / gm.degree(k, weight='weight')) for k, v in ppr_vec_m1.items()], key=lambda x:x[1], reverse=True)]
    ppr_m2 = [k for k, _ in sorted(
        [(k, v / gm.degree(k, weight='weight')) for k, v in ppr_vec_m2.items()], key=lambda x:x[1], reverse=True)]
    hidden = set(cm2) - set(ca) - set(cb)
    records = []
    labels = [
        "rank",
        # f"cm (hidden: { round(len(hidden) / len(cm) * 100, 1)}%)",
        f"node",
        "ab_both", "a_only", "b_only", "other",
        # f"rank_A+B-{STRATEGY1[:2]}", f"rank_A+B-{STRATEGY2[:2]}", "rank_A", "rank_B",
        # f"DNPPR_A+B-{STRATEGY1[:2]}", f"DNPPR_A+B-{STRATEGY2[:2]}", "DNPPR_A", "DNPPR_B",
        # "dist_A+B", "dist_A", "dist_B",
        # "deg_A+B", "deg_A", "deg_B",
        # "#paths_A+B", "#paths_A", "#paths_B"
    ]
    # print(seed in ga, seed in gb, file=sys.stderr)
    rank = 0
    CLUSTER = "A"
    # CLUSTER = "B"
    # CLUSTER = "A+B"
    # if CLUSTER == "A":
    #     c = ca
    # elif CLUSTER == "B":
    #     c = cb
    # elif CLUSTER == "A+B":
    #     c = cm
    c = cm1

    for nd in c:
        rank += 1
        if CLUSTER == "A+B":
            if nd in ca or nd in cb:
                continue
        ndm = f'{nd}'
        try:
            score_m1 = ppr_vec_m1[nd] / gm.degree(nd, 'weight')
        except:
            score_m1 = 0
        try:
            score_m2 = ppr_vec_m2[nd] / gm.degree(nd, 'weight')
        except:
            score_m2 = 0
        try:
            score_a = ppr_vec_a[nd] / ga.degree(nd, 'weight')
        except:
            score_a = 0
        try:
            score_b = ppr_vec_b[nd] / gb.degree(nd, 'weight')
        except:
            score_b = 0
        dist_m = dm[nd]
        if nd in da:
            dist_a = da[nd]
        elif nd in ga:
            dist_a = "inf"
        else:
            dist_a = "not exists"
        if nd in db:
            dist_b = db[nd]
        elif nd in gb:
            dist_b = "inf"
        else:
            dist_b = "not exists"
        try:
            rank_in_a = ppr_a.index(nd) + 1
            if rank_in_a > len(ca):
                rank_in_a = f'{rank_in_a} (out)'
        except:
            rank_in_a = "out"
        try:
            rank_in_b = ppr_b.index(nd) + 1
            if rank_in_b > len(cb):
                rank_in_b = f'{rank_in_b} (out)'
        except:
            rank_in_b = "out"
        try:
            rank_in_m1 = ppr_m1.index(nd) + 1
            if rank_in_m1 > len(cm1):
                rank_in_m1 = f'{rank_in_m1} (out)'
        except:
            rank_in_m1 = "out"
        try:
            rank_in_m2 = ppr_m2.index(nd) + 1
            if rank_in_m2 > len(cm2):
                rank_in_m2 = f'{rank_in_m2} (out)'
        except:
            rank_in_m2 = "out"
        try:
            dega = ga.degree(nd)
        except:
            dega = 0
        try:
            degb = gb.degree(nd)
        except:
            degb = 0
        degm = gm.degree(nd, weight='weight')
        try:
            num_paths_a = len(
                [_ for _ in nx.all_shortest_paths(ga, seed, nd, weight=None)])
        except:
            num_paths_a = 0
        try:
            num_paths_b = len(
                [_ for _ in nx.all_shortest_paths(gb, seed, nd, weight=None)])
        except:
            num_paths_b = 0
        path = [""] * max(2, dist_m)
        a_only, b_only, ab_both, other = 0, 0, 0, 0
        try:
            all_paths = [p for p in nx.all_shortest_paths(
                gm, seed, nd, weight=None)]
            num_paths_m = len(all_paths)
            cnt_a_only = 0
            cnt_b_only = 0
            cnt_ab_both = 0
            cnt_cnt_rwer_can_reach_only_on_mutual_edges = 0
            for p in all_paths:
                a_flag, b_flag = False, False
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    if ga.has_edge(u, v) and not gb.has_edge(u, v):
                        a_flag = True
                    elif not ga.has_edge(u, v) and gb.has_edge(u, v):
                        b_flag = True
                    elif ga.has_edge(u, v) and gb.has_edge(u, v):
                        pass
                    else:
                        print("This is impossible")
                        exit(0)
                if a_flag and b_flag:
                    cnt_ab_both += 1
                elif a_flag:
                    cnt_a_only += 1
                elif b_flag:
                    cnt_b_only += 1
                else:
                    cnt_rwer_can_reach_only_on_mutual_edges += 1
            a_only = cnt_a_only / num_paths_m
            b_only = cnt_b_only / num_paths_m
            ab_both = cnt_ab_both / num_paths_m
            other = cnt_cnt_rwer_can_reach_only_on_mutual_edges / num_paths_m
        except:
            num_paths_m = 0
        # if dist_a != dist_m and dist_b != dist_m:
        #     continue
        records.append([
            f"{rank} / {len(c)}", ndm,
            ab_both, a_only, b_only, other,
            # rank_in_m1, rank_in_m2, rank_in_a, rank_in_b,
            # float(f"{score_m1:.03g}"), float(f"{score_m2:.03g}"),  float(
            #     f"{score_a:.03g}"), float(f"{score_b:.03g}"),
            # dist_m, dist_a, dist_b,
            # degm, dega, degb,
            # num_paths_m, num_paths_a, num_paths_b,
        ])
    # records.sort(key=lambda x: x[11])
    export_table(records, labels)
    # export_to_csv(records, labels)
    # export_to_google_sheets(
    #     records,
    #     labels,
    #     title=f"cluster in {CLUSTER} (A: {ganame}, B: {gbname}, seed: {seed}, c: {RATIO})",
    #     template_id="1uzK51pSicVp-2lLcNzCOIN4IPjwjJQdodDEylvjhbtg",
    # )


def compare_strategy_with_another():
    PATHS = [
        ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        # ("graph/0.1-0.01-3-100-normalorder.gr",
        #  "graph/0.3-0.01-3-100-mixedorder.gr"),
        # ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        # ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        # ("graph/Email-Enron.txt", "graph/web-edu.mtx")
    ]
    list_csize_errors, list_ros, list_hsize_errors, list_hratio_errors = [], [], [], []
    labels = []
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)

        # STRATEGY1 = "allocation-rw"
        # STRATEGY2 = "NWF"
        STRATEGY1 = "NWF"
        STRATEGY2 = "allocation-rw"
        RATIO = 1.0

        if STRATEGY1 == "WAPPR" or STRATEGY2 == "WAPPR":
            print("strategy weight is currently unavailable")
            exit()

        cas = read_clusters(f"cluster/{ganame}/appr.txt")
        cbs = read_clusters(f"cluster/{gbname}/appr.txt")
        cms1 = read_clusters(
            f"cluster/{ganame}-{gbname}/{STRATEGY1}-{RATIO}.txt")
        cms2 = read_clusters(
            f"cluster/{ganame}-{gbname}/{STRATEGY2}-{RATIO}.txt")

        gm = merge_two_graphs(ga, gb, data=False)
        nodes_in_common = list(
            set([nd for nd in ga]).intersection([nd for nd in gb]))
        csize_errors, ros, hsize_errors, hratio_errors = [], [], [], []
        ros_seeds, csizes_seeds = [], []
        records = []
        for seed in nodes_in_common:
            ca = cas[seed]
            cb = cbs[seed]
            cm1 = cms1[seed]
            cm2 = cms2[seed]
            hidden1 = cm1 - ca - cb
            hidden2 = cm2 - ca - cb
            csize_error = (len(cm2) - len(cm1)) / len(cm1)
            csize_errors.append(csize_error)
            csizes_seeds.append((len(cm1), seed))
            try:
                ro = len(hidden1.intersection(hidden2)) / \
                    len(hidden1.union(hidden2))
                ros.append(ro)
                ros_seeds.append((ro, seed))
            except ZeroDivisionError:
                ro = "inf"
                pass
            try:
                hsize_error = (len(hidden2) - len(hidden1)) / len(hidden1)
                hsize_errors.append(hsize_error)
            except ZeroDivisionError:
                hsize_error = "inf"
                pass
            try:
                hratio_error = len(hidden2) / len(cm2) - \
                    len(hidden1) / len(cm1)
                hratio_errors.append(hratio_error)
            except ZeroDivisionError:
                hratio_error = "inf"
                pass
            records.append((seed, csize_error, ro, hsize_error, hratio_error, len(
                cm1), len(cm2), len(hidden1), len(hidden2)))
        list_csize_errors.append(csize_errors)
        list_ros.append(ros)
        list_hsize_errors.append(hsize_errors)
        list_hratio_errors.append(hratio_errors)
        labels.append(f'A: {ganame}\nB: {gbname}')
        # try:
        #     print(statistics.median(ros_seeds))
        # except TypeError:
        #     print(statistics.median(ros_seeds[:-1]))
        csizes_seeds.sort()
        export_to_google_sheets(
            records,
            ["seed", "csize_error", "ro", "hsize_error", "hratio_error",
                "cm1_size", "cm2_size", "hidden1_size", "hidden2_size"]
        )
    draw_boxplot(
        list_csize_errors,
        labels=labels,
        rotation=30,
        title=f"diff of cluster size ({STRATEGY2} - {STRATEGY1}) / {STRATEGY1}",
        filename="tmp/csize_errors.png"
    )
    draw_boxplot(
        list_ros,
        labels=labels,
        rotation=30,
        title=f"Relative Overlap of communities ({STRATEGY1}, {STRATEGY2})",
        filename="tmp/ros.png"
    )
    draw_boxplot(
        list_hsize_errors,
        labels=labels,
        rotation=30,
        title=f"diff of #hiddens ({STRATEGY2} - {STRATEGY1}) / {STRATEGY1}",
        filename="tmp/hsize_errors.png"
    )
    draw_boxplot(
        list_hratio_errors,
        labels=labels,
        rotation=30,
        title=f"diff of %hiddens ({STRATEGY2} - {STRATEGY1})",
        filename="tmp/hratio_errors.png"
    )


def exp_rwer_at_switching_points():
    PATHS = [
        ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        ("graph/0.1-0.01-3-100-normalorder.gr",
         "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        ("graph/Email-Enron.txt", "graph/web-edu.mtx")
    ]
    COMPUTE_THE_VALUES = False
    RATIOS = [1/1024 * 2 ** (i) for i in range(21)]
    strategy = "dynamic-rw"
    # strategy = "allocation-rw"
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)

        rw_x_axis_title = "user-input ratio c"
        rw_y_axis_title = f"#RWer"
        rw_title = "Total #RWers that pass the graph's edges"
        ratio_title = "resulting ratio of #Rwer"
        ratio_x_axis_title = "user-input ratio c"
        ratio_y_axis_title = "ratio B/A"
        ratio_labels = ["#RWers", "UI"]
        error_title = "resulting error of #Rwer"
        error_x_axis_title = "user-input ratio c"
        error_y_axis_title = "error (B/A) / c"
        error_labels = ["#RWers"]

        if COMPUTE_THE_VALUES:
            eta = ETA()
            for tpr_idx, transition_probability_ratio in enumerate(RATIOS):
                filename_separate = f"tmp/separate-{ganame}-{gbname}-{strategy}-{transition_probability_ratio}-rweras-rwerbs.csv"
                filename_rwers = f"tmp/merged-{ganame}-{gbname}-{strategy}-{transition_probability_ratio}-rwers.csv"
                appra = APPR(ga)
                apprb = APPR(gb)
                gm = merge_two_graphs(ga, gb, data=False)
                apprm = APPR(gm)
                nodes_in_common = list(
                    set([nd for nd in ga]).intersection([nd for nd in gb]))
                rwers = []
                rweras, rwerbs = [], []
                NUM_SAMPLES = 1000
                nodes_in_common = list(
                    set([nd for nd in ga]).intersection([nd for nd in gb]))
                if NUM_SAMPLES and len(nodes_in_common) > NUM_SAMPLES:
                    nodes_in_common = random.sample(
                        nodes_in_common, NUM_SAMPLES)
                seeds = nodes_in_common
                for i, seed in enumerate(seeds):
                    ca = set(appra.compute_appr(seed))
                    cb = set(apprb.compute_appr(seed))
                    if strategy == "dynamic-rw":
                        cm, _, _, rwera, rwerb = apprm.compute_dynamic_appr(
                            seed, ga, gb, c=transition_probability_ratio, data=True)
                    elif strategy == "allocation-rw":
                        cm, rwera, rwerb = apprm.compute_allocation_appr(
                            seed, ga, gb, c=transition_probability_ratio, data=True)
                    cm = set(cm)
                    # print(seed, len(two), len(seven)
                    rwers.append(rwera - rwerb)
                    rweras.append(rwera)
                    rwerbs.append(rwerb)
                    if transition_probability_ratio == 1 and rwerb / rwera > 2:
                        print(rwera, rwerb)
                    vm = apprm.get_appr_vec()
                    for nd in gm:
                        if nd not in vm:
                            vm[nd] = 0

                f = open(filename_separate, 'w')
                f.write(f"rweras,rwerbs\n")
                for i in range(len(rweras)):
                    f.write(f"{rweras[i]},{rwerbs[i]}\n")
                f.close()

                f = open(filename_rwers, 'w')
                f.write(f"rwers\n")
                for i in range(len(rwers)):
                    f.write(f"{rwers[i]}\n")
                f.close()
                print(
                    "eta:",
                    eta.eta((tpr_idx+1) / len(RATIOS))
                )

        rwer_transition_ratio(
            ganame,
            gbname,
            RATIOS,
            strategy,
            rw_x_axis_title,
            rw_y_axis_title,
            rw_title,
            ratio_title,
            ratio_x_axis_title,
            ratio_y_axis_title,
            ratio_labels,
            error_title,
            error_x_axis_title,
            error_y_axis_title,
            error_labels,
            send_to_slack=False,
        )


def rwer_transition_ratio(
    ganame: str,
    gbname: str,
    transition_probability_ratios,
    strategy: str,
    rw_x_axis_title: str,
    rw_y_axis_title: str,
    rw_title: str,
    ratio_title: str,
    ratio_x_axis_title: str,
    ratio_y_axis_title: str,
    ratio_labels: str,
    error_title: str,
    error_x_axis_title: str,
    error_y_axis_title: str,
    error_labels: str,
    send_to_slack: bool = True,
):
    med_rweras, med_rwerbs, ratio_rwers = [], [], []
    q1_rweras, q1_rwerbs, q1_ratio_rwers = [], [], []
    q3_rweras, q3_rwerbs, q3_ratio_rwers = [], [], []
    for transition_probability_ratio in transition_probability_ratios:
        filename = f"tmp/separate-{ganame}-{gbname}-{strategy}-{transition_probability_ratio}-rweras-rwerbs.csv"
        rweras, rwerbs, ratio_rwer = [], [], []
        f = open(filename)
        for line in f.readlines()[1:]:
            rwera, rwerb = map(float, line.split(','))
            rweras.append(rwera)
            rwerbs.append(rwerb)
            if rwera != 0:
                ratio_rwer.append(rwerb / rwera)

        med_rwera = statistics.median(rweras)
        med_rwerb = statistics.median(rwerbs)
        med_ratio_rwer = statistics.median(ratio_rwer)
        med_rweras.append(med_rwera)
        med_rwerbs.append(med_rwerb)
        ratio_rwers.append(med_ratio_rwer)
        q1_rweras.append(med_rwera - np.percentile(rweras, 25))
        q1_rwerbs.append(med_rwerb - np.percentile(rwerbs, 25))
        q1_ratio_rwers.append(med_ratio_rwer - np.percentile(ratio_rwer, 25))
        q3_rweras.append(np.percentile(rweras, 75) - med_rwera)
        q3_rwerbs.append(np.percentile(rwerbs, 75) - med_rwerb)
        q3_ratio_rwers.append(np.percentile(ratio_rwer, 75) - med_ratio_rwer)
    filename = f"tmp/rwers-{ganame}-{gbname}-median.png"
    draw_chart(
        transition_probability_ratios,
        [med_rweras, med_rwerbs],
        list_errorbars=[[q1_rweras, q3_rweras], [q1_rwerbs, q3_rwerbs]],
        labels=[f"A: {ganame}", f"B: {gbname}"],
        title=f"{rw_title}\n(A: {ganame}, B: {gbname}, strategy: {strategy})",
        x_axis_title=rw_x_axis_title,
        y_axis_title=rw_y_axis_title,
        left=transition_probability_ratios[0],
        bottom=0,
        # top=1,
        xscale="log",
        filename=filename
    )
    image_url = upload_to_imgbb(filename)
    if send_to_slack:
        notify_slack(
            title=f"{rw_title}\n(A: {ganame}, B: {gbname}, strategy: {strategy})",
            result=image_url
        )

    filename = f"tmp/ratios-{ganame}-{gbname}-median.png"
    draw_chart(
        transition_probability_ratios,
        [ratio_rwers],
        list_errorbars=[[q1_ratio_rwers, q3_ratio_rwers]],
        # labels=[f"#RWers", f"UI"],
        labels=ratio_labels,
        title=f"{ratio_title}\n(A: {ganame}, B: {gbname}, strategy: {strategy})",
        x_axis_title=ratio_x_axis_title,
        y_axis_title=ratio_y_axis_title,
        left=transition_probability_ratios[0],
        bottom=10 ** (-3),
        # top=1,
        xscale="log",
        yscale="log",
        filename=filename
    )
    image_url = upload_to_imgbb(filename)
    if send_to_slack:
        notify_slack(
            title=f"{ratio_title} (A: {ganame}, B: {gbname}, strategy: {strategy})",
            result=image_url
        )

    filename = f"tmp/errors-{ganame}-{gbname}-median.png"
    error_rwers = [ratio_rwers[i] / transition_probability_ratios[i]
                   for i, _ in enumerate(ratio_rwers)]

    q1_error_rwers = [q1_ratio_rwers[i] / transition_probability_ratios[i]
                      for i, _ in enumerate(ratio_rwers)]
    q3_error_rwers = [q3_ratio_rwers[i] / transition_probability_ratios[i]
                      for i, _ in enumerate(ratio_rwers)]
    draw_chart(
        transition_probability_ratios,
        [error_rwers],
        list_errorbars=[q1_error_rwers, q3_error_rwers],
        labels=error_labels,
        # title=f"ratio of #RWers and UI (A: {id_a}, B: {id_b}, strategy: {strategy})",
        title=f"{error_title} (A: {ganame}, B: {gbname}, strategy: {strategy})",
        x_axis_title=error_x_axis_title,
        y_axis_title=error_y_axis_title,
        left=transition_probability_ratios[0],
        bottom=0,
        # top=1,
        xscale="log",
        filename=filename
    )
    image_url = upload_to_imgbb(filename)
    if send_to_slack:
        notify_slack(
            title=f"{ratio_title} (A: {ganame}, B: {gbname}, strategy: {strategy})",
            result=image_url
        )


def cluster_info():
    PATHS = [
        ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        ("graph/0.1-0.01-3-100-normalorder.gr",
         "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        ("graph/Email-Enron.txt", "graph/web-edu.mtx")
    ]
    filenames = []
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)
        cas = read_clusters(f"cluster/{ganame}/appr.txt")
        cbs = read_clusters(f"cluster/{gbname}/appr.txt")
        nodes_in_common = list(
            set([nd for nd in ga]).intersection([nd for nd in gb]))
        casizes = [len(cas[nd]) for nd in nodes_in_common]
        cbsizes = [len(cbs[nd]) for nd in nodes_in_common]
        list_csizes = [casizes, cbsizes]
        labels = [ganame, gbname]
        fname = f'tmp/{ganame}-{gbname}-cluster_sizes_nodes_in_common.png'
        draw_boxplot(list_csizes, title="cluster size",
                     y_axis_title="size", labels=labels, bottom=0, filename=fname)
        filenames.append(fname)
    fname = concatanate_images(filenames, "tmp/image", 3, 2)
    upload_to_imgbb(fname)


def convert_two_graphs_to_digraph(ga: nx.Graph, gb: nx.Graph, ratio: float, one_based: bool = True) -> nx.DiGraph:
    # ga and gb needs to be undirected and unweighted
    dg = nx.DiGraph()
    dg.add_nodes_from([nd for nd in ga])
    dg.add_nodes_from([nd for nd in gb])
    ma = ga.number_of_edges()
    mb = gb.number_of_edges()
    nab = dg.number_of_nodes()

    if one_based:
        norm = (2 * ma + 2 * ratio * mb + (1 + ratio)
                * nab) / (2 * ma + 2 * mb)
    else:
        norm = (2 * ma + 2 * ratio * mb) / (2 * ma + 2 * mb)

    for nd in dg:
        da = ga.degree(nd) if nd in ga else 0
        db = gb.degree(nd) if nd in gb else 0
        if nd in ga and nd in gb:
            if one_based:
                d = (da + ratio * db + 1 + ratio) / norm
            else:
                d = (da + ratio * db) / norm
            ws = defaultdict(float)
            wa = d / (1 + ratio) / da
            wb = d * ratio / (1 + ratio) / db
            for nbr in ga.neighbors(nd):
                ws[nbr] += wa
            for nbr in gb.neighbors(nd):
                ws[nbr] += wb
            for nbr, w in ws.items():
                dg.add_edge(nd, nbr, weight=w)
        elif nd in ga:
            if one_based:
                d = (1 + ratio + da) / norm
            else:
                d = da / norm
            for nbr in ga.neighbors(nd):
                dg.add_edge(nd, nbr, weight=d/da)
        elif nd in gb:
            if one_based:
                d = (1 + ratio + ratio * db) / norm
            else:
                d = ratio * db / norm
            for nbr in gb.neighbors(nd):
                dg.add_edge(nd, nbr, weight=d/db)
            pass
        else:
            print("error")
            exit(0)
    return dg


# def convert_two_graphs_to_digraph_one_based(ga: nx.Graph, gb: nx.Graph, ratio: float) -> nx.DiGraph:
#     # ga and gb needs to be undirected and unweighted
#     dg = nx.DiGraph()
#     dg.add_nodes_from([nd for nd in ga])
#     dg.add_nodes_from([nd for nd in gb])
#     ma = ga.number_of_edges()
#     mb = gb.number_of_edges()
#     nab = dg.number_of_nodes()

#     norm = (2 * ma + 2 * ratio * mb + (1 + ratio)
#             * nab) / (2 * ma + 2 * mb)
#     for nd in dg:
#         da = ga.degree(nd) if nd in ga else 0
#         db = gb.degree(nd) if nd in gb else 0
#         if nd in ga and nd in gb:
#             d = (da + ratio * db + 1 + ratio) / norm
#             ws = defaultdict(float)
#             wa = d / (da + ratio * db)
#             wb = d * ratio / (da + ratio * db)
#             for nbr in ga.neighbors(nd):
#                 ws[nbr] += wa
#             for nbr in gb.neighbors(nd):
#                 ws[nbr] += wb
#             for nbr, w in ws.items():
#                 dg.add_edge(nd, nbr, weight=w)
#         elif nd in ga:
#             d = (1 + ratio + da) / norm
#             for nbr in ga.neighbors(nd):
#                 dg.add_edge(nd, nbr, weight=d/da)
#         elif nd in gb:
#             d = (1 + ratio + ratio * db) / norm
#             for nbr in gb.neighbors(nd):
#                 dg.add_edge(nd, nbr, weight=d/db)
#             pass
#         else:
#             print("error")
#             exit(0)
#     return dg


def normalize_axes(axes: Dict[int, tuple]):
    max_latitude = max(axes.values(), key=lambda t: t[0])[0]
    min_latitude = min(axes.values(), key=lambda t: t[0])[0]
    max_longitude = max(axes.values(), key=lambda t: t[1])[1]
    min_longitude = min(axes.values(), key=lambda t: t[1])[1]
    center_latitude = (max_latitude + min_latitude)/2
    center_longitude = (max_longitude + min_longitude)/2
    divisor_latitude = (max_latitude - min_latitude)/2
    divisor_longitude = (max_longitude - min_longitude)/2
    pos = {}
    for nd, axis in axes.items():
        pos[nd] = (
            (axis[0] - center_latitude) / divisor_latitude,
            -1 * (axis[1] - center_longitude) / divisor_longitude,
        )
    return pos


def main2():
    # exp_sbm_all_nodes()
    # exp_sbm_one_node_merged()
    # exp_sbm_sameedge_differentcsize()
    # exp_sbm_samedensity_differentcsize()
    # exp_sbm_three_nodes()
    # exp_radial_graph()
    # exp_radial_graph_dist_to_appr()
    # exp_two_radial_graph_dist_to_appr()
    # exp_two_realgraphs()
    # wiki_graph_exp()
    # thenterMTG0921()
    # rw_point()
    # exp_ppr_distribution()
    # exp_unique_frequency()
    # exp_uiu_rwer_comparison()
    # exp_uiu_rwer_comparison_bw_strategies()
    # thenterMTG0930()
    # exp_rwer_control()
    # exp_rwer_transition_ratio()
    # compute_appr("graph/com-dblp.ungraph.txt")
    # create_newinfo_figs()
    # create_hidden_info_figs()
    # compare_overlap_bw_strategies()
    # compare_ppr_methods()
    # draw_subgraph_around_specific_node()
    # check_hidden()
    # check_hidden2()
    # count_deg_zero_nodes()
    # compare_hiddens_am_datasets()
    # output_unique_nodes_am_datasets()
    # compute_dist_matrix()
    # compare_hidden_to_none_with_seed()
    # compare_strategy_with_another()
    # exp_rwer_at_switching_points()
    # cluster_info()
    pass
