from japanmap import picture, get_data, pref_map, pref_names, pref_code
from louvain import louvain_communitie
from fileinput import filename
from motif_cluster import MotifCluster
from local_motif_cluster import MAPPR
from random import sample, shuffle
from collections import deque, defaultdict
from spectral import compute_global_cluster
from read_graph import *
import copy
from time import sleep
from louvain import louvain_partitions
from local_motif_cluster import MAPPR


def motifcluster():
    g = read_digraph(sys.argv[1])
    mc = MotifCluster(g, M7)
    clustering = mc.get_motif_clustering()


def localmotifcluster():
    g = read_graph(sys.argv[1])
    start = time.time()
    mappr = MAPPR(g, clique3)
    end = time.time()
    mappr_base = end - start

    alpha = 0.98
    eps = 0.0001
    largest_component = mappr.get_processed_graph().subgraph(
        max(nx.connected_components(mappr.get_processed_graph()), key=len))
    print("seed,MAPPR size,MAPPR MC,MAPPR time,fGraph size,fMAPPR size,fMAPPR MC,fMAPPR time,hit,hit ratio")
    for seed in largest_component:
        # print("seed:", seed)
        start = time.time()
        ma = mappr.compute_appr(seed, alpha, eps / mappr.get_total_vol()
                                * mappr.get_processed_graph().number_of_nodes())
        # print(ma)
        # print("motif conductance:", mappr.compute_motif_clustering(ma))
        end = time.time()
        mappr_time = end-start + mappr_base

        start = time.time()
        appr = APPR(g)
        alpha = 0.98
        eps = 0.0001
        appr.compute_appr(seed, alpha, eps / appr.get_total_vol()
                          * appr.get_graph().number_of_nodes()/10)
        a = appr.top_appr()
        if len(ma) == 0:
            continue

        sub = g.subgraph(a)
        mappr2 = MAPPR(sub, clique3)
        ma2 = mappr2.compute_appr(seed, alpha, eps / mappr2.get_total_vol()
                                  * mappr2.get_processed_graph().number_of_nodes(), total_vol=mappr2.get_total_vol())
        # print("motif conductance:", mappr.compute_motif_clustering(ma2))
        end = time.time()
        fmappr_time = end-start

        ma2_set = set(ma2)
        hit, miss = 0, 0
        for m in ma:
            if m in ma2_set:
                hit += 1
            else:
                miss += 1
        # print("MAPPR: ", len(ma),
        #       ", fMAPPR: ", len(ma2_set),
        #       ", hit: ", hit,
        #       ", hit ratio: ", hit / len(ma),
        #       ", miss: ",  miss,
        #       ", miss ratio: ", miss / len(ma),
        #       sep='')
        # print(ma)
        print(seed,
              len(ma), mappr.compute_motif_clustering(ma), mappr_time,
              len(sub), len(ma2_set), mappr.compute_motif_clustering(
                  ma2), fmappr_time,
              hit, hit / len(ma), sep=',')


def exp_localmotifcluster():
    graph_names = [
        "graph/example.gr",
        "graph/undirected.gr",
        "graph/fb-caltech-connected.gr",
        "graph/email-Eu-core.txt",
        "graph/ca-grqc.gr",
        "graph/fb-pages-company.gr",
        "graph/email-enron.gr",
        "graph/soc-slashdot.gr",
        "graph/soc-Slashdot0902.txt",
        "graph/com-amazon-connected.gr",
    ]
    for graph_name in graph_names:
        g = read_graph(graph_name)
        print(graph_name)
        mappr = MAPPR(g, clique3, 11)
        print()


def exp_ppr_mppr():
    g = read_graph(sys.argv[1])
    mappr = MAPPR(g, clique3)
    mg = mappr.get_processed_graph()

    seed = 4
    ppr = nx.pagerank(g, personalization={seed: 1}, alpha=0.8, tol=0.0001)
    mppr = nx.pagerank(mg, personalization={seed: 1}, alpha=0.8, tol=0.0001)

    result = []
    for nd, p in ppr.items():
        if g.degree(nd) == 0:
            result.append((nd, 0))
        else:
            result.append((nd, p / g.degree(nd)))

    result.sort(reverse=True, key=lambda x: x[1])
    rank = {}
    for i, x in enumerate(result):
        rank[x[0]] = i + 1

    result = []
    for nd, p in mppr.items():
        if mg.degree(nd, weight="weight") == 0:
            result.append((nd, 0))
        else:
            result.append((nd, p / mg.degree(nd, weight="weight")))
    result.sort(reverse=True, key=lambda x: x[1])

    dist = nx.single_source_dijkstra_path_length(g, seed)
    mdist = nx.single_source_dijkstra_path_length(mg, seed, weight=None)

    print("|node|MPPRwg rank|PPRd rank|diff|mdeg|deg|mdist|dist|MPPR|PPR|")
    print("|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    for i, x in enumerate(result[:20]):
        nd = x[0]
        rank_diff = rank[nd] - i - 1
        if rank_diff > 0:
            rank_diff = "+" + str(rank_diff)
        elif rank_diff == 0:
            rank_diff = "±0"
        if nd not in dist:
            d = "∞"
        else:
            d = dist[nd]
        if nd not in mdist:
            md = "∞"
        else:
            md = mdist[nd]
        print("|", nd, "|",
              i + 1, "|", rank[nd], "|", rank_diff, "|",
              mg.degree(nd, weight="weight"), "|", g.degree(nd), "|",
              md, "|", d, "|",
              x[1], "|", ppr[nd], "|", sep='')


def exp_ground_truth():
    if len(sys.argv) != 3:
        print("usage: python3", sys.argv[0],
              "<path to graph file>", "<path to community file>")
        return
    g = read_graph(sys.argv[1])
    all_communities = read_community(sys.argv[2])
    communities = []
    for c in all_communities:
        # if 10 <= len(c) <= 200:
        communities.append(c)

    mappr = MAPPR(g, clique3)
    alpha = 0.98
    eps = 0.0001
    precisions, recalls, f1s = [], [], []
    computed_clusters = dict()
    for idx, c in enumerate(communities):
        max_p, max_r, max_f1 = 0, 0, 0
        for seed in c:
            if seed in computed_clusters:
                ma = computed_clusters[seed]
            else:
                ma = tuple(mappr.compute_appr(seed, alpha, eps / mappr.get_total_vol()
                                              * mappr.get_processed_graph().number_of_nodes()))
                computed_clusters[seed] = ma
            precision, recall, f1 = compute_precision_recall_f1(ma, c, seed)
            if f1 > max_f1:
                max_p, max_r, max_f1 = precision, recall, f1
        print(idx, max_p, max_r, max_f1)
        precisions.append(max_p)
        recalls.append(max_r)
        f1s.append(max_f1)
    print("precision:", sum(precisions)/len(precisions))
    print("recall:", sum(recalls)/len(recalls))
    print("F1:", sum(f1s)/len(f1s))


def exp_all_5000_amazon():
    g = read_graph("graph/com-amazon.ungraph.txt")
    communities = read_community("community/com-amazon.top5000.cmty.txt")
    f = open("output/com-amazon.top5000.output.txt", "w")

    mappr = MAPPR(g, clique3)
    alpha = 0.98
    eps = 0.0001
    precisions, recalls, f1s = [], [], []
    computed_clusters = dict()
    f.write("idx,precision,recall,f1\n")
    for idx, c in enumerate(communities):
        max_p, max_r, max_f1 = 0, 0, 0
        for seed in c:
            if seed in computed_clusters:
                ma = computed_clusters[seed]
            else:
                ma = tuple(mappr.compute_appr(seed, alpha, eps / mappr.get_total_vol()
                                              * mappr.get_processed_graph().number_of_nodes()))
                computed_clusters[seed] = ma
            precision, recall, f1 = compute_precision_recall_f1(ma, c, seed)
            if f1 > max_f1:
                max_p, max_r, max_f1 = precision, recall, f1
        f.write(str(idx) + "," + str(max_p) + "," +
                str(max_r) + "," + str(max_f1) + "\n")
        print(idx, max_p, max_r, max_f1)
        precisions.append(max_p)
        recalls.append(max_r)
        f1s.append(max_f1)


def exp_all_5000_amazon_edited():
    g = read_graph("graph/com-amazon.ungraph.txt")
    communities = read_community("community/com-amazon.top5000.cmty.txt")
    f = open("output/com-amazon.top5000.output.edited.txt", "w")

    mappr = MAPPR(g, clique3)
    alpha = 0.98
    eps = 0.0001
    precisions, recalls, f1s = [], [], []
    computed_clusters = dict()
    f.write("idx,precision,recall,f1\n")
    for idx, c in enumerate(communities):
        max_p, max_r, max_f1 = 0, 0, 0
        for seed in c:
            if seed in computed_clusters:
                ma = computed_clusters[seed]
            else:
                ma = tuple(mappr.compute_appr(seed, alpha, eps / mappr.get_total_vol()
                                              * mappr.get_processed_graph().number_of_nodes(), original=False))
                computed_clusters[seed] = ma
            precision, recall, f1 = compute_precision_recall_f1(ma, c, seed)
            if f1 > max_f1:
                max_p, max_r, max_f1 = precision, recall, f1
        f.write(str(idx) + "," + str(max_p) + "," +
                str(max_r) + "," + str(max_f1) + "\n")
        print(idx, max_p, max_r, max_f1)
        precisions.append(max_p)
        recalls.append(max_r)
        f1s.append(max_f1)


def exp_all_5000_dblp():
    g = read_graph("graph/com-dblp.ungraph.txt")
    communities = read_community("community/com-dblp.top5000.cmty.txt")
    f = open("output/com-dblp.top5000.output.txt", "w")

    mappr = MAPPR(g, clique3)
    alpha = 0.98
    eps = 0.0001
    precisions, recalls, f1s = [], [], []
    computed_clusters = dict()
    f.write("idx,precision,recall,f1\n")
    for idx, c in enumerate(communities):
        max_p, max_r, max_f1 = 0, 0, 0
        for seed in c:
            if seed in computed_clusters:
                ma = computed_clusters[seed]
            else:
                ma = tuple(mappr.compute_appr(seed, alpha, eps / mappr.get_total_vol()
                                              * mappr.get_processed_graph().number_of_nodes(), original=True))
                computed_clusters[seed] = ma
            precision, recall, f1 = compute_precision_recall_f1(ma, c, seed)
            if f1 > max_f1:
                max_p, max_r, max_f1 = precision, recall, f1
        f.write(str(idx) + "," + str(max_p) + "," +
                str(max_r) + "," + str(max_f1) + "\n")
        print(idx, max_p, max_r, max_f1)
        precisions.append(max_p)
        recalls.append(max_r)
        f1s.append(max_f1)


def exp_all_5000_dblp_edited():
    g = read_graph("graph/com-dblp.ungraph.txt")
    communities = read_community("community/com-dblp.top5000.cmty.txt")
    f = open("output/com-dblp.top5000.output.edited.txt", "w")

    mappr = MAPPR(g, clique3)
    alpha = 0.98
    eps = 0.0001
    precisions, recalls, f1s = [], [], []
    computed_clusters = dict()
    f.write("idx,precision,recall,f1\n")
    for idx, c in enumerate(communities):
        max_p, max_r, max_f1 = 0, 0, 0
        for seed in c:
            if seed in computed_clusters:
                ma = computed_clusters[seed]
            else:
                ma = tuple(mappr.compute_appr(seed, alpha, eps / mappr.get_total_vol()
                                              * mappr.get_processed_graph().number_of_nodes(), original=False))
                computed_clusters[seed] = ma
            precision, recall, f1 = compute_precision_recall_f1(ma, c, seed)
            if f1 > max_f1:
                max_p, max_r, max_f1 = precision, recall, f1
        f.write(str(idx) + "," + str(max_p) + "," +
                str(max_r) + "," + str(max_f1) + "\n")
        print(idx, max_p, max_r, max_f1)
        precisions.append(max_p)
        recalls.append(max_r)
        f1s.append(max_f1)


def exp_all_5000_orkut():
    g = read_graph("graph/com-orkut.gr")
    communities = read_community("community/com-orkut.top5000.cmty.txt")
    f = open("output/com-orkut.top5000.output.txt", "w")

    mappr = MAPPR(g, clique3)
    alpha = 0.98
    eps = 0.0001
    precisions, recalls, f1s = [], [], []
    computed_clusters = dict()
    f.write("idx,precision,recall,f1\n")
    for idx, c in enumerate(communities):
        max_p, max_r, max_f1 = 0, 0, 0
        for seed in c:
            if seed in computed_clusters:
                ma = computed_clusters[seed]
            else:
                ma = tuple(mappr.compute_appr(seed, alpha, eps / mappr.get_total_vol()
                                              * mappr.get_processed_graph().number_of_nodes(), original=True))
                computed_clusters[seed] = ma
            precision, recall, f1 = compute_precision_recall_f1(ma, c, seed)
            if f1 > max_f1:
                max_p, max_r, max_f1 = precision, recall, f1
        f.write(str(idx) + "," + str(max_p) + "," +
                str(max_r) + "," + str(max_f1) + "\n")
        print(idx, max_p, max_r, max_f1)
        precisions.append(max_p)
        recalls.append(max_r)
        f1s.append(max_f1)


def exp_all_5000_orkut_edited():
    g = read_graph("graph/com-orkut.gr")
    communities = read_community("community/com-orkut.top5000.cmty.txt")
    f = open("output/com-orkut.top5000.output.edited.txt", "w")

    mappr = MAPPR(g, clique3)
    alpha = 0.98
    eps = 0.0001
    precisions, recalls, f1s = [], [], []
    computed_clusters = dict()
    f.write("idx,precision,recall,f1\n")
    for idx, c in enumerate(communities):
        max_p, max_r, max_f1 = 0, 0, 0
        for seed in c:
            if seed in computed_clusters:
                ma = computed_clusters[seed]
            else:
                ma = tuple(mappr.compute_appr(seed, alpha, eps / mappr.get_total_vol()
                                              * mappr.get_processed_graph().number_of_nodes(), original=False))
                computed_clusters[seed] = ma
            precision, recall, f1 = compute_precision_recall_f1(ma, c, seed)
            if f1 > max_f1:
                max_p, max_r, max_f1 = precision, recall, f1
        f.write(str(idx) + "," + str(max_p) + "," +
                str(max_r) + "," + str(max_f1) + "\n")
        print(idx, max_p, max_r, max_f1)
        precisions.append(max_p)
        recalls.append(max_r)
        f1s.append(max_f1)


def exp_amazon_recall_10():
    if len(sys.argv) != 1:
        print("usage: python3", sys.argv[0])
        return
    g = read_graph("graph/com-amazon.ungraph.txt")
    communities = read_community("community/com-amazon.recall10.cmty.txt")

    mappr = MAPPR(g, clique3)
    alpha = 0.98
    eps = 0.0001
    precisions, recalls, f1s = [], [], []
    computed_clusters = dict()
    for idx, c in enumerate(communities):
        max_p, max_r, max_f1 = 0, 0, 0
        best_seed = -1
        for seed in c:
            if seed in computed_clusters:
                ma = computed_clusters[seed]
            else:
                ma = tuple(mappr.compute_appr(seed, alpha, eps / mappr.get_total_vol()
                                              * mappr.get_processed_graph().number_of_nodes()))
                computed_clusters[seed] = ma
            precision, recall, f1 = compute_precision_recall_f1(ma, c, seed)
            if f1 > max_f1:
                max_p, max_r, max_f1 = precision, recall, f1
                best_seed = seed
        precisions.append(max_p)
        recalls.append(max_r)
        f1s.append(max_f1)
        if max_r > 2 * max_p:
            cnt += 1
            print("### community ", idx, sep='')
            print()
            print("- precision: ", max_p, ", recall: ",
                  max_r, ", f1: ", max_f1, sep='')
            print("- seed: ", seed, sep='')
            print("- cluster: ", sep='', end='')
            for nd in computed_clusters[best_seed]:
                if nd not in c:
                    print(nd, end=', ')
                else:
                    print("**", nd, "**", sep='', end=', ')
            print()
            print("- ground-truth: ", c, sep='')
            print()
            if cnt >= 10:
                break
    print("precision:", sum(precisions)/len(precisions))
    print("recall:", sum(recalls)/len(recalls))
    print("F1:", sum(f1s)/len(f1s))


def exp_amazon_random_100():
    if len(sys.argv) != 1:
        print("usage: python3", sys.argv[0])
        return
    g = read_graph("graph/com-amazon.ungraph.txt")
    communities = read_community("community/com-amazon.recall10.cmty.txt")

    mappr = MAPPR(g, clique3)
    alpha = 0.98
    eps = 0.0001
    precisions, recalls, f1s = [], [], []
    computed_clusters = dict()
    for idx, c in enumerate(communities):
        max_p, max_r, max_f1 = 0, 0, 0
        best_seed = -1
        for seed in c:
            if seed in computed_clusters:
                ma = computed_clusters[seed]
            else:
                ma = tuple(mappr.compute_appr(seed, alpha, eps / mappr.get_total_vol()
                                              * mappr.get_processed_graph().number_of_nodes()))
                computed_clusters[seed] = ma
            precision, recall, f1 = compute_precision_recall_f1(ma, c, seed)
            if f1 > max_f1:
                max_p, max_r, max_f1 = precision, recall, f1
                best_seed = seed
        precisions.append(max_p)
        recalls.append(max_r)
        f1s.append(max_f1)
    print("precision:", sum(precisions)/len(precisions))
    print("recall:", sum(recalls)/len(recalls))
    print("F1:", sum(f1s)/len(f1s))


def create_processed_graph():
    graph_file = sys.argv[1]
    output_file_path = "processed_graph/" + graph_file[6:]
    f = open(output_file_path, "w")
    print(output_file_path)
    g = read_graph(sys.argv[1])
    mappr = MAPPR(g, clique3)
    gw = mappr.get_processed_graph()
    for (u, v, w) in gw.edges(data=True):
        line = str(u) + " " + str(v) + " " + str(w['weight']) + "\n"
        f.write(line)
    for nd in gw.nodes():
        if gw.degree(nd) == 0:
            f.write(str(nd) + "\n")


def appr_with_processed():
    communities = read_community(sys.argv[2])
    from collections import defaultdict
    d = defaultdict(int)
    nd2comms = defaultdict(list)
    cnt = 0
    for i, community in enumerate(communities):
        similar = False
        for c in communities[:i]:
            p, r, f1 = compute_precision_recall_f1(community, c)
            if p > 0.5 or r > 0.5:
                similar = True
        if similar:
            continue
        for nd in community:
            d[nd] += 1
            nd2comms[nd].append(community)
        cnt += 1
    all = list(d.items())
    print("distinct communities:", cnt)
    l = []
    for idx, num in all:
        if num <= 1:
            continue
        l.append((idx, num))
    l.sort(key=lambda x: x[1])
    # print(l)
    targets = sample(l, 5)
    # targets = l
    # targets = l[-5:]
    print(targets)
    # g must be a processed graph
    g = read_graph(sys.argv[1])
    appr = APPR(g)
    for seed, num_comms in targets:
        print("seed: ", seed, ", degree: ", g.degree(
            seed), ", #communities:", num_comms, sep='')
        cluster = appr.compute_appr(seed)
        cluster.sort()
        print("cluster: ", cluster, sep='')
        for i, c in enumerate(nd2comms[seed]):
            s = list(c)
            s.sort()
            print("community ", i, ": ", s, sep='')
        draw_graph(g.subgraph(cluster), "image/" + str(seed) +
                   "-" + str(num_comms) + ".pdf", seed)


def from_processed():
    g = read_graph(sys.argv[1])
    print("graph read: ", g, sep='')
    appr = APPR(g)
    communities = read_community(sys.argv[2])
    print("community read")
    alpha = 0.98
    eps = 0.0001
    computed_clusters = dict()
    target_ids = [4676, 4692, 4693, 4694, 4700]
    for target in target_ids:
        c = communities[target]
        max_p, max_r, max_f1 = 0, 0, 0
        best_seed = -1
        for seed in c:
            if seed in computed_clusters:
                ma = computed_clusters[seed]
            else:
                ma = tuple(appr.compute_appr(seed, alpha, eps / appr.get_total_vol()
                                             * appr.get_graph().number_of_nodes(), original=False))
                computed_clusters[seed] = ma
            precision, recall, f1 = compute_precision_recall_f1(ma, c, seed)
            if f1 > max_f1:
                max_p, max_r, max_f1 = precision, recall, f1
                best_seed = seed
        print("seed: ", best_seed,  ", precision:", max_p,
              ", recall:", max_r, ", F1:", max_f1, sep='')
        print("cluster: ", end='')
        for nd in computed_clusters[best_seed]:
            if nd in c:
                print("**", nd, "**", sep='', end=', ')
            else:
                print(nd, sep='', end=', ')
        print("community:", c)
        sub = g.subgraph(computed_clusters[best_seed])
        draw_graph(sub, "image/" + str(best_seed) + ".pdf", best_seed)


def find_interesting_edge():
    g = read_graph(sys.argv[1])
    print("graph read")
    degrees = [(nd, g.degree(nd)) for nd in g.nodes()]
    degrees.sort(key=lambda x: x[1], reverse=True)
    degrees = degrees[:len(degrees) // 10]
    shuffle(degrees)
    print("nodes shuffled")

    # communities
    communities = read_community(sys.argv[2])
    print("community read")
    nd2comms = defaultdict(list)
    for community in communities:
        for nd in community:
            nd2comms[nd].append(community)

    # output
    name = "output/" + sys.argv[1][6:] + "-triangles.csv"
    print(name)
    f = open(name, "w")
    f.write("4-clique,same ground-truth,count\n")
    dict_4clique_samegt = defaultdict(int)
    for nd, deg in degrees[:1000]:
        # print("node: ", nd, ", degree: ", deg, end=', ')
        assert (nd in g)
        nbrs = list(g.neighbors(nd))
        triangles = {}
        from collections import Counter
        counter = Counter()
        for i in range(deg):
            u = nbrs[i]
            if u == nd:
                continue
            for j in range(i + 1, deg):
                v = nbrs[j]
                if v == nd:
                    continue
                if not g.has_edge(u, v):
                    continue
                triangles[(u, v)] = 0
                for w in nbrs:
                    if nd == w or u == w or v == w:
                        continue
                    if g.has_edge(u, w) and g.has_edge(v, w):
                        triangles[(u, v)] += 1
        for triangle in triangles.values():
            counter[triangle] += 1
        # print(triangles)
        comms = nd2comms[nd]
        for (u, v), clique_cnt in triangles.items():
            cnt = 0
            for comm in comms:
                if u in comm and v in comm:
                    cnt += 1
            line = f'{nd},{u},{v},{clique_cnt},{cnt},{len(comms)}\n'
            # f.write(line)
            dict_4clique_samegt[(clique_cnt, cnt)] += 1
    for (clique, samegt), count in dict_4clique_samegt.items():
        line = f'{clique},{samegt},{count}\n'
        f.write(line)
    print(dict_4clique_samegt)


def choose_communities_within_neighbors(
    g: nx.Graph,
    nd: int,
    cs: list,
    remove_1node=True,
    remove_included=True,
) -> set:
    # subgraph of g that nd can see in its own graph
    subgraph = g.subgraph(g.neighbors(nd))
    # all nodes in subgraph
    node_2hop = set(subgraph.nodes())
    including_comms: set(list) = set()
    for c in cs:
        for nd in node_2hop:
            if nd in c:
                clist = []
                for nd_c in c:
                    if nd_c in node_2hop:
                        clist.append(nd_c)
                clist.sort()
                including_comms.add(tuple(clist))
    if remove_1node:
        for c in list(including_comms):
            if len(c) <= 1:
                including_comms.remove(c)
    if remove_included:
        for c1 in list(including_comms):
            for c2 in list(including_comms):
                if c1 == c2:
                    continue
                c1set = set(c1)
                c2set = set(c2)
                intersection = c1set.intersection(c2set)
                try:
                    if len(intersection) >= len(c1set) // 2:
                        including_comms.remove(c1)
                    if len(intersection) >= len(c2set) // 2:
                        including_comms.remove(c2)
                except KeyError:
                    pass

    return including_comms


def two_hop_ground_truth():
    # cs = read_community(sys.argv[1])
    # counter = defaultdict(int)
    # for c in cs:
    #     for nd in c:
    #         counter[nd] += 1
    # nds = list(counter.items())
    # nds.sort(reverse=True, key=lambda x: x[1])
    # print(nds[:len(nds) // 10])
    g = read_graph(sys.argv[1])
    cs = read_community(sys.argv[2])

    for vp in g.nodes():  # view point node
        sub = create_n_hop_graph(g, vp, 2)
        cs_within_2hop = choose_communities_within_neighbors(g, vp, cs)
        sub.remove_node(vp)
        if len(cs_within_2hop) < 2:
            continue
        print("seed: ", vp, ", #gtcomms: ", len(cs_within_2hop), ", #subcomponents: ",
              nx.number_connected_components(sub), sep='')


def create_n_hop_graph_with_false_edges(g: nx.Graph, seed: int, n: int, false_edge_rate: float) -> nx.Graph:
    q = deque([(seed, 0)])
    seen = set([seed])
    dists = defaultdict(set)
    dists[0].add(seed)
    n_hop_edges = set()
    while q:
        nd, dist = q.popleft()
        if dist >= n:
            break
        for nbr in g.neighbors(nd):
            if nbr not in seen:
                seen.add(nbr)
                dists[dist + 1].add(nbr)
                q.append((nbr, dist + 1))
                if dist == n - 1:
                    n_hop_edges.add((nd, nbr))
    subgraph: nx.Graph = g.subgraph(seen).copy()
    num_remove = math.floor(len(n_hop_edges) * false_edge_rate)
    if n not in dists or num_remove < 2:
        return subgraph
    n_hop_nodes = dists[n]
    remove_edges = random.sample(n_hop_edges, num_remove)
    for e in remove_edges:
        subgraph.remove_edge(e[0], e[1])
    for _ in range(num_remove):
        u, v = random.sample(n_hop_nodes, 2)
        subgraph.add_edge(u, v)
    return subgraph


def get_n_hop_nodes(g: nx.Graph, seed: int, n: int):
    q = deque([(seed, 0)])
    seen = set([seed])
    dists = defaultdict(set)
    dists[0].add(seed)
    while q:
        nd, dist = q.popleft()
        if dist >= n:
            break
        for nbr in g.neighbors(nd):
            if nbr not in seen:
                seen.add(nbr)
                dists[dist + 1].add(nbr)
                q.append((nbr, dist + 1))
    return dists[n]


def exp_2_hop_comparison():
    seed = 478354

    mg = read_graph(sys.argv[2])
    mappr = APPR(mg)
    mg = create_n_hop_graph(mg, seed, 2)

    g = read_graph(sys.argv[1])
    hop3_g = create_n_hop_graph(g, seed, 3, include_boundary_edges=True)
    g = create_n_hop_graph(g, seed, 2)
    submappr = MAPPR(g, clique3)
    submapprg = submappr.get_processed_graph()

    subgcluster = submappr.compute_appr(seed, total_vol=mappr.get_total_vol())
    subcluster_set = set(subgcluster)
    allcluster = mappr.compute_appr(seed)
    allcluster_set = set(allcluster)
    hop2s = set(nd for nd in mg.nodes())

    print("- subgcluster: [", end='')
    for nd in subgcluster:
        if nd in allcluster_set:
            print(f'***{nd}***', end=', ')
        else:
            print(f'{nd} ', end=', ')
    print("]")

    print("- allcluster: [",  end="")
    for nd in allcluster:
        if nd in subcluster_set:
            print(f'***{nd}***', end=', ')
        elif nd in hop2s:
            print(f'**{nd}**', end=', ')
        else:
            print(f'{nd} ', end=', ')
    print("]")

    draw_graph(g, "image/" + str(seed) + ".pdf", special_node=seed,
               special_nodes=[nd for nd in g.neighbors(seed)])
    draw_graph(
        submapprg.subgraph(nx.node_connected_component(submapprg, seed)),
        "image/" + str(seed) + "-sub-mg.pdf",
        special_node=seed,
        special_nodes=[nd for nd in submapprg.neighbors(seed)]
    )
    draw_graph(mg, "image/" + str(seed) + "-all-mg.pdf", special_node=seed,
               special_nodes=[nd for nd in mg.neighbors(seed)])
    draw_graph(hop3_g, "image/" + str(seed) + "-all-mg-3hop.pdf", special_node=seed,
               special_nodes=[nd for nd in hop3_g.neighbors(seed)], special_nodes2=get_n_hop_nodes(g, seed, 2))
    pass


def exp_2hop_vs_louvain():
    g = read_graph(sys.argv[1])
    # mg = read_graph(sys.argv[2])
    louvain_communities = nx.community.louvain_communities(g)
    louvain_clusters = {}
    for c in louvain_communities:
        for nd in c:
            louvain_clusters[nd] = c
    f = open("output/" + sys.argv[1][6:] +
             "-intersection.louvain.output.csv", "w")
    names = ["subgcluster", "all2hops", "globalcluster"]

    line = "seed,|B∩A|/|B|,|B∩C|/|B|,|A∩B|/|A|,|A∩C|/|A|,|C∩B|/|C|,|C∩A|/|C|,|A|,|B|,|C|,#nbrs,#connceted_nodes in motif of 2hop"
    line += "\n"
    f.write(line)
    print(line, end='')

    # print()
    seeds = [nd for nd in g.nodes()]
    if len(seeds) >= 10000:
        seeds = random.sample(seeds, 10000)

    cnt = 0

    for seed in seeds:
        cnt += 1
        if len(seeds) < 100:
            print(cnt / len(seeds) * 100, "%")
        elif len(seeds) == 10000 and cnt % 100 == 0:
            print(cnt // 100, "%")
        elif len(seeds) != 10000 and cnt % (len(seeds) // 100) == 0:
            print(cnt // (len(seeds) // 100), "%")
        line = str(seed) + ","
        subg = create_n_hop_graph(g, seed, 2)
        subgmappr = MAPPR(subg, clique3)
        v = subgmappr.compute_appr(seed, total_vol=g.number_of_edges())
        if v is None:
            continue
        subgcluster = set(v)
        all2hops = set(nd for nd in subg.nodes())

        globalcluster = louvain_clusters[seed]
        l = [subgcluster, all2hops, globalcluster]
        for i, s1 in enumerate(l):
            for j, s2 in enumerate(l):
                if i == j:
                    continue
                line += f'{len(s1.intersection(s2))/len(s1)},'
        line += f'{len(subg)},{len(subgcluster)},{len(globalcluster)},{g.degree(seed)},{len(nx.node_connected_component(subgmappr.get_processed_graph(), seed))}'
        line += '\n'
        f.write(line)
        # print(line, end='')

    pass


def exp_louvain_with_seeds():
    file_path = sys.argv[1]
    g = read_graph(sys.argv[1])
    clusters = read_clusters(sys.argv[2])
    louvain_communities = nx.community.louvain_communities(g)
    louvain_clusters = {}
    for c in louvain_communities:
        for nd in c:
            louvain_clusters[nd] = c

    graph_name = file_path[6:file_path.rfind(".")]
    f = open("cluster/" + graph_name + "/louvain.output", "w")
    for seed in clusters.keys():
        write_to_file(f, seed, louvain_clusters[seed])
    f.close()


def exp_appr_with_seeds():
    file_path = sys.argv[1]
    g = read_graph(sys.argv[1])
    clusters = read_clusters(sys.argv[2])
    appr = APPR(g)

    graph_name = file_path[6:file_path.rfind(".")]
    f = open("cluster/" + graph_name + "/appr.output", "w")
    cnt = 0
    for seed in clusters.keys():
        print(cnt)
        cnt += 1
        write_to_file(f, seed, appr.compute_appr(seed))
    f.close()


def write_to_file(f, s, c):
    line = f'{s}'
    for nd in c:
        line += f' {nd}'
    line += "\n"
    f.write(line)


def exp_2hop_vs_all(motif=True):
    g = read_graph(sys.argv[1])
    # mg = read_graph(sys.argv[2])
    names = ["subgcluster", "all2hops", "globalcluster"]

    if motif:
        mappr = MAPPR(g, clique3)
        f = open("output/" + sys.argv[1][6:] +
                 "-intersection.output.csv", "w")
    else:
        mappr = APPR(g)
        f = open("output/" + sys.argv[1][6:] +
                 "-intersection.nonmotif.output.csv", "w")
    line = "seed,|A∩B|/|A|,|A∩B|/|B|,|A∩C|/|A|,|A∩C|/|C|,|B∩C|/|B|,|B∩C|/|C|,|A|,|B|,|C|," + \
        "#nodes," + \
        "#motifconncetednodes," + \
        "#edges," + \
        "summotifweights," + \
        "neighborweighteddeg," + \
        "CC," + \
        "aveCC," + \
        "aveweightedCC," + \
        "deg," + \
        "weighteddeg," + \
        "avedeg," + \
        "aveweighteddeg," + \
        "density," + \
        "#triangles"
    line += "\n"
    f.write(line)
    print(line, end='')

    # print()
    seeds = [nd for nd in g.nodes()]
    SAMPLES = 10000
    if len(seeds) >= SAMPLES:
        seeds = random.sample(seeds, SAMPLES)

    cnt = 0
    cc = nx.clustering(g, g.nodes())
    processed = mappr.get_processed_graph()
    weightedcc = nx.clustering(
        mappr.get_processed_graph(), processed, weight="weight")
    for seed in seeds:
        cnt += 1
        per1 = len(seeds) // 100
        if len(seeds) < 100:
            print(cnt / len(seeds) * 100, "%")
        elif len(seeds) == SAMPLES and cnt % per1 == 0:
            print(cnt // 100, "%")
        elif len(seeds) != SAMPLES and cnt % per1 == 0:
            print(cnt // (len(seeds) // 100), "%")
        line = str(seed) + ","
        subg = create_n_hop_graph(g, seed, 2)
        subgmappr = MAPPR(subg, clique3)
        submotifg = subgmappr.get_processed_graph()
        v = subgmappr.compute_appr(seed, total_vol=mappr.get_total_vol())
        # v = subgmappr.compute_appr(seed)
        if v is None:
            continue
        subgcluster = set(v)
        all2hops = set(nd for nd in subg.nodes())

        globalcluster = set(mappr.compute_appr(seed))
        pairs = [(all2hops, subgcluster), (all2hops, globalcluster),
                 (subgcluster, globalcluster), ]
        for i, (s1, s2) in enumerate(pairs):
            line += f'{len(s1.intersection(s2))/len(s1)},{len(s2.intersection(s1))/len(s2)},'
        line += f'{len(subg)},' + f'{len(subgcluster)},' + f'{len(globalcluster)},' + \
            f'{len(subg)},' + \
            f'{len(nx.node_connected_component(subgmappr.get_processed_graph(), seed))},' + \
            f'{subg.number_of_edges()},' + \
            f'{submotifg.size(weight="weight")},' + \
            f'{sum(submotifg.degree(nbr, weight="weight") for nbr in submotifg.neighbors(seed)) + submotifg.degree(seed, weight="weight")},' +  \
            f'{cc[seed]},' + \
            f'{sum(cc[nd] for nd in subg.nodes()) / subg.number_of_nodes()},' + \
            f'{sum(weightedcc[nd] for nd in subg.nodes()) / subg.number_of_nodes()},' + \
            f'{g.degree(seed)},' + \
            f'{submotifg.degree(seed, weight="weight")},' + \
            f'{sum(subg.degree(nd) for nd in subg) / subg.number_of_nodes()},' + \
            f'{sum(processed.degree(nd, weight="weight") for nd in subg) / subg.number_of_nodes()},' + \
            f'{nx.density(subg)}'
        line += '\n'
        f.write(line)

    pass


def exp_2hop_vs_all_false_edges():
    g = read_graph(sys.argv[1])
    # mg = read_graph(sys.argv[2])
    names = ["subgcluster", "all2hops", "globalcluster"]

    mappr = MAPPR(g, clique3)
    f = open("output/" + sys.argv[1][6:] +
             "-false_edges.output.csv", "w")
    false_rates = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25,
                   0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    line = ","
    line += "".join([(str(false_rate) + ",") *
                     6 for false_rate in false_rates])
    line += "\n"
    line += "seed,"
    for _ in false_rates:
        line += "|A∩B|/|A|,|A∩B|/|B|,|A∩C|/|A|,|A∩C|/|C|,|B∩C|/|B|,|B∩C|/|C|,"
    line += "\n"
    f.write(line)
    print(line, end='')

    # print()
    seeds = [nd for nd in g.nodes()]
    SAMPLES = 1000
    if len(seeds) >= SAMPLES:
        seeds = random.sample(seeds, SAMPLES)

    cnt = 0
    for seed in seeds:
        cnt += 1
        per1 = len(seeds) // 100
        if len(seeds) < 100:
            print(cnt / len(seeds) * 100, "%")
        elif len(seeds) == SAMPLES and cnt % per1 == 0:
            print(cnt // per1, "%")
        elif len(seeds) != SAMPLES and cnt % per1 == 0:
            print(cnt // per1, "%")
        line = str(seed) + ","
        globalcluster = set(mappr.compute_appr(seed))

        for false_rate in false_rates:
            subg = create_n_hop_graph_with_false_edges(
                g, seed, 2, false_rate)
            subgmappr = MAPPR(subg, clique3)
            v = subgmappr.compute_appr(seed, total_vol=mappr.get_total_vol())
            if v is None:
                continue
            subgcluster = set(v)
            all2hops = set(nd for nd in subg.nodes())

            pairs = [(all2hops, subgcluster), (all2hops, globalcluster),
                     (subgcluster, globalcluster), ]
            for i, (s1, s2) in enumerate(pairs):
                line += f'{len(s1.intersection(s2))/len(s1)},{len(s2.intersection(s1))/len(s2)},'
        line += '\n'
        f.write(line)

    pass


def extract_cluster():
    file_path = sys.argv[1]
    graph_name = file_path[6:file_path.rfind(".")]
    g = read_graph(file_path)
    mp = MAPPR(g, clique3)
    seeds = [nd for nd in g.nodes()]
    SAMPLES = 10000
    if len(seeds) >= SAMPLES:
        seeds = random.sample(seeds, SAMPLES)
    try:
        os.makedirs("cluster/" + graph_name)
    except FileExistsError:
        pass
    # f2 = open("cluster/" + graph_name + "/cluster-2hop.output", "w")
    # f2i = open("cluster/" + graph_name + "/cluster-2hop-induced.output", "w")
    # f3i = open("cluster/" + graph_name + "/cluster-3hop-induced.output", "w")
    # f4i = open("cluster/" + graph_name + "/cluster-4hop-induced.output", "w")
    # f2t = open("cluster/" + graph_name + "/cluster-2hop-totalvol.output", "w")
    # f2it = open("cluster/" + graph_name +
    #             "/cluster-2hop-induced-totalvol.output", "w")
    # f3it = open("cluster/" + graph_name +
    #             "/cluster-3hop-induced-totalvol.output", "w")
    # f4it = open("cluster/" + graph_name +
    #             "/cluster-4hop-induced-totalvol.output", "w")
    f = open("cluster/" + graph_name +
             "/cluster-all.output", "w")
    # f2all = open("cluster/" + graph_name +
    #              "/2hop-all.output", "w")

    for seed in seeds:
        # 2-hop graph
        # hop2 = create_n_hop_graph(g, seed, 2, False)
        # hop2i = create_n_hop_graph(g, seed, 2, True)
        # hop3i = create_n_hop_graph(g, seed, 3, True)
        # hop4i = create_n_hop_graph(g, seed, 4, True)

        # mp2 = MAPPR(hop2, clique3)
        # mp2i = MAPPR(hop2i, clique3)
        # mp3i = MAPPR(hop3i, clique3)
        # mp4i = MAPPR(hop4i, clique3)

        # c2 = mp2.compute_appr(seed)
        # c2t = mp2.compute_appr(seed, total_vol=INF)
        # c2i = mp2i.compute_appr(seed)
        # c2it = mp2i.compute_appr(seed, total_vol=INF)
        # c3i = mp3i.compute_appr(seed)
        # c3it = mp3i.compute_appr(seed, total_vol=INF)
        # c4i = mp4i.compute_appr(seed)
        # c4it = mp4i.compute_appr(seed, total_vol=INF)
        c = mp.compute_appr(seed)
        # c2all = [nd for nd in hop2.nodes()]

        # write_to_file(f2, seed, c2)
        # write_to_file(f2t, seed, c2t)
        # write_to_file(f2i, seed, c2i)
        # write_to_file(f2it, seed, c2it)
        # write_to_file(f3i, seed, c3i)
        # write_to_file(f3it, seed, c3it)
        # write_to_file(f4i, seed, c4i)
        # write_to_file(f4it, seed, c4it)
        write_to_file(f, seed, c)
        # write_to_file(f2all, seed, c2all)

    # f2.close()
    # f2i.close()
    # f3i.close()
    # f4i.close()
    # f2t.close()
    # f2it.close()
    # f3it.close()
    # f4it.close()
    f.close()


def exp_compare_2hop_all():
    file_path = sys.argv[1]
    graph_name = file_path[6:file_path.rfind(".")]
    g = read_graph(sys.argv[1])
    mg = read_graph(sys.argv[2])
    graph_name = file_path[6:file_path.rfind(".")]
    seeds = [nd for nd in g.nodes()]
    SAMPLES = 10000
    if len(seeds) >= SAMPLES:
        seeds = random.sample(seeds, SAMPLES)

    f = open("output/remove_edge-" + graph_name + ".csv", "w")
    f.write("seed,remove_1hop,remove_w_1hop,remove_2hop,remove_w_2hop,diff_2hop,diff_w_2hop,same_2hop,same_w_2hop,remove_3hop,remove_w_3hop,diff_3hop,diff_w_3hop,edge_1hop,edge_w_1hop,edge_2hop,edge_w_2hop,edge_3hop,edge_w_3hop\n")
    print("output/remove_edge-" + graph_name + ".csv")
    cnt = 0
    for seed in seeds:
        hop2 = create_n_hop_graph(g, seed, 2)
        nbrs = set([nd for nd in hop2.neighbors(seed)])
        hop2mp = MAPPR(hop2, clique3)
        hop2g = hop2mp.get_processed_graph()
        hop2sub = create_n_hop_graph(mg, seed, 2, True)
        remove_1hop, remove_w_1hop, remove_2hop, remove_w_2hop, diff_2hop, diff_w_2hop, \
            same_2hop, same_w_2hop, remove_3hop, remove_w_3hop, diff_3hop, diff_w_3hop, \
            edge_1hop, edge_2hop, edge_3hop, edge_w_1hop, edge_w_2hop, edge_w_3hop =\
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for u, v, a in hop2sub.edges(data=True):
            w = a['weight']
            if u == seed or v == seed:
                edge_1hop += 1
                edge_w_1hop += w
            elif u in nbrs or v in nbrs:
                edge_2hop += 1
                edge_w_2hop += w
            else:
                edge_3hop += 1
                edge_w_3hop += w

            if hop2g.has_edge(u, v):
                if w == hop2g[u][v]['weight'] and (u != seed and v != seed) and (u in nbrs or v in nbrs):
                    same_2hop += 1
                    same_w_2hop += w
                    continue
                if u == seed or v == seed:
                    remove_1hop += 1
                    remove_w_1hop += w
                elif u in nbrs or v in nbrs:
                    diff_2hop += 1
                    diff_w_2hop += w - hop2g[u][v]['weight']
                else:
                    diff_3hop += 1
                    diff_w_3hop += w - hop2g[u][v]['weight']
            else:
                if u == seed or v == seed:
                    remove_1hop += 1
                    remove_w_1hop += w
                elif u in nbrs or v in nbrs:
                    remove_2hop += 1
                    remove_w_2hop += w
                else:
                    remove_3hop += 1
                    remove_w_3hop += w
        f.write(
            f'{seed},{remove_1hop},{remove_w_1hop},{remove_2hop},{remove_w_2hop},{diff_2hop},{diff_w_2hop},{same_2hop},{same_w_2hop},{remove_3hop},{remove_w_3hop},{diff_3hop},{diff_w_3hop},{edge_1hop},{edge_w_1hop},{edge_2hop},{edge_w_2hop},{edge_3hop},{edge_w_3hop}\n')


def read_non_seeded_clusters(filepath) -> dict:
    f = open(filepath)
    clusters = {}
    for line in f.readlines():
        nodes = set(map(int, line.split()[1:]))
        for nd in nodes:
            clusters[nd] = nodes
    f.close()
    return clusters


def exp_louvain_with_seeds_and_size():
    file_path = sys.argv[1]
    g = read_graph(file_path)
    clusters = read_clusters(sys.argv[2])
    from louvain import louvain_partitions
    level_partitions = louvain_partitions(g)
    louvain_clusters = {}
    levels = len(level_partitions)
    for i in range(levels):
        louvain_clusters[i] = dict()
    for level, partition in enumerate(level_partitions):
        for cluster in partition:
            for nd in cluster:
                louvain_clusters[level][nd] = cluster
    # print(louvain_clusters)

    best_clusters = {}
    for seed, cluster in clusters.items():
        min_diff = INF
        best_c = -1
        for i in range(level):
            lc = louvain_clusters[i][seed]
            if abs(len(lc) - len(cluster)) < min_diff:
                best_c = lc
                min_diff = abs(len(lc) - len(cluster))
        best_clusters[seed] = best_c

    # print("best clusters:", best_clusters)

    graph_name = file_path[6:file_path.rfind(".")]
    f = open("cluster/" + graph_name + "/louvain-near-size.output", "w")
    for seed in clusters.keys():
        write_to_file(f, seed, best_clusters[seed])
    f.close()


def exp_compare():
    file_path = sys.argv[1]
    g = read_graph(sys.argv[1])
    clusters = read_clusters(sys.argv[2])
    from louvain import louvain_partitions
    level_partitions = louvain_partitions(g)
    louvain_clusters = {}
    levels = len(level_partitions)
    for i in range(levels):
        louvain_clusters[i] = dict()
    for level, partition in enumerate(level_partitions):
        for cluster in partition:
            for nd in cluster:
                louvain_clusters[level][nd] = cluster
    # print(louvain_clusters)

    best_clusters = {}
    for seed, cluster in clusters.items():
        min_diff = INF
        best_c = -1
        for i in range(level):
            lc = louvain_clusters[i][seed]
            if abs(len(lc) - len(cluster)) < min_diff:
                best_c = lc
                min_diff = abs(len(lc) - len(cluster))
        best_clusters[seed] = best_c

    # print("best clusters:", best_clusters)

    graph_name = file_path[6:file_path.rfind(".")]
    f = open("cluster/" + graph_name + "/louvain-near-size.output", "w")
    for seed in clusters.keys():
        write_to_file(f, seed, best_clusters[seed])
    f.close()


def print_md_table(labels: list, lines: list):
    assert len(labels) > 0
    for line in lines:
        assert len(labels) == len(line)

    print("|", end='')
    for label in labels:
        print(label, end="|")
    print()
    print("|", end='')
    for _ in labels:
        print(":--:|", end='')
    print()
    for line in lines:
        print("|", end='')
        for value in line:
            print(value, end="|")
        print()


def compute_mappr():
    graph_name = get_gname(sys.argv[1])
    g = read_graph(sys.argv[1])
    f = open("cluster/" + graph_name + "/all-mappr.output", "w")
    mappr = MAPPR(g, clique3)
    for seed in g:
        cluster = mappr.compute_appr(seed)
        cluster.sort()
        line = f'{seed}'
        for nd in cluster:
            line += f' {nd}'
        line += '\n'
        f.write(line)
        print(seed, "done")
    pass


def compute_hop_appr(
    graph_name: str,
    g: nx.Graph,
    hop,
    alpha: float = 0.98,
    eps_exp: int = -4,
    total_vol: int = None
):
    f = open("cluster/" + graph_name + "/all-" +
             str(hop) + "-hop-appr-" + str(alpha) + "-10e" + str(eps_exp) + "-" + str(total_vol) + ".output", "w")
    num_seeds = g.number_of_nodes()
    percent = num_seeds // 100
    cnt = 0
    start = time.time()
    for seed in g:
        subg = create_n_hop_graph(g, seed, hop)
        appr = APPR(subg)
        cluster = appr.compute_appr(
            seed, alpha, 10 ** eps_exp, total_vol=total_vol)
        cluster.sort()
        line = f'{seed}'
        for nd in cluster:
            line += f' {nd}'
        line += '\n'
        f.write(line)
        cnt += 1
        if cnt % percent == 0:
            now = time.time()
            done_percent = cnt // percent
            current = now - start
            reamaining_time = current / done_percent * (100 - done_percent)
            print(cnt // percent, "% done; ",
                  round(reamaining_time, 1), "seconds remaining")
    f.close()


def compute_hop_mappr(
    graph_name: str,
    g: nx.Graph,
    hop,
    alpha: float = 0.98,
    eps_exp: int = -4,
    total_vol: int = None
):
    f = open("cluster/" + graph_name + "/all-" +
             str(hop) + "-hop-mappr-" + str(alpha) + "-10e" +
             str(eps_exp) + "-" + str(total_vol) + ".output", "w")
    num_seeds = g.number_of_nodes()
    percent = num_seeds // 100
    cnt = 0
    start = time.time()
    for seed in g:
        subg = create_n_hop_graph(g, seed, hop)
        mappr = MAPPR(subg, clique3)
        cluster = mappr.compute_appr(seed, alpha, 10**eps_exp, total_vol)
        cluster.sort()
        line = f'{seed}'
        for nd in cluster:
            line += f' {nd}'
        line += '\n'
        f.write(line)
        cnt += 1
        if cnt % percent == 0:
            now = time.time()
            done_percent = cnt // percent
            current = now - start
            reamaining_time = current / done_percent * (100 - done_percent)
            print(cnt // percent, "% done; ",
                  round(reamaining_time, 1), "seconds remaining")


def compute_hop_louvain():
    graph_name = get_gname(sys.argv[1])
    g = read_graph(sys.argv[1])
    for hop in [2, 3, 4, 5]:
        f = open("cluster/" + graph_name + "/all-" +
                 str(hop) + "-hop-louvain.output", "w")
        for seed in g:
            subg = create_n_hop_graph(g, seed, hop)
            louvain_communities = nx.community.louvain_communities(subg)
            louvain_clusters = {}
            for c in louvain_communities:
                if seed in c:
                    cluster = list(c)
                    break
            cluster.sort()
            line = f'{seed}'
            for nd in cluster:
                line += f' {nd}'
            line += '\n'
            f.write(line)
            # print("node", seed, "done")
    pass


def compute_hop_louvain_resolution(
    graph_name,
    g,
    resolution,
    hops=[2, 3, 4, 5],
):
    for hop in hops:
        f = open("cluster/" + graph_name + "/all-" +
                 str(hop) + "-hop-louvain" + str(resolution) + ".output", "w")

        num_seeds = g.number_of_nodes()
        percent = num_seeds // 100
        cnt = 0
        start = time.time()
        for seed in g:
            subg = create_n_hop_graph(g, seed, hop)
            louvain_communities = nx.community.louvain_communities(
                subg, resolution=resolution)
            louvain_clusters = {}
            for c in louvain_communities:
                if seed in c:
                    cluster = list(c)
                    break
            cluster.sort()
            line = f'{seed}'
            for nd in cluster:
                line += f' {nd}'
            line += '\n'
            f.write(line)
            # print("node", seed, "done")
            cnt += 1
            if cnt % percent == 0:
                now = time.time()
                done_percent = cnt // percent
                current = now - start
                reamaining_time = current / done_percent * (100 - done_percent)
                print(cnt // percent, "% done; ",
                      round(reamaining_time, 1), "seconds remaining")
    pass


def compute_louvain(graph_name):
    g = read_graph("graph/" + graph_name + ".gr")
    for resolution in [0.125, 0.25, 0.5, 2, 4, 8, 16, 32, 64]:
        print("resolution:", resolution, end='')
        start = time.time()
        f = open("cluster/" + graph_name + "/louvain-" +
                 str(resolution) + ".output", "w")
        louvain_communities = nx.community.louvain_communities(
            g, resolution=resolution)
        louvain_clusters = {}
        for c in louvain_communities:
            for nd in c:
                louvain_clusters[nd] = c
        for seed in g:
            cluster = list(louvain_clusters[seed])
            cluster.sort()
            line = f'{seed}'
            for nd in cluster:
                line += f' {nd}'
            line += '\n'
            f.write(line)
            # print(seed, "done")
        end = time.time()
        print(", time:", end - start)
    pass


def compute_non_seeded_louvain(graph_name):
    g = read_graph("graph/" + graph_name + ".gr")
    for resolution in [0.125, 0.25, 0.5, 2, 4, 8, 16, 32, 64]:
        print("resolution:", resolution, end='')
        f = open("cluster/" + graph_name + "/louvain-" +
                 str(resolution) + ".output", "w")
        start = time.time()
        louvain_communities = nx.community.louvain_communities(
            g, resolution=resolution)
        end = time.time()
        print(", time:", end - start)
        for c in louvain_communities:
            comms = list(c)
            line = f'{comms[0]}'
            if len(comms) > 1:
                for nd in comms[1:]:
                    line += f' {nd}'
            line += '\n'
            f.write(line)
    pass


def write_hop_graph(graph_name):
    hop = 2
    g = read_graph("graph/" + graph_name + ".gr")
    f = open("cluster/" + graph_name + "/sub-" +
             str(hop) + "-hop.output", "w")
    for seed in g:
        subg = create_n_hop_graph(g, seed, hop)
        line = f'{seed}'
        for nd in subg:
            line += f' {nd}'
        line += '\n'
        f.write(line)
        print(seed, "done")


def exp_compare_all_vs_local():
    # g = read_graph(sys.argv[1])
    # graph_name = get_gname(sys.argv[1])
    # objective = "conductance"
    # local_method = "appr"
    # if len(sys.argv) >= 3:
    #     objective = sys.argv[2]
    # if len(sys.argv) >= 4:
    #     local_method = sys.argv[3]
    # print("###", graph_name)
    # print()
    # local_clusters = read_clusters(
    #     "cluster/" + graph_name + "/all-" + local_method + ".output")
    # paths = [
    #     "cluster/" + graph_name + "/" + objective + "-0.01.output",
    #     "cluster/" + graph_name + "/" + objective + "-0.05.output",
    #     "cluster/" + graph_name + "/" + objective + "-0.1.output",
    #     "cluster/" + graph_name + "/" + objective + "-0.15.output",
    #     "cluster/" + graph_name + "/" + objective + "-0.2.output",
    #     "cluster/" + graph_name + "/" + objective + "-0.25.output",
    #     "cluster/" + graph_name + "/" + objective + "-0.3.output",
    #     "cluster/" + graph_name + "/" + objective + "-0.35.output",
    #     "cluster/" + graph_name + "/" + objective + "-0.4.output",
    # ]
    # for path in paths:
    #     all_clusters = read_clusters(path)
    #     precs, recs, f1s = [], [], []
    #     threshold = path[path.rfind("-")+1:path.rfind(".")]
    #     f = open("output/" + objective + "-" + graph_name + "-all-vs-local-" +
    #              threshold + ".csv", "w")
    #     f.write(
    #         "clsuter_id,global_conductance,local_seed,global_size,local_size,precision,recall,f1\n")
    #     print("#### conductance threshold:", threshold)
    #     print()
    #     print("|clsuter_id|global_conductance|local_seed|global_size|local_size|precision|recall|f1|")
    #     print("|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    #     for clsuter_id, cluster in all_clusters.items():
    #         mprec, mrec, mf1, best_seed, best_cluster_size = 0, 0, 0, -1, -1
    #         for nd in cluster:
    #             prec, rec, f1 = compute_precision_recall_f1(
    #                 local_clusters[nd], cluster)
    #             if f1 > mf1:
    #                 mprec = prec
    #                 mrec = rec
    #                 mf1 = f1
    #                 best_seed = nd
    #                 best_cluster_size = len(local_clusters[nd])
    #         try:
    #             cond = nx.conductance(g, cluster)
    #         except ZeroDivisionError:
    #             cond = 1
    #         f.write(
    #             f'{clsuter_id},{round(cond, 3)},{best_seed},{len(cluster)},{best_cluster_size},{mprec},{mrec},{mf1}\n')
    #         print(
    #             f'|{clsuter_id}|{round(cond, 3)}|{best_seed}|{len(cluster)}|{best_cluster_size}|{prcnt(mprec)}|{prcnt(mrec)}|{prcnt(mf1)}|')
    #     print()
    pass


def prcnt(decimal):
    return str(round(decimal * 100, 2)) + "%"


def exp_local_global_precision_recall_f1_with_hop(
    g,
    graph_name,
    objective,
    local_method,
    global_choice,
    hop=-1,
    thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
):
    # if len(sys.argv) not in [4, 5]:
    #     print("usage: python3", sys.argv[0],
    #           "graph local_method global_choice hop")
    #     return
    if global_choice in ["best", "seed"]:
        pass
    else:
        print(global_choice, "must be one of [\"best\", \"seed\"]")
        exit()
    if hop in [2, 3, 4, 5]:
        hop_label = str(hop) + "-hop-"
    else:
        hop_label = ""
    local_clusters = read_clusters(
        "cluster/" + graph_name + "/all-" + hop_label + local_method + ".output")
    list_global_clusters = {
        threshold: read_non_seeded_clusters(
            "cluster/" + graph_name + "/" + objective + "-" + str(threshold) + ".output")
        for threshold in thresholds
    }
    if hop in [2, 3, 4, 5]:
        subgs = read_clusters(
            "cluster/" + graph_name + "/sub-" + str(hop) + "-hop.output")
        hop_label = str(hop) + "-hop-"
    else:
        hop_label = ""
    f_prec = open("output/" + graph_name + "-" + hop_label +
                  local_method + "-" + global_choice + "-precision.csv", "w")
    f_rec = open("output/" + graph_name + "-" + hop_label +
                 local_method + "-" + global_choice + "-recall.csv", "w")
    f_f1 = open("output/" + graph_name + "-" + hop_label +
                local_method + "-" + global_choice + "-f1.csv", "w")
    f_rand = open("output/" + graph_name + "-" + hop_label +
                  local_method + "-" + global_choice + "-random.csv", "w")
    line = "seed"
    for threshold in thresholds:
        line += "," + str(threshold)
    line += "\n"
    f_prec.write(line)
    f_rec.write(line)
    f_f1.write(line)
    f_rand.write(line)
    num_seeds = g.number_of_nodes()
    percent = num_seeds // 100
    cnt = 0
    start = time.time()
    for seed, local_cluster in local_clusters.items():
        line_prec, line_rec, line_f1, line_rand = str(
            seed), str(seed), str(seed), str(seed)
        if hop in [2, 3, 4, 5]:
            subg_nodes = subgs[seed]
            num_nhop = len(subg_nodes)
        else:
            num_nhop = g.number_of_nodes()
        for threshold in thresholds:
            precision, recall, f1, rand = 0, 0, 0, 0

            if global_choice == "seed":
                global_cluster: set = list_global_clusters[threshold][seed]
                if hop in [2, 3, 4, 5]:
                    global_cluster: set = global_cluster.intersection(
                        subg_nodes)
                precision, recall, f1 = compute_precision_recall_f1(
                    local_cluster,
                    global_cluster,
                )
                rand = len(global_cluster) / num_nhop
            elif global_choice == "best":
                cluster_candidates = set()
                for nd in local_cluster:
                    global_cluster: set = list_global_clusters[threshold][nd]
                    if hop in [2, 3, 4, 5]:
                        global_cluster: set = global_cluster.intersection(
                            subg_nodes)
                    cluster_candidates.add(tuple(sorted(list(global_cluster))))
                for global_cluster in cluster_candidates:
                    _precision, _recall, _f1 = compute_precision_recall_f1(
                        local_cluster,
                        global_cluster,
                    )
                    if _f1 > f1:
                        precision = _precision
                        recall = _recall
                        f1 = _f1
                        rand = len(global_cluster) / num_nhop

            line_prec += "," + str(precision)
            line_rec += "," + str(recall)
            line_f1 += "," + str(f1)
            line_rand += "," + str(rand)
        line_prec += "\n"
        line_rec += "\n"
        line_f1 += "\n"
        line_rand += "\n"
        f_prec.write(line_prec)
        f_rec.write(line_rec)
        f_f1.write(line_f1)
        f_rand.write(line_rand)
        cnt += 1
        if cnt % percent == 0:
            now = time.time()
            done_percent = cnt // percent
            current = now - start
            reamaining_time = current / done_percent * (100 - done_percent)
            print(cnt // percent, "% done; ",
                  round(reamaining_time, 1), "seconds remaining")


def read_global_clusters(thresholds) -> dict:
    global_clusters = {}
    for threshold in thresholds:
        global_cluster = read_non_seeded_clusters(
            "cluster/ca-grqc-connected/k-means-" + str(threshold) + ".output")
        global_clusters[threshold] = global_cluster
    return global_clusters


def compute_average_f1(local_cluster, global_clusters):
    # f1sum = 0
    # for gc in global_clusters:
    #     precision, recall, f1 = compute_precision_recall_f1(
    #         local_cluster, gc)
    #     f1sum += f1
    # return f1sum / len(global_clusters)
    pass


def exp_same_size_local_clusters():
    # thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # # comparison = ("appr", "mappr")
    # # comparison = ("mappr", "louvain")
    # # comparison = ("louvain", "appr")
    # comparisons = [("appr", "mappr"), ("mappr", "louvain"),
    #                ("louvain", "appr")]

    # filter = 2  # 0 for min, 1 for Q1, 2 for median, 3 for Q3, 4 for max

    # global_clusters = read_global_clusters(thresholds)

    # for num_hop in [2, 3, 4, 5]:
    #     local_methods = [str(num_hop) + "-hop-appr", str(num_hop) +
    #                      "-hop-mappr", str(num_hop) + "-hop-louvain"]
    #     clusters_appr = read_clusters(
    #         "cluster/ca-grqc-connected/all-" + str(num_hop) + "-hop-appr.output")
    #     clusters_mappr = read_clusters(
    #         "cluster/ca-grqc-connected/all-" + str(num_hop) + "-hop-mappr.output")
    #     clusters_louvain = read_clusters(
    #         "cluster/ca-grqc-connected/all-" + str(num_hop) + "-hop-louvain.output")
    #     subg = read_clusters("cluster/ca-grqc-connected/sub-" +
    #                          str(num_hop) + "-hop.output")
    #     for comparison in comparisons:
    #         match_seeds = []
    #         average_f1s_dict = read_and_compute_f1_box_points(local_methods)

    #         data = []
    #         for seed in clusters_appr.keys():
    #             ca = clusters_appr[seed]
    #             cm = clusters_mappr[seed]
    #             cl = clusters_louvain[seed]
    #             sg = subg[seed]
    #             gcs = [
    #                 global_clusters[threshold][seed].intersection(sg)
    #                 for threshold in thresholds
    #             ]
    #             appr_size = len(ca)
    #             mappr_size = len(cm)
    #             louvain_size = len(cl)
    #             if comparison == ("appr", "mappr"):
    #                 c1 = ca
    #                 s1 = appr_size
    #                 c2 = cm
    #                 s2 = mappr_size
    #                 pass
    #             elif comparison == ("mappr", "louvain"):
    #                 c1 = cm
    #                 s1 = mappr_size
    #                 c2 = cl
    #                 s2 = louvain_size
    #                 pass
    #             elif comparison == ("louvain", "appr"):
    #                 c1 = cl
    #                 s1 = louvain_size
    #                 c2 = ca
    #                 s2 = appr_size
    #                 pass
    #             else:
    #                 print("error")
    #                 exit(0)

    #             smaller, larger = min(s1, s2), max(s1, s2)

    #             if (larger - smaller) / smaller <= 0.25:
    #                 labels = ["seed", "size", "match", str(num_hop) + "-hop-size",
    #                           "global_size_ave", "f1_ave_" + comparison[0], "f1_ave_" + comparison[1]]

    #                 f1_ave1 = compute_average_f1(c1, gcs)
    #                 f1_ave2 = compute_average_f1(c2, gcs)
    #                 if f1_ave1 <= average_f1s_dict[str(num_hop) + "-hop-" + comparison[0]][filter] or f1_ave2 <= average_f1s_dict[str(num_hop) + "-hop-" + comparison[1]][filter]:
    #                     continue

    #                 match_seeds.append(seed)
    #                 data.append(
    #                     [seed,
    #                      appr_size,
    #                      len(set(c1).intersection(c2)),
    #                      len(sg),
    #                      sum([len(gc) for gc in gcs]) / len(gcs),
    #                      prcnt(compute_average_f1(c1, gcs)),
    #                      prcnt(compute_average_f1(c2, gcs)),
    #                      ]
    #                 )
    #                 pass
    #         # print_md_table(labels, data)
    #         print("hop: ", num_hop, ", comparison: ", comparison,
    #               ", match: ", len(match_seeds), sep='')
    #         # print(match_seeds)
    pass


# This function needs to deal with different number of global clusters in each conductance threshold
def read_and_compute_f1_box_points(
    labels,
    graph_name="ca-grqc-connected",
    metric="f1",
    global_choice="seed",
    valid_thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
):
    # num_valid_thresholds = len(valid_thresholds)
    # data = {}
    # all_vals = []
    # for label in labels:
    #     f = open("output/" + graph_name + "-" + label + "-" +
    #              global_choice + "-" + metric + ".csv")
    #     cnt = 0
    #     aves = []
    #     for line in f.readlines()[1:]:
    #         cnt += 1
    #         elems = list(line.split(','))
    #         seed = int(elems[0])
    #         sum_f1 = 0
    #         f1s = list(map(float, elems[1:]))
    #         for f1 in f1s[:num_valid_thresholds]:
    #             sum_f1 += f1
    #         ave_f1 = sum_f1 / num_valid_thresholds
    #         aves.append(ave_f1)
    #     all_vals.append(aves)
    #     # print(label,
    #     #       ", min: ", min(aves),
    #     #       ", Q1: ", np.percentile(aves, 25),
    #     #       ", median: ", np.percentile(aves, 50),
    #     #       ", Q3: ", np.percentile(aves, 75),
    #     #       ", max: ", max(aves),
    #     #       sep='')
    #     data[label] = (min(aves), np.percentile(aves, 25), np.percentile(
    #         aves, 50), np.percentile(aves, 75), max(aves))

    # draw_boxplot(
    #     all_vals,
    #     metric + " distribution",
    #     labels=list(map(lambda x: x[6:], labels)),
    #     rotation=15,
    #     filename="tmp/" + metric + " distribution",
    #     show_graph=True
    # )
    # return data
    pass


def compute_min_q1_median_q3_max(data: list) -> tuple:
    return (min(data), np.percentile(data, 25), np.percentile(
            data, 50), np.percentile(data, 75), max(data))


def read_two_value_csv(filepath: str) -> dict:
    f = open(filepath)
    d = {}
    for line in f.readlines()[1:]:
        k, v = map(int, line.split(","))
        d[k] = v
    return d


# access data like data[label][threshold][seed]
def read_metric_data(
    seeds,
    graph_name,
    labels,
    global_choice,
    metric
):
    set_seeds = set(seeds)
    data = {}
    for label in labels:
        print("file:", "output/" + graph_name + "-" +
              label + "-" + global_choice + "-" + metric + ".csv")
        f = open("output/" + graph_name + "-" +
                 label + "-" + global_choice + "-" + metric + ".csv")
        lines = f.readlines()
        keys = lines[0]
        data_for_label = {}
        thresholds = list(map(float, keys.split(",")[1:]))
        for threshold in thresholds:
            data_for_label[threshold] = {}
        for line in lines[1:]:
            elements = line.split(",")
            seed = int(elements[0])
            if seed not in set_seeds:
                continue
            for i, metricvalue in enumerate(elements[1:]):
                threshold = thresholds[i]
                data_for_label[threshold][seed] = float(metricvalue)
        data[label] = data_for_label
    return data


def compute_average_metric_data_for_global_cluster(
        seeds,
        label_threshold_seed_metricvalue,
        thresholds,
        graph_name,
):
    set_seeds = set(seeds)
    set_thresholds = set(thresholds)
    label_threshold_global_metricvalue = {}
    for label, threshold_seed_metricvalue in label_threshold_seed_metricvalue.items():
        threshold_global_metricvalue = {}
        for threshold, seed_metricvalue in threshold_seed_metricvalue.items():
            if threshold not in set_thresholds:
                continue
            global_metricvalue = {}
            global_clusters = read_clusters(
                "cluster/" + graph_name + "/k-means-" + str(threshold) + ".output")
            for cluster_id, cluster in global_clusters.items():
                sum_metricvalue = 0
                cluster_size = 0
                for seed in cluster:
                    if seed not in set_seeds:
                        continue
                    sum_metricvalue += seed_metricvalue[seed]
                    cluster_size += 1
                if cluster_size == 0:
                    continue
                global_metricvalue[cluster_id] = sum_metricvalue / cluster_size
            threshold_global_metricvalue[threshold] = global_metricvalue
        label_threshold_global_metricvalue[label] = threshold_global_metricvalue
    return label_threshold_global_metricvalue


def get_average_f1s(
    seeds,
    graph_name,
    labels,
    thresholds,
    global_choice,
    metric,
    point_of_view,
):
    if point_of_view == "global":
        label_threshold_seed_metricvalue = read_metric_data(
            seeds, graph_name, labels, global_choice, metric)
        label_threshold_global_metricvalue = compute_average_metric_data_for_global_cluster(
            seeds,
            label_threshold_seed_metricvalue,
            thresholds,
            graph_name,
        )
    elif point_of_view == "local":
        label_threshold_global_metricvalue = read_metric_data(
            seeds, graph_name, labels, global_choice, metric)
    else:
        print("point_of_view must be one of [\"global\", \"local\"]")
        exit(0)

    data_for_chart = [[0] * len(thresholds)
                      for _ in range(len(labels))]
    for i, label in enumerate(labels):
        threshold_global_metricvalue = label_threshold_global_metricvalue[label]
        for j, threshold in enumerate(thresholds):
            global_metricvalue = threshold_global_metricvalue[threshold]
            data_for_chart[i][j] = sum(
                global_metricvalue.values()) / len(global_metricvalue)

    return data_for_chart


def create_fig_with_specific_seeds(
    seeds: list,
    graph_name,
    labels=["2-hop-appr"],
    thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                0.85, 0.9, 0.95],
    metric="f1",
    global_choice="seed",
    title_suffix="",
    filename_suffix="",
):
    data_for_chart = get_average_f1s(
        seeds,
        graph_name,
        labels,
        thresholds,
        global_choice,
        metric,
        "local"
    )

    filename = "tmp/2-hop-with-specific-seeds-" + \
        metric + "-" + filename_suffix + ".png"
    draw_chart(
        thresholds,
        data_for_chart,
        # vals,
        title=metric + " average " + title_suffix,
        labels=labels,
        x_axis_title="conductance threshold",
        y_axis_title="" + metric + "",
        left=0,
        top=1.01,
        bottom=-0.01,
        colors=COLORS,
        filename=filename,
        show_graph=True
    )


def create_fig_with_multiple_types(
    list_seeds: list,
    types: list,
    graph_name,
    labels=["2-hop-appr"],
    thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                0.85, 0.9, 0.95],
    metric="f1",
    global_choice="seed",
    filter="2-hop-size",
    title_suffix="",
    filename_suffix="",
):
    columns = []
    num_x = len(thresholds)
    for label in labels:
        df = pd.read_csv("output/" + graph_name + "-" +
                         label + "-" + global_choice + "-" + metric + ".csv")
        column = df.transpose().values.tolist()
        columns.append(column)
    totals = []
    for seeds in list_seeds:
        total = get_average_f1s(
            seeds,
            graph_name,
            labels,
            thresholds,
            global_choice,
            metric,
            "local"
        )
        totals += total

    filename = "tmp/2-hop-with-specific-seeds" + \
        metric + "-" + filename_suffix + ".png"
    list_labels = []
    for type in types:
        for label in labels:
            list_labels.append(label + " " + filter + " " + type)
    num_types = len(types)
    num_labels = len(labels)
    colors = [c for c in COLORS[:num_labels]] * num_types
    linestyles = []
    for i in range(num_types):
        for j in range(num_labels):
            linestyles.append(LINESTYLES[i])

    draw_chart(
        thresholds,
        totals,
        title=metric + " average for specific seeds with " + title_suffix,
        labels=list_labels,
        x_axis_title="conductance threshold",
        y_axis_title="" + metric + "",
        left=0,
        top=1.01,
        bottom=-0.01,
        colors=colors,
        linestyles=linestyles,
        filename=filename,
        show_graph=True
    )


def exp_hopsize_to_f1(
    graph_name
):
    num_hop = 2
    subg = read_clusters("cluster/" + graph_name + "/sub-" +
                         str(num_hop) + "-hop.output")
    local_methods = [
        "2-hop-louvain0.305",
        "2-hop-appr-0.98-10e-4-1700",
        "2-hop-mappr-0.98-10e-4-523896",
    ]
    minsize, q1size, medsize, q3size, maxsize = compute_min_q1_median_q3_max(
        [len(sub) for sub in subg.values()]
    )
    metric = "f1"
    for Q in [True, False]:
        if Q:
            seeds_lt_q1 = []
            seeds_gt_q3 = []
        else:
            seeds_lt_median = []
            seeds_gt_median = []
        for seed, subgnodes in subg.items():
            if Q:
                if len(subgnodes) < q1size:
                    seeds_lt_q1.append(seed)
                if len(subgnodes) >= q3size:
                    seeds_gt_q3.append(seed)
            else:
                if len(subgnodes) < medsize:
                    seeds_lt_median.append(seed)
                if len(subgnodes) >= medsize:
                    seeds_gt_median.append(seed)
        if Q:
            list_seeds = [seeds_lt_q1, seeds_gt_q3]
            types = ["< Q1", ">= Q3"]
            title_suffix = "2-hop-size compared w/ Q1 & Q3"
            filename_suffix = "2-hop-size-q1q3"
        else:
            list_seeds = [seeds_lt_median, seeds_gt_median]
            types = ["< median", ">= median"]
            title_suffix = "2-hop-size compared w/ median"
            filename_suffix = "2-hop-size-median"
        create_fig_with_multiple_types(
            list_seeds,
            types,
            graph_name,
            labels=local_methods,
            metric=metric,
            filter="2-hop-size",
            thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            title_suffix=title_suffix,
            filename_suffix=filename_suffix,
        )


def exp_triangles_to_f1(
    graph_name
):
    num_hop = 2
    dict_num_triangles = read_two_value_csv(
        "output/" + graph_name + "-2-hop-triangles.csv")
    local_methods = [
        str(num_hop) + "-hop-appr",
        str(num_hop) + "-hop-mappr",
        str(num_hop) + "-hop-louvain",
        str(num_hop) + "-hop-louvain0.125",
        str(num_hop) + "-hop-louvain8"
    ]
    minsize, q1size, medsize, q3size, maxsize = compute_min_q1_median_q3_max(
        [cnt for cnt in dict_num_triangles.values()]
    )
    metric = "f1"
    filter = "num_triangles"
    for Q in [True, False]:
        if Q:
            seeds_lt_q1 = []
            seeds_gt_q3 = []
        else:
            seeds_lt_median = []
            seeds_gt_median = []
        for seed, triangle_cnt in dict_num_triangles.items():
            if Q:
                if triangle_cnt < q1size:
                    seeds_lt_q1.append(seed)
                if triangle_cnt >= q3size:
                    seeds_gt_q3.append(seed)
            else:
                if triangle_cnt < medsize:
                    seeds_lt_median.append(seed)
                if triangle_cnt >= medsize:
                    seeds_gt_median.append(seed)
        if Q:
            list_seeds = [seeds_lt_q1, seeds_gt_q3]
            types = ["< Q1", ">= Q3"]
            title_suffix = filter + " compared w/ Q1 & Q3"
            filename_suffix = filter + "-q1q3"
        else:
            list_seeds = [seeds_lt_median, seeds_gt_median]
            types = ["< median", ">= median"]
            title_suffix = filter + " compared w/ median"
            filename_suffix = filter + "-median"
        create_fig_with_multiple_types(
            list_seeds,
            types,
            labels=local_methods,
            metric=metric,
            filter="num_triangles",
            thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            title_suffix=title_suffix,
            filename_suffix=filename_suffix,
        )


def exp_num_edges_to_f1(
        graph_name
):
    num_hop = 2
    g = read_graph("graph/" + graph_name + ".gr")
    num_edges_sub = {}
    for seed in g.nodes():
        subg = create_n_hop_graph(g, seed, num_hop)
        num_edges_sub[seed] = subg.number_of_edges()
    local_methods = [
        str(num_hop) + "-hop-appr",
        str(num_hop) + "-hop-mappr",
        str(num_hop) + "-hop-louvain",
        str(num_hop) + "-hop-louvain0.125",
        str(num_hop) + "-hop-louvain8"
    ]
    minsize, q1size, medsize, q3size, maxsize = compute_min_q1_median_q3_max(
        [cnt for cnt in num_edges_sub.values()]
    )
    metric = "f1"
    filter = "num_edges"
    for Q in [True, False]:
        if Q:
            seeds_lt_q1 = []
            seeds_gt_q3 = []
        else:
            seeds_lt_median = []
            seeds_gt_median = []
        for seed, triangle_cnt in num_edges_sub.items():
            if Q:
                if triangle_cnt < q1size:
                    seeds_lt_q1.append(seed)
                if triangle_cnt >= q3size:
                    seeds_gt_q3.append(seed)
            else:
                if triangle_cnt < medsize:
                    seeds_lt_median.append(seed)
                if triangle_cnt >= medsize:
                    seeds_gt_median.append(seed)
        if Q:
            list_seeds = [seeds_lt_q1, seeds_gt_q3]
            types = ["< Q1", ">= Q3"]
            title_suffix = filter + " compared w/ Q1 & Q3"
            filename_suffix = filter + "-q1q3"
        else:
            list_seeds = [seeds_lt_median, seeds_gt_median]
            types = ["< median", ">= median"]
            title_suffix = filter + " compared w/ median"
            filename_suffix = filter + "-median"
        create_fig_with_multiple_types(
            list_seeds,
            types,
            labels=local_methods,
            metric=metric,
            filter=filter,
            thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            title_suffix=title_suffix,
            filename_suffix=filename_suffix,
        )


def exp_chart_f1(
    graph_name,
    local_methods=None
):
    num_hop = 2
    subg = read_clusters("cluster/" + graph_name + "/sub-" +
                         str(num_hop) + "-hop.output")
    if local_methods is None:
        local_methods = [
            str(num_hop) + "-hop-louvain0.01",
            str(num_hop) + "-hop-louvain0.032",
            str(num_hop) + "-hop-louvain0.1",
            str(num_hop) + "-hop-louvain0.32",
            str(num_hop) + "-hop-sub",
        ]
    metric = "f1"
    seeds = []
    for seed in subg.keys():
        seeds.append(seed)
    create_fig_with_specific_seeds(
        seeds,
        graph_name,
        labels=local_methods,
        metric=metric,
        thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    )


def exp_f1_box_distribution(
    graph_name,
    local_methods=None,
):
    num_hop = 2
    subg = read_clusters("cluster/" + graph_name + "/sub-" +
                         str(num_hop) + "-hop.output")
    if local_methods is None:
        local_methods = [
            str(num_hop) + "-hop-appr",
            # str(num_hop) + "-hop-louvain0.125",
            # str(num_hop) + "-hop-louvain0.25",
            # str(num_hop) + "-hop-louvain0.5",
            # str(num_hop) + "-hop-louvain0.6",
            str(num_hop) + "-hop-louvain0.75",
            # str(num_hop) + "-hop-louvain",
            str(num_hop) + "-hop-mappr",
            # str(num_hop) + "-hop-louvain1.8",
            str(num_hop) + "-hop-louvain2",
            # str(num_hop) + "-hop-louvain4",
            # str(num_hop) + "-hop-louvain8"
        ]
    metric = "f1"
    read_and_compute_f1_box_points(
        local_methods,
        graph_name,
        metric=metric,
        global_choice="seed",
        valid_thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    )


def count_subg_triangles(
        graph_name
):
    num_hop = 2
    g = read_graph("graph/" + graph_name + ".gr")
    f = open("output/" + graph_name + "-" +
             str(num_hop) + "-hop-triangles.csv", "w")
    f.write("seed,#triangles\n")
    cnt = 0
    percent = g.number_of_nodes() // 100
    for seed in g.nodes():
        if cnt % percent == 0:
            print(cnt // percent, "%")
        cnt += 1
        subg = create_n_hop_graph(g, seed, num_hop)
        trid = nx.triangles(subg)
        num_triangles = sum(trid.values()) // 3
        f.write(f'{seed},{num_triangles}\n')
    f.close()


def exp_hopsize_box_distribution(
    graph_name,
    local_methods=None,
):
    num_hop = 2
    if local_methods is None:
        local_methods = [
            str(num_hop) + "-hop-appr",
            str(num_hop) + "-hop-mappr",
            str(num_hop) + "-hop-louvain0.125",
            str(num_hop) + "-hop-louvain0.25",
            str(num_hop) + "-hop-louvain0.5",
            str(num_hop) + "-hop-louvain0.6",
            str(num_hop) + "-hop-louvain0.75",
            str(num_hop) + "-hop-louvain",
            str(num_hop) + "-hop-louvain1.8",
            str(num_hop) + "-hop-louvain2",
            str(num_hop) + "-hop-louvain4",
            str(num_hop) + "-hop-louvain8"
        ]
    list_cluster_sizes = []
    for local_method in local_methods:
        cluster_sizes = []
        clusters = read_clusters(
            "cluster/" + graph_name + "/all-" + local_method + ".output")
        for cluster in clusters.values():
            cluster_sizes.append(len(cluster))
        list_cluster_sizes.append(cluster_sizes)
    draw_boxplot(
        list_cluster_sizes,
        "2-hop local cluster size distribution",
        labels=list(map(lambda x: x[6:], local_methods)),
        y_axis_title="2-hop local cluster size",
        rotation=15,
        show_graph=True
    )


def chart_global_cluster(
    graph_name,
    local_methods=[
        "2-hop-appr",
        "2-hop-mappr",
        "2-hop-louvain",
    ]
):
    num_hop = 2
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    y_axes = []

    subgnodes = read_clusters(
        "cluster/" + graph_name + "/sub-" + str(num_hop) + "-hop.output")
    sum_subg_size = 0
    for subg in subgnodes.values():
        sum_subg_size += len(subg)
    y_axes.append([sum_subg_size/len(subgnodes)] * len(thresholds))

    aves = []
    list_sizes = []
    for threshold in thresholds:
        communities = read_non_seeded_clusters(
            "cluster/" + graph_name + "/k-means-" + str(threshold) + ".output")
        sizes = []
        for seed, community in communities.items():
            subg = subgnodes[seed]
            hop_global_cluster = community.intersection(subg)
            sizes.append(len(hop_global_cluster))
        list_sizes.append(sizes)
        ave = sum(sizes) / len(communities)
        aves.append(ave)
    y_axes.append(aves)

    average_of_global = sum(aves) / len(aves)
    aves_of_global = [average_of_global] * len(thresholds)
    y_axes.append(aves_of_global)

    for local_method in local_methods:
        cluster_sizes = []
        clusters = read_clusters(
            "cluster/" + graph_name + "/all-" + local_method + ".output")
        for cluster in clusters.values():
            cluster_sizes.append(len(cluster))
        ave = sum(cluster_sizes) / len(cluster_sizes)
        aves = [ave] * len(thresholds)
        y_axes.append(aves)
        list_sizes.append(cluster_sizes)

    draw_boxplot(
        list_sizes,
        str(num_hop) + "-hop global cluster size distribution",
        labels=["global-" + str(threshold)
                for threshold in thresholds] + local_methods,
        # x_axis_title="conductance threshold",
        y_axis_title=str(num_hop) + "-hop cluster size",
        filename="tmp/cluster-size-distribution.png",
        rotation=15,
        show_graph=True
    )

    labels = ["2-hop-size", "2-hop global cluster",
              "2-hop-global cluster average"] + local_methods
    draw_chart(
        thresholds,
        y_axes,
        title="conductance threshold vs average " +
        str(num_hop) + "-hop global cluster size",
        labels=labels,
        x_axis_title="conductance threshold",
        y_axis_title="average " + str(num_hop) + "-hop global cluster size",
        xscale="linear",
        yscale="linear",
        right=0.41,
        left=0,
        top=None,
        bottom=0,
        colors=None,
        linestyles=None,
        filename="tmp/" + str(num_hop) + "-hop-cluster-size.png",
        show_graph=True,
    )


def create_new_louvain_dataset(
    resolution,
    graph_name,
    num_hop=2
):
    g = read_graph("graph/" + graph_name + ".gr")
    compute_hop_louvain_resolution(
        graph_name=graph_name,
        g=g,
        resolution=resolution,
        hops=[num_hop],
    )
    exp_local_global_precision_recall_f1_with_hop(
        g=g,
        graph_name=graph_name,
        objective="k-means",
        local_method="louvain" + str(resolution),
        global_choice="seed",
        hop=num_hop,
    )
    # local_methods = ["2-hop-louvain" + str(resolution), "2-hop-louvain0.25"]
    # exp_hopsize_box_distribution(local_methods=local_methods)
    # exp_chart_f1(local_methods=local_methods)


def create_new_appr_dataset(
    total_vol,
    graph_name,
    hop=2,
    alpha=0.98,
    eps_exp=-4,
):
    g = read_graph("graph/" + graph_name + ".gr")
    compute_hop_appr(
        graph_name,
        g,
        hop,
        alpha,
        eps_exp,
        total_vol
    )
    local_method = str(hop) + "-hop-appr-" + str(alpha) + \
        "-10e" + str(eps_exp) + "-" + str(total_vol)
    exp_local_global_precision_recall_f1_with_hop(
        g=g,
        graph_name=graph_name,
        objective="k-means",
        local_method=local_method[6:],
        global_choice="seed",
        hop=hop,
    )
    local_methods = [
        local_method,
        # "2-hop-appr"
    ]
    chart_global_cluster(
        graph_name,
        local_methods=local_methods + [
            # "2-hop-mappr",
            "2-hop-louvain1",
        ]
    )
    exp_hopsize_box_distribution(graph_name, local_methods=local_methods)


def create_new_mappr_dataset(
    total_vol,
    graph_name
):
    g = read_graph("graph/" + graph_name + ".gr")
    hop = 2
    alpha = 0.98
    eps_exp = -4
    compute_hop_mappr(
        graph_name,
        g,
        hop,
        alpha,
        eps_exp,
        total_vol
    )
    local_method = str(hop) + "-hop-mappr-" + str(alpha) + \
        "-10e" + str(eps_exp) + "-" + str(total_vol)
    exp_local_global_precision_recall_f1_with_hop(
        g=g,
        graph_name=graph_name,
        objective="k-means",
        local_method=local_method[6:],
        global_choice="seed",
        hop=hop,
    )
    local_methods = [
        local_method,
        # "2-hop-mappr"
    ]
    chart_global_cluster(
        graph_name,
        local_methods=local_methods + [
            # "2-hop-appr",
            "2-hop-louvain1",
        ]
    )
    exp_hopsize_box_distribution(graph_name, local_methods=local_methods)


def chart_same_size_global_local_clusters(graph_name):
    local_methods = [
        "2-hop-louvain0.305",
        "2-hop-appr-0.98-10e-4-1700",
        "2-hop-mappr-0.98-10e-4-523896",
    ]
    chart_global_cluster(graph_name, local_methods)
    exp_chart_f1(graph_name, local_methods)
    exp_f1_box_distribution(graph_name, local_methods)
    pass


def louvain_multiple_resolution(
    graph_name,
):
    resolutions = [0.0032, 0.032, 0.32]
    hop = 2
    local_methods = []
    for resolution in resolutions:
        local_method = str(hop) + "-hop-louvain" + str(resolution)
        local_methods.append(local_method)
        filepath = "cluster/" + graph_name + "/all-" + \
            local_method + ".output"
        if os.path.isfile(filepath):
            print("exists!")
            continue
        create_new_louvain_dataset(resolution, graph_name)
    local_methods.append("2-hop-sub")
    chart_global_cluster(graph_name, local_methods)
    exp_chart_f1(graph_name, local_methods)


def appr_multiple_total_vol(graph_name):
    g = read_graph("graph/" + graph_name + ".gr")
    total_vols = [2500, 5000, 10000, 20000]

    alpha = 0.98
    eps_exp = -4
    hop = 2
    local_methods = []
    for total_vol in total_vols:
        local_method = str(hop) + "-hop-appr-" + str(alpha) + \
            "-10e" + str(eps_exp) + "-" + str(total_vol)
        local_methods.append(local_method)
        filepath = "cluster/" + graph_name + "/all-" + \
            local_method + ".output"
        if os.path.isfile(filepath):
            print("exists!")
            continue
        create_new_appr_dataset(total_vol, graph_name)
    local_methods.append("2-hop-sub")
    chart_global_cluster(graph_name, local_methods)
    exp_chart_f1(graph_name, local_methods)


def mappr_multiple_total_vol(graph_name):
    g = read_graph("graph/" + graph_name + ".gr")
    trid = nx.triangles(g)
    num_triangles = sum(trid.values())
    total_vols = [1500, 1750, 2000, num_triangles]
    alpha = 0.98
    eps_exp = -4
    hop = 2
    local_methods = []
    for total_vol in total_vols:
        local_method = str(hop) + "-hop-mappr-" + str(alpha) + \
            "-10e" + str(eps_exp) + "-" + str(total_vol)
        local_methods.append(local_method)
        filepath = "cluster/" + graph_name + "/all-" + \
            local_method + ".output"
        if os.path.isfile(filepath):
            print("exists!")
            continue
        create_new_mappr_dataset(total_vol, graph_name)
    local_methods.append("2-hop-sub")
    chart_global_cluster(graph_name, local_methods)
    exp_chart_f1(graph_name, local_methods)


def louvain_count_variety_cluster_size(graph_name):
    g = read_graph("graph/" + graph_name + ".gr")
    hop = 2
    f = open("output/" + graph_name +
             "-louvain-cluster-size_distribution.csv", "w")
    f.write(f'seed,2-hop-size,same_cnt,fraction\n')

    def is_ok(size, expected_size):
        if size >= expected_size:
            return True
        return False

    nodes = [nd for nd in g]
    shuffle(nodes)
    for i, seed in enumerate(nodes):
        subg = create_n_hop_graph(g, seed, hop)
        d = {}
        same_cnt = 0
        for expected_size in range(2, subg.number_of_nodes()):
            left = 0
            right = subg.number_of_edges()
            cnt = 0
            while (right > left):
                mid = (right + left) / 2
                if mid in d:
                    size = d[mid]
                else:
                    louvain_communities = nx.community.louvain_communities(
                        subg, resolution=mid)
                    for c in louvain_communities:
                        if seed in c:
                            size = len(c)
                            break
                    d[mid] = size
                if size == expected_size:
                    same_cnt += 1
                    break
                if (is_ok(size, expected_size)):
                    left = mid
                else:
                    right = mid
                cnt += 1
                if cnt >= 1000000:
                    break

            print("expected_size: ", expected_size, ", size: ",
                  size, ", resolution: ", mid, sep='')
        print("seed: ", seed, ", same: ", same_cnt,
              "(", round(same_cnt / subg.number_of_nodes() * 100, 2), "%)", sep='')
        print(
            f'{seed},{subg.number_of_nodes()},{same_cnt},{same_cnt / subg.number_of_nodes()}\n')
        f.write(
            f'{seed},{subg.number_of_nodes()},{same_cnt},{same_cnt / subg.number_of_nodes()}\n')


def appr_count_variety_cluster_size(graph_name):
    g = read_graph("graph/" + graph_name + ".gr")
    hop = 2
    f = open("output/" + graph_name +
             "-appr-cluster-size_distribution.csv", "w")
    f.write(f'seed,2-hop-size,same_cnt,fraction\n')

    def is_ok(size, expected_size):
        if size >= expected_size:
            return True
        return False

    nodes = [nd for nd in g]
    shuffle(nodes)
    for i, seed in enumerate(nodes):
        subg = create_n_hop_graph(g, seed, hop)
        d = {}
        same_cnt = 0
        for expected_size in range(2, subg.number_of_nodes()):
            left = 0
            right = g.number_of_edges()
            cnt = 0
            while (right > left):
                mid = (right + left) / 2
                if mid in d:
                    size = d[mid]
                else:
                    appr = APPR(subg)
                    community = appr.compute_appr(seed, total_vol=mid)
                    size = len(community)
                    d[mid] = size
                if size == expected_size:
                    same_cnt += 1
                    break
                if (is_ok(size, expected_size)):
                    right = mid
                else:
                    left = mid
                cnt += 1
                if cnt >= 1000000:
                    break

            print("expected_size: ", expected_size, ", size: ",
                  size, ", total_vol: ", mid, sep='')
        print(
            f'{seed},{subg.number_of_nodes()},{same_cnt},{same_cnt / subg.number_of_nodes()}\n')
        f.write(
            f'{seed},{subg.number_of_nodes()},{same_cnt},{same_cnt / subg.number_of_nodes()}\n')


def mappr_count_variety_cluster_size(graph_name):
    g = read_graph("graph/" + graph_name + ".gr")
    trid = nx.triangles(g)
    num_triangles = sum(trid.values()) // 3
    hop = 2
    f = open("output/" + graph_name +
             "-mappr-cluster-size_distribution.csv", "w")
    f.write(f'seed,2-hop-size,same_cnt,fraction\n')

    def is_ok(size, expected_size):
        if size >= expected_size:
            return True
        return False

    nodes = [nd for nd in g]
    shuffle(nodes)
    for i, seed in enumerate(nodes):
        subg = create_n_hop_graph(g, seed, hop)
        d = {}
        same_cnt = 0
        for expected_size in range(2, subg.number_of_nodes()):
            left = 0
            right = num_triangles
            cnt = 0
            while (right > left):
                mid = (right + left) / 2
                if mid in d:
                    size = d[mid]
                else:
                    appr = MAPPR(subg, clique3)
                    community = appr.compute_appr(seed, total_vol=mid)
                    size = len(community)
                    d[mid] = size
                if size == expected_size:
                    same_cnt += 1
                    break
                if (is_ok(size, expected_size)):
                    right = mid
                else:
                    left = mid
                cnt += 1
                if cnt >= 1000000:
                    break

            print("expected_size: ", expected_size, ", size: ",
                  size, ", total_vol: ", mid, sep='')
        print(
            f'{seed},{subg.number_of_nodes()},{same_cnt},{same_cnt / subg.number_of_nodes()}\n')
        f.write(
            f'{seed},{subg.number_of_nodes()},{same_cnt},{same_cnt / subg.number_of_nodes()}\n')


def cluster_size_control_distribution(graph_name):
    local_methods = [
        "2-hop-louvain",
        "2-hop-appr",
        "2-hop-mappr"
    ]
    data = []
    for i, local_method in enumerate(local_methods):
        filename = "output/" + graph_name + "-" + \
            local_method + "-cluster-size_distribution.csv"
        f = open(filename)
        data.append([])
        for line in f.readlines()[1:]:
            fraction = float(line.split(",")[3]) * 100
            data[i].append(fraction)
    draw_boxplot(data,
                 labels=local_methods,
                 y_axis_title="%controllable cluster size",
                 top=100,
                 bottom=0,
                 showfliers=False
                 )


def merge_local_clusters(graph_name):
    g = read_graph("graph/" + graph_name + ".gr")
    local_clusters = read_clusters(
        "cluster/" + graph_name + "/all-appr.output")
    hop = 2
    num_seeds = g.number_of_nodes()
    percent = num_seeds // 100
    if percent == 0:
        percent = 1
    cnt = 0
    start = time.time()
    pairs = []
    for seed, local_cluster in local_clusters.items():
        for nd in local_cluster:
            if nd < seed:
                continue
            if seed in local_clusters[nd]:
                pairs.append((seed, nd))

        cnt += 1
        if cnt % percent == 0:
            now = time.time()
            done_percent = cnt // percent
            current = now - start
            reamaining_time = current / done_percent * (100 - done_percent)
            print(cnt // percent, "% done; ",
                  round(reamaining_time, 1), "seconds remaining")
    clusters_dict = {nd: nd for nd in local_clusters.keys()}
    pairs.sort()
    for (u, v) in pairs:
        if u == 8310 or v == 8310:
            print(u, v, clusters_dict[v], "to", clusters_dict[u])
        clusters_dict[v] = clusters_dict[u]
    from collections import defaultdict
    clusters = defaultdict(set)
    for nd, cluster_id in clusters_dict.items():
        clusters[cluster_id].add(nd)
    sizes = []
    for cluster_id, cluster in clusters.items():
        # print(cluster_id, ": ", len(cluster), sep='')
        if len(cluster) == 1771:
            # print(cluster)
            c = cluster
        sizes.append(len(cluster))
    # print(c)
    subg: nx.Graph = g.subgraph(c)
    print(subg)
    for c in sorted(nx.connected_components(subg), key=len, reverse=True):
        if len(c) == 1:
            print(c)
            break


def draw_japan_map(
    path="img/tmp.pdf",
    color: dict = None
):
    import matplotlib.pyplot as plt
    pct = picture(color)  # numpy.ndarray
    # pct = picture({1: 'blue'})  # same to above
    plt.imshow(pct)  # show graphics
    plt.savefig(path)  # save to PNG file


def prefectures():
    g = read_graph("graph/prefectures.gr")
    g = nx.relabel_nodes(g, {i: v for i, v in enumerate(pref_names)})

    # Louvain
    from community import community_louvain
    partition = community_louvain.best_partition(g, resolution=1)

    # # Gloabl Clustering
    # cluster2nodes = read_clusters("cluster/prefectures/k-means-0.1.output")
    # renumbered = {}
    # for c in cluster2nodes.keys():
    #     if c not in renumbered:
    #         renumbered[c] = len(renumbered)
    # partition = {}
    # for c, nds in cluster2nodes.items():
    #     for nd in nds:
    #         pref = pref_names[nd]
    #         partition[pref] = renumbered[c]

    draw_map = False

    if draw_map:
        # draw japan map
        color = {}
        for nd, c in partition.items():
            color[nd] = tuple(map(lambda x: x * 256, list(COLORS[c])))
        draw_japan_map(color=color)
    else:
        # draw graph
        axes = [
            (606, 10),  # 1 北海道
            (617, 70),  # 2 青森
            (665, 128),
            (666, 199),
            (608, 131),  # 5 宮城
            (596, 196),
            (644, 265),  # 7 福島
            (713, 340),  # 8 茨城
            (655, 330),  # 9 栃木
            (608, 323),
            (649, 381),
            (682, 448),
            (635, 437),  # 13 東京
            (627, 486),
            (575, 274),  # 15 新潟
            (515, 304),
            (464, 294),
            (415, 324),
            (582, 426),
            (555, 366),
            (477, 352),
            (534, 464),
            (496, 412),
            (446, 409),
            (426, 372),
            (370, 373),  # 26 京都
            (333, 422),
            (316, 368),  # 28 兵庫
            (384, 422),
            (395, 475),  # 30 和歌山
            (275, 304),
            (225, 290),
            (266, 351),  # 33 岡山
            (216, 337),
            (164, 323),
            (262, 429),
            (253, 387),  # 37 香川
            (203, 393),
            (218, 438),
            (116, 337),
            (74, 323),
            (36, 291),
            (95, 372),
            (155, 371),
            (155, 427),
            (90, 433),
            (76, 502),  # 47 沖縄
        ]

        axes = normalize_axes(axes)
        pos = {}
        for i, axis in axes.items():
            pos[pref_names[i + 1]] = axis

        # color
        node_color = [LIGHT_COLORS[partition[nd]] for nd in g]

        draw_graph(
            g,
            pos=pos,
            node_size=600,
            figsize=(12.9, 9),
            node_color=node_color,
            font_family='Hiragino Maru Gothic Pro'
        )
    pass


def example_graph():
    g = read_graph("graph/example.gr")
    axes = [
        (981, 636),
        (802, 546),
        (860, 422),
        (741, 453),
        (652, 542),
        (676, 667),
        (799, 303),
        (682, 363),
        (557, 397),
        (501, 517),
        (380, 635),
        (853, 185),
        (675, 242),
        (562, 301),
        (402, 399),
        (292, 454),
        (227, 572),
        (706, 124),
        (595, 184),
        (437, 276),
        (316, 301),
        (157, 330),
        (55, 482),
        (595, 68),
        (475, 120),
        (291, 154),
        (77, 218),
        (358, 31)
    ]
    axes = normalize_axes(axes)
    pos = {}
    for i, axis in axes.items():
        pos[i + 1] = axis

    from community import community_louvain
    partition = community_louvain.best_partition(g, resolution=1)
    node_color = [LIGHT_COLORS[partition[nd]] for nd in g]

    draw_graph(
        g,
        pos=pos,
        path="tmp/sparse.pdf",
        node_size=750,
        figsize=(8, 6),
        node_color=node_color,
        font_family='Hiragino Kaku Gothic Pro'
    )

    g2 = read_graph("graph/example2.gr")

    g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    nodes = [nd for nd in g2]
    for nd in g2:
        for _ in range(17):
            nd2 = random.choice(nodes)
            while nd == nd2 or g.has_edge(nd, nd2):
                nd2 = random.choice(nodes)
            g2.add_edge(nd, nd2)

    from community import community_louvain
    partition = community_louvain.best_partition(g2, resolution=1)
    partition = {1: 2, 2: 0, 3: 2, 4: 0, 8: 1, 5: 1, 6: 0, 10: 1, 7: 0, 12: 0, 9: 2, 11: 0, 13: 1,
                 14: 0, 15: 0, 16: 2, 21: 0, 22: 1, 17: 1, 23: 0, 19: 2, 24: 1, 20: 0, 25: 2, 26: 1, 27: 1, 28: 2}

    node_color = [LIGHT_COLORS[partition[nd]] for nd in g2]

    draw_graph(
        g2,
        pos=pos,
        path="tmp/dense.pdf",
        node_size=750,
        figsize=(8, 6),
        node_color=node_color,
        font_family='Hiragino Kaku Gothic Pro'
    )

    g3 = nx.Graph()
    g3.add_edges_from(g.edges())
    g3.add_edges_from(g2.edges())

    from community import community_louvain
    partition = community_louvain.best_partition(g3, resolution=1)
    partition = {1: 0, 2: 1, 3: 0, 4: 0, 8: 2, 5: 2, 6: 1, 10: 2, 9: 0, 11: 1, 7: 1, 12: 1, 13: 2,
                 14: 1, 15: 1, 16: 2, 21: 1, 22: 2, 17: 2, 23: 1, 19: 0, 24: 2, 20: 1, 25: 0, 26: 2, 27: 0, 28: 0}
    converter = {0: 2, 1: 0, 2: 1}
    node_color = [LIGHT_COLORS[converter[partition[nd]]] for nd in g3]

    draw_graph(
        g3,
        pos=pos,
        path="tmp/merge.pdf",
        node_size=750,
        figsize=(8, 6),
        node_color=node_color,
        font_family='Hiragino Kaku Gothic Pro'
    )
    pass


def export_graph(g: nx.Graph, path):
    f = open(path, "w")
    s = set()
    for u, v in g.edges():
        s.add(u)
        s.add(v)
        line = f'{u} {v}\n'
        f.write(line)
    for nd in g.nodes():
        if nd in s:
            continue
        s.add(nd)
        line = f'{nd}\n'
        f.write(line)
    f.close()


def divide_graph():
    path = "graph/socfb-Texas80.gr"
    graph_name = get_gname(path)
    g = read_graph(path)

    for overlapping_perentage in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]:
        num_overlapping_edges = int(
            overlapping_perentage * g.number_of_edges())
        a = nx.Graph()
        b = nx.Graph()
        a.add_nodes_from(g.nodes())
        b.add_nodes_from(g.nodes())
        nds = [nd for nd in g.nodes()]
        nds.sort(key=lambda x: g.degree(x), reverse=True)
        for nd in nds:
            nbrs = list(g.neighbors(nd))
            overlapping_endpoints = sample(nbrs, 8 * len(nbrs) // 10)
            if num_overlapping_edges > len(overlapping_endpoints):
                num_overlapping_edges -= len(overlapping_endpoints)
            else:
                overlapping_endpoints = sample(
                    overlapping_endpoints, num_overlapping_edges)
                num_overlapping_edges = 0
            set_overlapping = set(overlapping_endpoints)
            # add overlapping edge to both a and b
            for nbr in overlapping_endpoints:
                a.add_edge(nd, nbr)
                b.add_edge(nd, nbr)
            for nbr in nbrs:
                if nbr in set_overlapping:
                    continue
                if nd > nbr:
                    continue
                if random.random() < 0.5:
                    a.add_edge(nd, nbr)
                else:
                    b.add_edge(nd, nbr)
        export_graph(a, f"graph/{graph_name}-a-{overlapping_perentage}.gr")
        export_graph(b, f"graph/{graph_name}-b-{overlapping_perentage}.gr")
    # from community import community_louvain
    # partition = community_louvain.best_partition(g, resolution=1)
    pass


def louvain_overlap():
    path = "graph/socfb-Texas80.gr"
    graph_name = get_gname(path)
    g = read_graph(path)
    from community import community_louvain
    partition = community_louvain.best_partition(g, resolution=1)
    f = open("cluster/" + graph_name + "/merged.cluster", "w")
    for nd, c in partition.items():
        f.write(f'{nd} {c}\n')
    f.close()

    for overlapping_perentage in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]:
        a = read_graph(f"graph/{graph_name}-a-{overlapping_perentage}.gr")
        f = open(
            f"cluster/{graph_name}/a-{overlapping_perentage}.cluster", "w")
        partition = community_louvain.best_partition(a, resolution=1)
        for nd, c in partition.items():
            f.write(f'{nd} {c}\n')
        f.close()

        b = read_graph(f"graph/{graph_name}-b-{overlapping_perentage}.gr")
        f = open(
            f"cluster/{graph_name}/b-{overlapping_perentage}.cluster", "w")
        partition = community_louvain.best_partition(b, resolution=1)
        for nd, c in partition.items():
            f.write(f'{nd} {c}\n')
        f.close()
    pass


def read_louvain_cluster(path):
    f = open(path)
    nd2cluster = {}
    cluster2nodes = defaultdict(set)
    for line in f.readlines():
        nd, c = map(int, line.split())
        nd2cluster[nd] = c
        cluster2nodes[c].add(nd)
    f.close()
    return nd2cluster, cluster2nodes


def dnp20220826():
    path = "graph/socfb-MSU24.gr"
    graph_name = get_gname(path)
    nd2mergedcluster, mergedcluster2nodes = read_louvain_cluster(
        f"cluster/{graph_name}/merged.cluster")
    data = []
    for overlapping_perentage in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]:
        a = read_graph(f"graph/{graph_name}-b-{overlapping_perentage}.gr")
        nd2acluster, acluster2nodes = read_louvain_cluster(
            f"cluster/{graph_name}/b-{overlapping_perentage}.cluster")
        f1s = []
        f1dict = {}
        for nd in nd2acluster.keys():
            acluster_id = nd2acluster[nd]
            acluster = acluster2nodes[acluster_id]
            mergedcluster_id = nd2mergedcluster[nd]
            mergedcluster = mergedcluster2nodes[mergedcluster_id]
            if (acluster_id, mergedcluster_id) in f1dict:
                f1 = f1dict[(acluster_id, mergedcluster_id)]
            else:
                f1 = compute_precision_recall_f1(acluster, mergedcluster)[2]
                f1dict[(acluster_id, mergedcluster_id)] = f1
            f1s.append(f1)
        data.append(f1s)
    draw_boxplot(
        data,
        title="",
        labels=["1%", "5%", "10%", "25%", "50%", "75%"],
        x_axis_title="edge overlapping percentage",
        y_axis_title="f1",
        top=None,
        bottom=None,
        rotation=None,
        filename="tmp/tmp.png",
        show_graph=False,
        showfliers=False,
    )
    pass


# https://github.com/TeraokaKanekoLab/thenter-journal/blob/master/2022/08/0829.md#趣味グラフ
def create_hobby_graph():
    hobbies = [
        "game",
        "sport",
        "book",
        "travel",
        "pc/internet",
        "anime/comic",
        "movie",
        "music",
        "groumet",
        "gamble"
    ]
    node_ids = [i for i in range(10000)]
    with_hobby_ids = {nd: [] for nd in sample(node_ids, 9080)}
    ids = list(with_hobby_ids.keys())
    points = [1750, 1670, 1290, 1040, 920, 920, 880, 830, 830, 750]
    for nd in with_hobby_ids.keys():
        current_sum = sum(points)
        rndint = random.randint(0, current_sum - 1)
        for i in range(len(points)):
            if rndint < points[i]:
                with_hobby_ids[nd].append(i)
                points[i] -= 1
                break
            else:
                rndint -= points[i]
    for hobby_id, point in enumerate(points):
        for _ in range(point):
            nd = random.choice(ids)
            while hobby_id in with_hobby_ids[nd] or len(with_hobby_ids[nd]) >= 3:
                nd = random.choice(ids)
            with_hobby_ids[nd].append(hobby_id)
    hobby2nodes = {i: [] for i in range(10)}
    for nd, hobbies in with_hobby_ids.items():
        for hobby_id in hobbies:
            hobby2nodes[hobby_id].append(nd)
    g = nx.Graph()
    g.add_nodes_from([nd for nd in node_ids])
    for hobid, nodes in hobby2nodes.items():
        for u in nodes:
            for v in nodes:
                if u < v:
                    g.add_edge(u, v)
    print(g)
    export_graph(g, "graph/dnp_hobby.gr")


def exp_similarity_compare_global_clustering():
    # ga_inner_densities = [float("0." + str(i)) for i in range(1, 6)]
    # ga_inner_densities = [float("0." + str(i)) for i in range(10, 51)]
    # ga_inner_densities = [float("0." + str(i)) for i in range(29, 32)]
    ga_inner_densities = [0.3]
    inter_density = 0.01
    num_communities = 3
    community_size = 100
    directory = "tmp/graph"

    clustering_label = "global_clusteirng"

    THRESHOLD = 0.4
    gb = read_graph(
        f"{directory}/{0.3}-{inter_density}-{num_communities}-{community_size}-mixedorder.gr"
    )
    cb = compute_global_cluster(gb, THRESHOLD)

    nmis_a, nmis_b = [], []
    infls_a, infls_b = [], []
    eds_a, eds_b = [], []
    eta = ETA()
    for cnt, ga_inner_density in enumerate(ga_inner_densities):
        ga = sbm(
            inter_density,
            ga_inner_density,
            num_communities,
            community_size,
            seed=None
        )
        # nga, ngb = normalize_edge_weight(ga, gb)
        nga, ngb = ga, gb
        gm = merge_two_graphs(nga, ngb)
        ca = compute_global_cluster(ga, THRESHOLD)
        cm = compute_global_cluster(gm, THRESHOLD)

        nmis_a.append(normalized_mutual_information(ca, cm))
        infls_a.append(f_score(ca, cm))
        eds_a.append(edit_distance(ca, cm))
        nmis_b.append(normalized_mutual_information(cb, cm))
        infls_b.append(f_score(cb, cm))
        eds_b.append(edit_distance(cb, cm))
        print(ga_inner_density, "done, eta:", eta.eta(
            (cnt+1) / len(ga_inner_densities)))

    filenames = ["tmp/nmi", "tmp/f", "tmp/ed"]
    for i in range(len(filenames)):
        filenames[i] += f"-{clustering_label}-{num_communities}-{community_size}.png"

    draw_scatter(
        ga_inner_densities,
        [nmis_a, nmis_b],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)"
        ],
        title=f"NMI (strategy: {strategy})",
        x_axis_title="inner-edge density of graph A",
        y_axis_title="NMI",
        top=1.01,
        bottom=-0.01,
        filename=filenames[0],
    )
    draw_scatter(
        ga_inner_densities,
        [infls_a, infls_b],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)"
        ],
        title=f"F-score (strategy: {strategy})",
        x_axis_title="inner-edge density of graph A",
        y_axis_title="F-score",
        top=1.01,
        bottom=-0.01,
        filename=filenames[1],
    )
    draw_scatter(
        ga_inner_densities,
        [eds_a, eds_b],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)",
        ],
        title=f"Edit distance (strategy: {strategy})",
        x_axis_title="inner-edge density of graph A",
        y_axis_title="Edit distance",
        bottom=0,
        filename=filenames[2],
    )

    filename = concatanate_images(
        filenames, f"tmp/{clustering_label}", 2, 2)
    image_url = upload_to_imgbb(filename)
    notify_slack(
        title=f'ga_inner_densities: {min(ga_inner_densities)}-{max(ga_inner_densities)}, clustering: {clustering_label}',
        result=image_url
    )


def weight_edge_by_appr():
    # ga_inner_densities = [float("0." + str(i)) for i in range(100, 501, 10)]
    ga_inner_densities = [i for i in range(1, 11)]
    num_communities = 3
    community_size = 100
    gb = read_graph(
        # f"tmp/graph/{0.3}-{0.01}-{num_communities}-{community_size}-mixedorder.gr"
        f"graph/barabasi-albert/ba-10000-{5}.gr"
    )
    gb = randomize_graph_id(gb)
    SEED = 200
    clustering_label = "APPR"
    STRATEGIES = [0, 1, 2, 3]
    STRATEGY = STRATEGIES[3]

    apprb = APPR(gb)
    cb = set(apprb.compute_appr(SEED))
    ros_a, ros_b = [], []
    fs_a, fs_b = [], []
    csize_a, csize_b, csize_m = [], [], []
    eta = ETA()
    for cnt, ga_inner_density in enumerate(ga_inner_densities):
        ga = read_graph(
            # f"tmp/graph/{ga_inner_density}-{0.01}-3-100-normalorder.gr"
            f"graph/barabasi-albert/ba-10000-{ga_inner_density}.gr"
        )
        appra = APPR(ga)
        ca = set(appra.compute_appr(SEED))

        if STRATEGY == 0:  # No Strategy
            nga, ngb = ga, gb
        elif STRATEGY == 1:  # Strategy I
            nga, ngb = normalize_edge_weight(
                ga, gb, gb.number_of_edges(), ga.number_of_edges())
        elif STRATEGY == 2:  # Strategy II
            gm = merge_two_graphs(ga, gb)
            apprm = APPR(gm)
            apprm.compute_appr(SEED)
            edge_apprs = apprm.compute_edge_apprs()
            sum_ppr_a, sum_ppr_b = 0, 0
            for u, v in gm.edges():
                if u > v:
                    u, v = v, u
                if ga.has_edge(u, v):
                    sum_ppr_a += edge_apprs[(u, v)]
                if gb.has_edge(u, v):
                    sum_ppr_b += edge_apprs[(u, v)]
            nga, ngb = normalize_edge_weight(
                ga, gb, sum_ppr_b, sum_ppr_a)
        elif STRATEGY == 3:  # Strategy III
            gm = merge_two_graphs(ga, gb)
            apprm = APPR(gm)
            initial_cluster = set(apprm.compute_appr(SEED))
            edge_apprs = apprm.compute_edge_apprs()
            sum_ppr_a, sum_ppr_b = 0, 0
            for u, v in gm.edges():
                if u > v:
                    u, v = v, u
                if u not in initial_cluster or v not in initial_cluster:
                    continue
                if ga.has_edge(u, v):
                    sum_ppr_a += edge_apprs[(u, v)]
                if gb.has_edge(u, v):
                    sum_ppr_b += edge_apprs[(u, v)]
            nga, ngb = normalize_edge_weight(
                ga, gb, gb.number_of_edges(), ga.number_of_edges())

        gm = merge_two_graphs(nga, ngb)
        apprm = APPR(gm)
        cm = set(apprm.compute_appr(SEED))
        ros_a.append(relative_overlap(ca, cm))
        ros_b.append(relative_overlap(cb, cm))
        fs_a.append(f1(ca, cm))
        fs_b.append(f1(cb, cm))
        csize_a.append(len(ca))
        csize_b.append(len(cb))
        csize_m.append(len(cm))
        print(ga_inner_density, "done, eta:", eta.eta(
            (cnt+1) / len(ga_inner_densities)))
    filenames = ["tmp/ro", "tmp/f", "tmp/csize"]
    for i in range(len(filenames)):
        filenames[i] += f"-{clustering_label}-{num_communities}-{community_size}.png"

    draw_scatter(
        ga_inner_densities,
        [ros_a, ros_b],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)"
        ],
        title=f"Relative Overlap (clustering: {clustering_label}, strategy: {STRATEGY})",
        x_axis_title="inner-edge density of graph A",
        y_axis_title="Relative Overlap of graph A",
        top=1.01,
        bottom=-0.01,
        filename=filenames[0],
    )
    draw_scatter(
        ga_inner_densities,
        [fs_a, fs_b],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)"
        ],
        title=f"F-score (clustering: {clustering_label}, strategy: {STRATEGY})",
        x_axis_title="inner-edge density of graph A",
        y_axis_title="F-score",
        top=1.01,
        bottom=-0.01,
        filename=filenames[1],
    )
    draw_scatter(
        ga_inner_densities,
        [csize_a, csize_b, csize_m],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)",
            "graph A+B (merged graph of A and B)"
        ],
        title=f"Cluster size (clustering: {clustering_label}, strategy: {STRATEGY})",
        x_axis_title="inner-edge density of graph A",
        y_axis_title="Cluster size",
        bottom=0,
        filename=filenames[2],
    )
    filename = concatanate_images(
        filenames, f"tmp/{clustering_label}", 2, 2)
    image_url = upload_to_imgbb(filename)
    notify_slack(
        title=f'weighting strategy: {STRATEGY}, clustering: {clustering_label}',
        result=image_url
    )
    pass


def weight_edge_same_graph_diff_weight():
    weights = [i / 10 for i in range(1, 101)]
    # ga_inner_densities = [i for i in range(1, 11)]
    num_communities = 3
    community_size = 100
    gb = read_graph(
        f"tmp/graph/{0.3}-{0.01}-{num_communities}-{community_size}-mixedorder.gr"
    )
    SEED = 0
    clustering_label = "APPR"
    STRATEGIES = [0, 1, 2, 3]
    STRATEGY = STRATEGIES[1]

    apprb = APPR(gb)
    cb = set(apprb.compute_appr(SEED))
    ros_a, ros_b = [], []
    fs_a, fs_b = [], []
    csize_a, csize_b, csize_m = [], [], []
    ga_inner_density = 0.25
    ga = read_graph(
        f"tmp/graph/{ga_inner_density}-{0.01}-3-100-normalorder.gr"
    )
    appra = APPR(ga)
    ca = set(appra.compute_appr(SEED))
    eta = ETA()
    for cnt, w in enumerate(weights):
        nga, ngb = normalize_edge_weight(
            ga, gb,  1, w)
        gm = merge_two_graphs(nga, ngb)
        apprm = APPR(gm)
        cm = set(apprm.compute_appr(SEED))
        ros_a.append(relative_overlap(ca, cm))
        ros_b.append(relative_overlap(cb, cm))
        fs_a.append(f1(ca, cm))
        fs_b.append(f1(cb, cm))
        csize_a.append(len(ca))
        csize_b.append(len(cb))
        csize_m.append(len(cm))
        print(w, "done, eta:", eta.eta(
            (cnt+1) / len(weights)))
    filenames = ["tmp/ro", "tmp/f", "tmp/csize"]
    for i in range(len(filenames)):
        filenames[i] += f"-{clustering_label}-{num_communities}-{community_size}.png"

    draw_chart(
        weights,
        [ros_a, ros_b],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)"
        ],
        title=f"Relative Overlap (clustering: {clustering_label}, strategy: {STRATEGY}, seed: {SEED})",
        x_axis_title="weight on A",
        y_axis_title="Relative Overlap of graph A",
        top=1.01,
        bottom=-0.01,
        filename=filenames[0],
    )
    draw_chart(
        weights,
        [fs_a, fs_b],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)"
        ],
        title=f"F-score (clustering: {clustering_label}, strategy: {STRATEGY}, seed: {SEED})",
        x_axis_title="weight on A",
        y_axis_title="F-score",
        top=1.01,
        bottom=-0.01,
        filename=filenames[1],
    )
    draw_chart(
        weights,
        [csize_a, csize_b, csize_m],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)",
            "graph A+B (merged graph of A and B)"
        ],
        title=f"Cluster size (clustering: {clustering_label}, strategy: {STRATEGY}, seed: {SEED})",
        x_axis_title="weight on A",
        y_axis_title="Cluster size",
        bottom=0,
        filename=filenames[2],
    )
    filename = concatanate_images(
        filenames, f"tmp/{clustering_label}", 2, 2)
    image_url = upload_to_imgbb(filename)
    notify_slack(
        title=f'weighting strategy: {STRATEGY}, clustering: {clustering_label}',
        result=image_url
    )
    pass


def exp_similarity_compare_boxplot_appr():
    # ga_inner_densities = [float("0." + str(i)) for i in range(25, 36)]
    ga_inner_densities = [float("0." + str(i)) for i in range(1, 6)]
    # ga_inner_densities = [float("0." + str(i)) for i in range(290, 311, 2)]
    # ga_inner_densities = [float("0." + str(i)) for i in range(300, 301, 2)]
    inter_density = 0.01
    num_communities = 3
    community_size = 100
    directory = "tmp/graph"
    num_iterations = 10

    clustering_label = "APPR"

    gb = read_graph(
        f"{directory}/{0.3}-{inter_density}-{num_communities}-{community_size}-mixedorder.gr"
    )
    apprb = APPR(gb)

    ros_a, ros_b = [], []
    fs_a, fs_b = [], []
    for ga_inner_density in ga_inner_densities:
        ro_a = []
        f_a = []
        ro_b = []
        f_b = []
        for _ in range(num_iterations):
            ga = sbm(
                inter_density,
                ga_inner_density,
                num_communities,
                community_size,
                seed=None
            )
            gm = merge_two_graphs(ga, gb)
            appra = APPR(ga)
            apprm = APPR(gm)
            sum_roa = 0
            sum_rob = 0
            sum_fa = 0
            sum_fb = 0

            for seed in gm:
                ca = set(appra.compute_appr(seed))
                cb = set(apprb.compute_appr(seed))
                cm = set(apprm.compute_appr(seed))
                sum_roa += relative_overlap(ca, cm)
                sum_rob += relative_overlap(cb, cm)
                sum_fa += f1(ca, cm)
                sum_fb += f1(cb, cm)

            ro_a.append(sum_roa / len(gm))
            ro_b.append(sum_rob/len(gm))
            f_a.append(sum_fa / len(gm))
            f_b.append(sum_fb / len(gm))

        ros_a.append(ro_a)
        fs_a.append(f_a)
        ros_b.append(ro_b)
        fs_b.append(f_b)
        print(ga_inner_density, "done")

    filenames = ["tmp/ro_a", "tmp/f_a",
                 "tmp/ro_b", "tmp/f_b"]
    for i in range(len(filenames)):
        filenames[i] += f"-{clustering_label}-{num_communities}-{community_size}.png"

    draw_boxplot(
        ros_a,
        f"Relative Overlap (strategy: {strategy})",
        labels=[str(ga_inner_density)
                for ga_inner_density in ga_inner_densities],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="relative overlap of graph A",
        top=1.01,
        bottom=-0.01,
        filename=filenames[0],
    )
    draw_boxplot(
        fs_a,
        f"F-score (strategy: {strategy})",
        labels=[str(ga_inner_density)
                for ga_inner_density in ga_inner_densities],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="f-score of graph A",
        top=1.01,
        bottom=-0.01,
        filename=filenames[1],
    )

    draw_boxplot(
        ros_b,
        f"Relative Overlap (strategy: {strategy})",
        labels=[str(ga_inner_density)
                for ga_inner_density in ga_inner_densities],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="relative overlap of graph B",
        top=1.01,
        bottom=-0.01,
        filename=filenames[2],
    )
    draw_boxplot(
        fs_b,
        f"F-score (strategy: {strategy})",
        labels=[str(ga_inner_density)
                for ga_inner_density in ga_inner_densities],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="f-score of graph B",
        top=1.01,
        bottom=-0.01,
        filename=filenames[3],
    )

    filename = concatanate_images(
        filenames, f"tmp/{clustering_label}-{num_iterations}trials", 2, 2)
    image_url = upload_to_imgbb(filename)
    notify_slack(
        title=f'ga_inner_densities: {ga_inner_densities}, num_iterations: {num_iterations}',
        result=image_url
    )


def create_sbm_dataset(
    inter_density=0.01,
    inner_densities=[0.1, 0.2, 0.3, 0.4, 0.5],
    community_size=100,
    num_communities=3,
    directory="graph/sbm",
):
    for inner_density in inner_densities:
        g = sbm(inter_density, inner_density,
                num_communities, community_size, seed=None)
        filename = f"{directory}/{inner_density}-{inter_density}-{num_communities}-{community_size}-normalorder.gr"
        nx.write_edgelist(g, filename, data=False)
    inner_density = 0.3
    g = sbm(inter_density, inner_density,
            num_communities, community_size, seed=None)
    h = nx.Graph()
    renumber = {nd: (nd - (nd // community_size) * community_size)
                * num_communities + (nd // community_size) for nd in g}
    for u, v in g.edges():
        h.add_edge(renumber[u], renumber[v])
    filename = f"{directory}/{inner_density}-{inter_density}-{num_communities}-{community_size}-mixedorder.gr"
    nx.write_edgelist(h, filename, data=False)


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


def exp_similarity_compare():
    ga_inner_densities = [float("0." + str(i)) for i in range(10, 51)]
    inter_density = 0.01
    num_communities = 3
    community_size = 100
    directory = "tmp/graph"
    # create_sbm_dataset(
    #     inter_density=inter_density,
    #     inner_densities=ga_inner_densities,
    #     community_size=community_size,
    #     num_communities=num_communities,
    #     directory=directory,
    # )

    gb = read_graph(
        f"{directory}/{0.3}-{inter_density}-{num_communities}-{community_size}-mixedorder.gr"
    )
    cb = louvain_communities(gb)

    nmis_a, nmis_b = [], []
    fs_a, fs_b = [], []
    eds_a, eds_b = [], []
    for ga_inner_density in ga_inner_densities:
        ga_path = f"{directory}/{ga_inner_density}-{inter_density}-{num_communities}-{community_size}-normalorder.gr"

        ga = read_graph(ga_path)
        gm = merge_two_graphs(ga, gb)
        for u, v, d in gm.edges(data=True):
            if ga.has_edge(u, v) and gb.has_edge(u, v):
                print(d['weight'])
        ca = louvain_communities(ga)
        cm = louvain_communities(gm)
        nmi = normalized_mutual_information(ca, cm)
        nmis_a.append(nmi)
        f = f_score(ca, cm)
        fs_a.append(f)
        ed = edit_distance(ca, cm)
        eds_a.append(ed)

        cm = louvain_communities(gm)
        nmi = normalized_mutual_information(cb, cm)
        nmis_b.append(nmi)
        f = f_score(cb, cm)
        fs_b.append(f)
        ed = edit_distance(cb, cm)
        eds_b.append(ed)

    filenames = ["tmp/nmi", "tmp/f", "tmp/ed", ]

    for i in range(len(filenames)):
        filenames[i] += f"-{THRESHOLD}-{clustering_label}-{num_communities}-{community_size}.png"

    draw_chart(
        ga_inner_densities,
        [nmis_a, nmis_b],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)"
        ],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="nmi",
        top=1.01,
        bottom=-0.01,
        title="NMI",
        filename=filenames[0],
        print_filename=False,
    )
    draw_chart(
        ga_inner_densities,
        [fs_a, fs_b],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)"
        ],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="Infl",
        top=1.01,
        bottom=-0.01,
        title="Infl",
        filename=filenames[1],
        print_filename=False,
    )
    draw_chart(
        ga_inner_densities,
        [eds_a, eds_b],
        labels=[
            "graph A (VARYING inner-edge desnity)",
            "graph B (CONSTANT inner-edge density)"
        ],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="Edge distance",
        title="Edge Distance",
        filename=filenames[2],
        print_filename=False,
    )
    filename = concatanate_images(filenames, "tmp/image", 2, 2)
    upload_to_imgbb(filename)
    pass


def exp_similarity_compare_boxplot():
    ga_inner_densities = [float("0." + str(i)) for i in range(25, 36)]
    # ga_inner_densities = [float("0." + str(i)) for i in range(30, 31, 2)]
    inter_density = 0.01
    num_communities = 3
    community_size = 100
    directory = "tmp/graph"
    num_iterations = 10

    THRESHOLD = 0.35
    clustering_algorithms = [compute_global_cluster, louvain_communities]
    clustering_algorithm_names = ["global_cluster", "louvain"]
    CLUSTERING_ID = 0
    clustering = clustering_algorithms[CLUSTERING_ID]
    clustering_label = clustering_algorithm_names[CLUSTERING_ID]

    gb = read_graph(
        f"{directory}/{0.3}-{inter_density}-{num_communities}-{community_size}-mixedorder.gr"
    )
    cb = clustering(gb, THRESHOLD)

    nmis_a, nmis_b = [], []
    fs_a, fs_b = [], []
    eds_a, eds_b = [], []
    eta = ETA()
    for cnt, ga_inner_density in enumerate(ga_inner_densities):
        nmi_a = []
        f_a = []
        ed_a = []
        nmi_b = []
        f_b = []
        ed_b = []
        for i in range(num_iterations):
            ga = sbm(
                inter_density,
                ga_inner_density,
                num_communities,
                community_size,
                None
            )
            gm = merge_two_graphs(ga, gb)
            cm = clustering(gm, THRESHOLD)
            ca = clustering(ga, THRESHOLD)

            nmi_a.append(normalized_mutual_information(ca, cm))
            f_a.append(f_score(ca, cm))
            ed_a.append(edit_distance(ca, cm))
            nmi_b.append(normalized_mutual_information(cb, cm))
            f_b.append(f_score(cb, cm))
            ed_b.append(edit_distance(cb, cm))

        nmis_a.append(nmi_a)
        fs_a.append(f_a)
        eds_a.append(ed_a)
        nmis_b.append(nmi_b)
        fs_b.append(f_b)
        eds_b.append(ed_b)
        print(ga_inner_density, "done, eta:", eta.eta(
            (cnt+1) / len(ga_inner_densities)))

    filenames = ["tmp/nmi_a", "tmp/f_a", "tmp/ed_a",
                 "tmp/nmi_b", "tmp/f_b", "tmp/ed_b"]
    for i in range(len(filenames)):
        filenames[i] += f"-{THRESHOLD}-{clustering_label}-{num_communities}-{community_size}.png"

    draw_boxplot(
        nmis_a,
        f"NMI (threshold: {THRESHOLD}, clustering: {clustering_label})",
        labels=[str(ga_inner_density)
                for ga_inner_density in ga_inner_densities],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="nmi of graph A",
        top=1.01,
        bottom=-0.01,
        filename=filenames[0],
    )
    draw_boxplot(
        fs_a,
        f"Infl (threshold: {THRESHOLD}, clustering: {clustering_label})",
        labels=[str(ga_inner_density)
                for ga_inner_density in ga_inner_densities],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="Infl of graph A",
        top=1.01,
        bottom=-0.01,
        filename=filenames[1],
    )
    draw_boxplot(
        eds_a,
        f"Edge density (threshold: {THRESHOLD}, clustering: {clustering_label})",
        labels=[str(ga_inner_density)
                for ga_inner_density in ga_inner_densities],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="edge density of graph A",
        bottom=-0.01,
        filename=filenames[2],
    )

    draw_boxplot(
        nmis_b,
        f"NMI (threshold: {THRESHOLD}, clustering: {clustering_label})",
        labels=[str(ga_inner_density)
                for ga_inner_density in ga_inner_densities],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="nmi of graph B",
        top=1.01,
        bottom=-0.01,
        filename=filenames[3],
    )
    draw_boxplot(
        fs_b,
        f"Infl (threshold: {THRESHOLD}, clustering: {clustering_label})",
        labels=[str(ga_inner_density)
                for ga_inner_density in ga_inner_densities],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="Infl of graph B",
        top=1.01,
        bottom=-0.01,
        filename=filenames[4],
    )
    draw_boxplot(
        eds_b,
        f"Edge density (threshold: {THRESHOLD}, clustering: {clustering_label})",
        labels=[str(ga_inner_density)
                for ga_inner_density in ga_inner_densities],
        x_axis_title="inner-edge density of graph A",
        y_axis_title="edge density of graph B",
        bottom=-0.01,
        filename=filenames[5],
    )

    filename = concatanate_images(
        filenames, f"tmp/{THRESHOLD}-{clustering_label}", 3, 2)
    image_url = upload_to_imgbb(filename)
    notify_slack(
        title=f'clustering: {clustering_label}, threshold: {THRESHOLD}, ga_inner_densities: {ga_inner_densities}, num_iterations: {num_iterations}',
        result=image_url
    )
    pass


def choosing_seed_strategy():
    ga = read_graph(
        f"graph/barabasi-albert/ba-10000-25.gr"
    )
    gb = read_graph(
        f"graph/barabasi-albert/ba-10000-50.gr"
    )
    gm = merge_two_graphs(ga, gb)
    apprm = APPR(gm)
    covers = set()
    sum_cluster_size = 0
    nodes_and_degrees = [(gm.degree(nd), nd) for nd in gm]
    nodes_and_degrees.sort(reverse=True)
    nodes_in_degree_order = [x[1] for x in nodes_and_degrees]
    seeds = set()
    cover_ratios = []
    list_cover_cluster_size = []
    print(gm)
    for i, seed in enumerate(nodes_in_degree_order):
        # has_neighbor_in_seeds = False
        # for nbr in gm.neighbors(seed):
        #     if nbr in seeds:
        #         has_neighbor_in_seeds = True
        #         break
        # if has_neighbor_in_seeds:
        #     continue
        if seed in covers:
            continue
        seeds.add(seed)
        cm = apprm.compute_appr(seed, eps=0.00001)
        for nd in cm:
            covers.add(nd)
        cover_ratios.append(len(covers) / gm.number_of_nodes())
        sum_cluster_size += len(cm)
        list_cover_cluster_size.append(len(covers) / sum_cluster_size)
    draw_chart(
        [i for i in range(len(cover_ratios))],
        [cover_ratios, list_cover_cluster_size],
        labels=["cover ratio", "cover/size"],
        x_axis_title="#seeds",
        y_axis_title="ratio",
        top=1.0,
        bottom=-0.0,
        left=-0,
    )
    for nd in nodes_in_degree_order:
        if nd not in covers:
            print(nd, gm.degree(nd))
    pass


def rinko0920():
    axes = [
        (247, 165),
        (680, 200),
        (201, 556),
        (765, 511),
        (496, 669),
        (1250, 340),
        (1491, 359),
        (1148, 615),
        (1495, 587),
        (685, 1092),
        (1001, 1085),
        (828, 1228),
    ]

    partition = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 2,
        10: 2,
        11: 2,
    }
    g = read_graph("graph/ygao.gr")

    axes = normalize_axes(axes)
    pos = {}
    for i, axis in axes.items():
        pos[i] = axis
    # color
    node_color = [LIGHT_COLORS[partition[nd]] for nd in g]
    draw_graph(
        g,
        pos=pos,
        node_size=600,
        figsize=(6, 4),
        special_node=3,
        node_color=node_color,
        filename="tmp/graph.pdf",
        font_family='Hiragino Maru Gothic Pro'
    )
    appr = APPR(g)
    print(appr.compute_appr(3))
    node_in_order = appr.get_node_in_order()
    appr_vec = appr.get_appr_vec()
    print("|", end="")
    for nd in node_in_order:
        print(nd, end="|")
    print()
    print("|", end="")
    for nd in node_in_order:
        print(end=":--:|")
    print()
    print("|", end="")
    for nd in node_in_order:
        print(round(appr_vec[nd], 3), end="|")
    print()
    print("|", end="")
    for i, nd in enumerate(node_in_order[:-1]):
        print(round(nx.conductance(g, node_in_order[:i+1]), 3), end="|")
    print('-|')


def exp_two_radial_graph_dist_to_appr():
    LEN_CHAIN = 4
    ga = create_radial_graph(
        num_branches=1,
        len_branch=LEN_CHAIN
    )
    gb = create_radial_graph(
        num_branches=1,
        len_branch=LEN_CHAIN
    )
    gb = relabel_all_nodes(gb, ga.number_of_nodes(), set([0]))
    a_sizes, b_sizes, m_sizes = [], [], []
    wbs = [i for i in range(99, 100)]
    for wb in wbs:
        a, b = normalize_edge_weight(ga, gb, 1, wb)
        gm = merge_two_graphs(a, b)
        appr = APPR(gm)
        cluster = appr.compute_appr(0)
        # print(cluster)
        a_sizes.append(len(set(cluster).intersection(
            set([nd for nd in ga.nodes()]))))
        b_sizes.append(len(set(cluster).intersection(
            set([nd for nd in gb.nodes()]))))
        m_sizes.append(len(set(cluster).intersection(
            set([nd for nd in gm.nodes()]))))

        # # draw grpah
        appr_vec = appr.get_appr_vec()
        subg = create_n_hop_graph(gm, 0, 3)
        scores = {nd: round(appr_vec[nd], 6) for nd in subg}
    path = "tmp/yeah.png"
    draw_graph(subg, special_node=0, scores=scores, bold_edge=False,
               filename=path)
    # upload_to_imgbb(path)
    for nd in cluster:
        print(nd, appr_vec[nd])

    path = "tmp/yeah2.png"
    draw_chart(wbs, [a_sizes, b_sizes, m_sizes],
               labels=["a", "b", "m"], bottom=0, filename="tmp/yeah2.png")

    pass


def exp_radial_graph_dist_to_appr():
    g = create_radial_graph(
        num_branches=1,
        len_branch=50
    )
    appr = APPR(g)
    appr.compute_appr(0)
    appr_vec = appr.get_appr_vec()
    print(appr_vec)
    x_axis = [nd for nd in g]
    print(x_axis)
    y_axis = [appr_vec[x] for x in x_axis]
    path = "tmp/yeah.png"
    draw_chart(
        x_axis,
        [y_axis],
        x_axis_title="distance from seed (node 0)",
        y_axis_title="APPR score",
        filename=path
    )
    upload_to_imgbb(path)


def create_radial_graph(
    num_branches=1,
    len_branch=10000
):
    num_edges = num_branches * len_branch
    g = nx.Graph()
    for i in range(1, num_edges + 1):
        parent = i - num_branches
        if parent < 0:
            parent = 0
        g.add_edge(parent, i)
    return g


def exp_radial_graph():
    lens_branch = [i for i in range(1, 101)]
    scores_of_1 = []
    eta = ETA()
    prev = 1
    for i in lens_branch:
        g = create_radial_graph(
            num_branches=1,
            len_branch=i
        )
        appr = APPR(g)
        appr.compute_appr(0)
        appr_vec = appr.get_appr_vec()
        score = appr_vec[1]
        scores_of_1.append(score)
        if score - prev == 0:
            print(i, "same")
            lens_branch = lens_branch[:i]
            break
        prev = score
        print(eta.eta(i / len(lens_branch)), "s")
    path = "tmp/tmp.png"
    draw_chart(lens_branch, [scores_of_1], filename=path)
    upload_to_imgbb(path)


def rw_point():
    ga = read_graph("graph/Email-Enron.txt")
    gb = read_graph("graph/CA-GrQc.txt")
    for strategy in [None, "edgesize", "degree", "density"]:
        if strategy is None:
            gm = merge_two_graphs(ga, gb, data=True)
        elif strategy == "edgesize":
            nga, ngb = normalize_edge_weight(
                ga,
                gb,
                gb.number_of_edges(),
                ga.number_of_edges()
            )
            gm = merge_two_graphs(nga, ngb, data=True)
        elif strategy == "degree":
            nga, ngb = normalize_edge_weight(
                ga,
                gb,
                gb.number_of_edges() / gb.number_of_nodes(),
                ga.number_of_edges() / ga.number_of_nodes()
            )
            gm = merge_two_graphs(nga, ngb, data=True)
        elif strategy == "density":
            nga, ngb = normalize_edge_weight(
                ga,
                gb,
                gb.number_of_edges() / gb.number_of_nodes() / (gb.number_of_nodes() - 1),
                ga.number_of_edges() / ga.number_of_nodes() / (ga.number_of_nodes() - 1)
            )
            gm = merge_two_graphs(nga, ngb, data=True)
        elif strategy == "dynamic":
            gm = merge_two_graphs(ga, gb, data=False)
        appr = APPR(gm)
        print("strategy:", strategy, end=', ')
        appr.compute_appr_with_graphs(98, ga=ga, gb=gb)
    gm = merge_two_graphs(ga, gb)
    print("strategy: dynamic", end=', ')
    appr.compute_dynamic_appr(98, ga, gb)
    pass


def thenterMTG0921():
    ga = nx.Graph()
    gb = nx.Graph()
    ga.add_edges_from([(0, 1), (0, 3), (0, 4), (1, 3), (2, 4), (3, 4)])
    gb.add_edges_from([(1, 2), (1, 4), (2, 3)])
    gm = merge_two_graphs(ga, gb, data=False)
    appr = APPR(gm)
    appr.compute_dynamic_appr(0, ga, gb)
    pass


def thenterMTG0930():
    ga = read_graph(
        f"tmp/graph/0.3-0.01-3-100-mixedorder.gr"
    )
    ga_inner_densities = [float("0." + str(i)) for i in range(3, 4)]
    gb = sbm(0.01, 0.1, 3, 100, seed=None)
    exp_compare_strategies(
        [ga],
        ga_inner_densities,
        gb,
        [nd for nd in gb],
        # ["NWF", "edgesize", "degree", "density", "allocation-rw", "dynamic-rw"],
        ["NWF", "dynamic-rw"],
        # ["dynamic-rw"],
        "exp_sbm_all_nodes"
    )
