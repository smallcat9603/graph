from cProfile import label
import statistics


def extract_cluster(f):
    c = dict()
    for line in f.readlines():
        seed_clusters = list(map(int, line.split()))
        c[seed_clusters[0]] = set(seed_clusters[1:])
    return c


def compare_cluster(clusters1: dict, clusters2: dict) -> list:
    result = []
    for seed in clusters1:
        c1: set = clusters1[seed]
        c2: set = clusters2[seed]
        c1_and_c2 = c1.intersection(c2)
        result.append((seed, len(c1), len(c2), len(c1_and_c2)))
    return result


def intrsc(c1: set, c2: set, c3: set = None):
    c1c2 = c1.intersection(c2)
    if c3 is None:
        return len(c1c2)
    return len(c1c2.intersection(c3))


def compare_amazon():
    graph_name = "com-amazon.ungraph"
    directory_path = "cluster/" + graph_name + "/"
    f2 = open(directory_path + "cluster-2hop.output")
    f2i = open(directory_path + "cluster-2hop-induced.output")
    f3i = open(directory_path + "cluster-3hop-induced.output")
    f4i = open(directory_path + "cluster-4hop-induced.output")
    f2t = open(directory_path + "cluster-2hop-totalvol.output")
    f2it = open(directory_path + "cluster-2hop-induced-totalvol.output")
    f3it = open(directory_path + "cluster-3hop-induced-totalvol.output")
    f4it = open(directory_path + "cluster-4hop-induced-totalvol.output")
    fa = open(directory_path + "cluster-all.output")
    f2all = open(directory_path + "2hop-all.output")

    clusters2 = extract_cluster(f2)
    clusters2i = extract_cluster(f2i)
    clusters3i = extract_cluster(f3i)
    clusters4i = extract_cluster(f4i)
    clusters2t = extract_cluster(f2t)
    clusters2it = extract_cluster(f2it)
    clusters3it = extract_cluster(f3it)
    clusters4it = extract_cluster(f4it)
    clustersa = extract_cluster(fa)
    clusters2all = extract_cluster(f2all)

    f = open("output/clusters-tv-" + graph_name + ".csv", "w")
    line = "seed,2-hop-tv,2-induced,3-induced,4-induced,2-hop-tv,2-induced-tv,3-induced-tv,4-induced-tv,all,"
    line += "2-hop-tv∩2-induced,2-hop-tv∩3-induced,2-hop-tv∩4-induced,2-hop-tv∩2-hop-tv,2-hop-tv∩2-induced-tv,2-hop-tv∩3-induced-tv,2-hop-tv∩4-induced-tv,2-hop-tv∩all,"
    line += "all∩2-hop-tv,all∩2-induced,all∩3-induced,all∩4-induced,all∩2-hop-tv,all∩2-induced-tv,all∩3-induced-tv,all∩4-induced-tv,2-hop-nodes\n"
    f.write(line)

    for seed in clusters2:
        c2 = clusters2[seed]
        c2i = clusters2i[seed]
        c3i = clusters3i[seed]
        c4i = clusters4i[seed]
        c2t = clusters2t[seed]
        c2it = clusters2i[seed]
        c3it = clusters3i[seed]
        c4it = clusters4i[seed]
        ca = clustersa[seed]
        c2all = clusters2all[seed]
        line = f'{seed}'
        line += f',{intrsc(c2, c2all)}'
        line += f',{intrsc(c2i, c2all)}'
        line += f',{intrsc(c3i, c2all)}'
        line += f',{intrsc(c4i, c2all)}'
        line += f',{intrsc(c2t, c2all)}'
        line += f',{intrsc(c2it, c2all)}'
        line += f',{intrsc(c3it, c2all)}'
        line += f',{intrsc(c4it, c2all)}'
        line += f',{intrsc(ca, c2all)}'
        line += f', {intrsc(c2, c2i)}'
        line += f', {intrsc(c2, c3i)}'
        line += f', {intrsc(c2, c4i)}'
        line += f', {intrsc(c2, c2t)}'
        line += f', {intrsc(c2, c2it)}'
        line += f', {intrsc(c2, c3it)}'
        line += f', {intrsc(c2, c4it)}'
        line += f', {intrsc(c2, ca)}'
        line += f', {intrsc(ca, c2, c2all)}'
        line += f', {intrsc(ca, c2i, c2all)}'
        line += f', {intrsc(ca, c3i, c2all)}'
        line += f', {intrsc(ca, c4i, c2all)}'
        line += f', {intrsc(ca, c2t, c2all)}'
        line += f', {intrsc(ca, c2it, c2all)}'
        line += f', {intrsc(ca, c3it, c2all)}'
        line += f', {intrsc(ca, c4it, c2all)}'
        line += f', {len(c2all)}'
        line += "\n"
        f.write(line)


def cluster_size():
    graph_name = "com-amazon.ungraph"
    directory_path = "cluster/" + graph_name + "/"
    fl = open(directory_path + "louvain.output")
    fln = open(directory_path + "louvain-near-size.output")
    fa = open(directory_path + "appr.output")
    fma = open(directory_path + "cluster-all.output")
    f2m = open(directory_path + "cluster-2hop.output")

    csl = extract_cluster(fl)
    csln = extract_cluster(fln)
    csa = extract_cluster(fa)
    csma = extract_cluster(fma)
    cs2m = extract_cluster(f2m)

    f = open("output/clustersize-" + graph_name + ".output", "w")
    f.write("seed,Louvain,near-sized Louvain,APPR,MAPPR,2-hop MAPPR\n")
    for seed in cs2m.keys():
        line = f'{seed},{len(csl[seed])},{len(csln[seed])},{len(csa[seed])},{len(csma[seed])},{len(cs2m[seed])}\n'
        f.write(line)


def prcnt(decimal):
    return str(round(decimal * 100, 2)) + "%"


def cluster_compare():
    graph_names = ["com-amazon.ungraph", "com-lj.ungraph",
                   "com-dblp.ungraph", "soc-Slashdot0902"]
    print("|graph|AVE. Louvain cluster|AVE. MAPPR cluster|AVE. diff|AVE. %diff|MED. diff|MED. %diff|#singles (%)|P(L\|M)|P(M\|L)|F1|")
    print("|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    for graph_name in graph_names:
        directory_path = "cluster/" + graph_name + "/"
        # fl = open(directory_path + "louvain.output")
        fln = open(directory_path + "louvain-" + graph_name + ".all.output")
        # fa = open(directory_path + "appr.output")
        fma = open(directory_path + "cluster-all.output")
        # f2m = open(directory_path + "cluster-2hop.output")

        # csl = extract_cluster(fl)
        csln = extract_cluster(fln)
        # csa = extract_cluster(fa)
        csma = extract_cluster(fma)
        # cs2m = extract_cluster(f2m)
        max_csma = 0

        f = open("output/mappr-vs-near-sized-louvain-" +
                 graph_name + ".csv", "w")

        seeds = [seed for seed in csln.keys()]
        sumln, summa, diff, diff_frac = [], [], [], []
        # methods = [csl, csln, csa, csma, cs2m]
        methods = [csln, csma]
        n = len(methods)
        precision_sum = [[0] * n for _ in range(n)]
        recall_sum = [[0] * n for _ in range(n)]
        f1_sum = [[0] * n for _ in range(n)]
        f.write("seed,MAPPR,near-sized Louvain,intersection\n")
        target_cnt = 0
        for seed in seeds:
            cln = csln[seed]
            cma = csma[seed]
            if len(cma) == 1:
                continue
            sumln.append(len(cln))
            summa.append(len(cma))
            diff.append(abs(len(cln) - len(cma)))
            diff_frac.append(
                abs(len(cln) - len(cma)) / len(cma))
            line = f'{seed},{len(cma)},{len(cln)},{len(cma.intersection(cln))}\n'
            f.write(line)
            for i, m1 in enumerate(methods):
                for j, m2 in enumerate(methods):
                    c1 = m1[seed]
                    c2 = m2[seed]
                    intersection = intrsc(c1, c2)
                    precision = intersection / len(c1)
                    recall = intersection / len(c2)
                    precision_sum[i][j] += precision
                    recall_sum[i][j] += recall
                    f1_sum[i][j] += 2 * precision * \
                        recall / (precision + recall)
            target_cnt += 1
        f.close()
        print("|", graph_name, "|", sep='', end='')
        print(sum(sumln) / target_cnt, end="|")
        print(sum(summa) / target_cnt, end="|")
        print(sum(diff) / target_cnt, end="|")
        print(prcnt(sum(diff_frac) / target_cnt), end="|")
        print(statistics.median(diff), end="|")
        print(prcnt(statistics.median(
            diff_frac)), end="|")
        num_singles = sum(1 if len(csma[seed]) == 1 else 0 for seed in seeds)
        print(num_singles, " (", prcnt(num_singles / len(seeds)), ")", sep='', end="|")
        precisions = [[0] * n for _ in range(n)]
        recalls = [[0] * n for _ in range(n)]
        f1s = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                precisions[i][j] = precision_sum[i][j] / target_cnt
                recalls[i][j] = recall_sum[i][j] / target_cnt
                f1s[i][j] = f1_sum[i][j] / target_cnt
        print(prcnt(precisions[1][0]), sep='', end='|')
        print(prcnt(precisions[0][1]), sep='', end='|')
        print("**", prcnt(f1s[0][1]), "**", sep='', end='|')
        print()


if __name__ == "__main__":
    cluster_compare()
    pass
