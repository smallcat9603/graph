from cmath import inf
import sys
import networkx as nx
from read_graph import *
from collections import defaultdict, deque
import time


def ppr(g, seed):
    pv = {nd: 0 for nd in g.nodes()}
    pv[seed] = 1
    return nx.pagerank(g, personalization=pv, alpha=0.8, tol=0.0001)
    # return nx.pagerank(g, alpha=0.8)


def manual_ppr(g: nx.Graph, seed: int, alpha=0.8, eps=0.0001):
    residue = {nd: 0 for nd in g}
    reserve = {nd: 0 for nd in g}
    residue[seed] = 1
    q = deque()
    q.append(seed)
    # PPR by Anderson
    rmax = 1
    while rmax > eps:
        new_residue = {nd: 0 for nd in g}
        for nd in g:
            if residue[nd] == 0:
                continue
            deg_w = g.degree(nd)
            if (deg_w == 0):
                reserve[nd] += residue[nd]
                residue[nd] = 0
                continue
            reserve[nd] += residue[nd] * (1 - alpha)
            for nbr in g.neighbors(nd):
                new_residue[nbr] += residue[nd] * alpha / deg_w
        rmax = max(new_residue.values())
        residue = new_residue
    return reserve


def appr(g: nx.Graph, seed: int, alpha=0.8, eps=0.0001):
    residue = {nd: 0 for nd in g}
    reserve = {nd: 0 for nd in g}
    residue[seed] = 1
    q = deque()
    q.append(seed)
    # PPR by Anderson
    while (q):
        nd = q.popleft()
        deg_w = g.degree(nd)
        if (deg_w == 0):
            reserve[nd] += residue[nd]
            residue[nd] = 0
            continue
        push_val = residue[nd]
        reserve[nd] += push_val * (1 - alpha)
        residue[nd] = 0
        push_val *= alpha / deg_w
        for nbr in g.neighbors(nd):
            nbr_val_old = residue[nbr]
            residue[nbr] = nbr_val_old + push_val
            if nbr_val_old <= eps * g.degree(nbr) < residue[nbr]:
                q.append(nbr)
    return reserve


def appr2(g: nx.Graph, seed: int, alpha=0.98, eps=0.0001):
    residue = defaultdict(float)
    reserve = defaultdict(float)
    residue[seed] = 1
    q = deque([seed])
    while q:
        new_residue = defaultdict(float)
        while q:
            nd = q.popleft()
            deg_w = g.degree(nd)
            if (deg_w == 0):
                reserve[nd] += residue[nd]
                residue[nd] = 0
                continue
            push_val = residue[nd]
            reserve[nd] += push_val * (1 - alpha)
            residue[nd] = 0
            push_val *= alpha / deg_w
            for nbr in g.neighbors(nd):
                new_residue[nbr] += push_val
        residue = new_residue
        for nd, r in residue.items():
            if r > eps * g.degree(nd):
                q.append(nd)

    return reserve


if __name__ == "__main__":
    g = read_graph(sys.argv[1])
    seed = max(g)
    # result = list(ppr(g, seed).items())
    # result.sort(reverse=True, key=lambda x: x[1])
    # print(result[:10])
    # start = time.time()
    # for _ in range(100):
    #     result = list(manual_ppr(g, seed).items())
    # end = time.time()
    # result.sort(reverse=True, key=lambda x: x[1])
    # print(result[:10], end-start)

    start = time.time()
    for _ in range(10):
        result = list(appr(g, seed).items())
    end = time.time()
    result.sort(reverse=True, key=lambda x: x[1])
    print(result[:10], end-start)

    start = time.time()
    for _ in range(10):
        result = list(appr2(g, seed).items())
    end = time.time()
    result.sort(reverse=True, key=lambda x: x[1])
    print(result[:10], end-start)
