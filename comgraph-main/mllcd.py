import networkx as nx
from collections import deque, defaultdict
import sys
from typing import Set, List, Dict
from statistics import stdev
import math
from read_graph import *

INF = float('inf')


class MLLCD:
    def __init__(self, layers: Set[nx.Graph], beta: float):
        self._layers: Set[nx.Graph] = layers
        self._beta: float = beta
        self._jaccard: Dict[Dict] = defaultdict(dict)
        self._num_edges_in_c_each_layer: Dict[nx.Graph, int] = {
            layer: 0 for layer in self._layers}
        self._B = set()
        self._S = set()
        self._boundary2shells: Dict[int, set] = defaultdict(set)
        self._C = []
        self._C_set: Set[int] = set()
        self._lc_int_sum = 0
        self._lc_ext_sum = 0

    def _sim(self, u: int, v: int, layer: nx.Graph):
        if u not in layer or v not in layer:
            return 0
        smaller = min(u, v)
        larger = max(u, v)
        if layer not in self._jaccard or (smaller, larger) not in self._jaccard[layer]:
            nbr_u = set(nd for nd in layer.neighbors(smaller))
            nbr_v = set(nd for nd in layer.neighbors(larger))
            intersec = nbr_u.intersection(nbr_v)
            union = nbr_u.union(nbr_v)
            self._jaccard[layer][(smaller, larger)] = len(
                intersec) / len(union)
        return self._jaccard[layer][(smaller, larger)]

    def _sim_beta(self, u: int, v: int, stdev_before: float, stdev_after: float, layer: nx.Graph):
        bf = self._beta * (stdev_after - stdev_before)
        return 2 * self._sim(u, v, layer) / (1 + math.exp(- bf))

    def _compute_lc_int_contribution(self, nd: int, revert: bool = True):
        contribution = 0
        layer2nbrs = defaultdict(list)
        if len(self._layers) > 1:
            stdev_before = stdev(self._num_edges_in_c_each_layer.values())
        else:
            stdev_before = 0
        for layer in self._layers:
            if nd not in layer:
                continue
            for nbr in layer.neighbors(nd):
                if nbr in self._B:
                    layer2nbrs[layer].append(nbr)
                    self._num_edges_in_c_each_layer[layer] += 1

        if len(self._layers) > 1:
            stdev_after = stdev(self._num_edges_in_c_each_layer.values())
        else:
            stdev_after = 0
        for layer, nbrs in layer2nbrs.items():
            if revert:
                self._num_edges_in_c_each_layer[layer] -= len(nbrs)
            for nbr in nbrs:
                contribution += 2 * self._sim_beta(
                    nd,
                    nbr,
                    stdev_before=stdev_before,
                    stdev_after=stdev_after,
                    layer=layer
                )
        return contribution

    def _compute_lc_ext_contribution(self, nd: int, revert: bool = True):
        contribution = - self._compute_lc_int_contribution(nd)
        layer2nbrs = defaultdict(list)
        layer2newshell = defaultdict(list)
        if len(self._layers) > 1:
            stdev_before = stdev(self._num_edges_in_c_each_layer.values())
        else:
            stdev_before = 0
        for layer in self._layers:
            if nd not in layer:
                continue
            for nbr in layer.neighbors(nd):
                if nbr in self._B:
                    layer2nbrs[layer].append(nbr)
                    self._num_edges_in_c_each_layer[layer] += 1
                if nbr not in self._C:
                    # nbr is a new shell node
                    layer2newshell[layer].append(nbr)

        if len(self._layers) > 1:
            stdev_after = stdev(self._num_edges_in_c_each_layer.values())
        else:
            stdev_after = 0
        for layer, nbrs in layer2nbrs.items():
            if revert:
                self._num_edges_in_c_each_layer[layer] -= len(nbrs)
        for layer, newshells in layer2newshell.items():
            for newshell in newshells:
                contribution += 2 * self._sim_beta(
                    nd,
                    newshell,
                    stdev_before=stdev_before,
                    stdev_after=stdev_after,
                    layer=layer
                )
        return contribution

    def _update_c(self, nd: float):
        self._lc_int_sum += self._compute_lc_int_contribution(nd, revert=True)
        self._lc_ext_sum += self._compute_lc_ext_contribution(nd, revert=False)
        self._B.add(nd)
        self._C.append(nd)
        self._C_set.add(nd)
        del_lst = []
        shell = set()
        for layer in self._layers:
            if nd not in layer:
                continue
            for nbr in layer.neighbors(nd):
                if nbr not in self._C_set:
                    shell.add(nbr)
        self._boundary2shells[nd] = shell
        if nd in self._S:
            self._S.remove(nd)
        self._S = self._S.union(shell)
        for b, shells in self._boundary2shells.items():
            if nd in shells:
                shells.remove(nd)
            if len(shells) == 0:
                del_lst.append(b)
        for boundary in del_lst:
            del self._boundary2shells[boundary]
            self._B.remove(boundary)
        # print(f"nd: {nd}, B: {self._B}, C: {self._C}, S: {self._S}")

    def _compute_new_b(self, nd) -> int:
        lenb = len(self._B)
        for b, shells in self._boundary2shells.items():
            if nd in shells and len(shells) == 1:
                lenb -= 1
        for layer in self._layers:
            if nd not in layer:
                continue
            for nbr in layer.neighbors(nd):
                if nbr not in self._C:
                    return lenb + 1
        return lenb

    def _compute_new_lc(self, nd: int) -> float:
        contribution_int = self._compute_lc_int_contribution(nd, True)
        contribution_ext = self._compute_lc_ext_contribution(nd, True)
        lc_int = (self._lc_int_sum + contribution_int) / (len(self._C) + 1)
        lenb = self._compute_new_b(nd)
        if self._lc_ext_sum + contribution_ext == 0:
            return INF
        else:
            return lc_int / (self._lc_ext_sum + contribution_ext) * lenb

    def _lc(self):
        return self._lc_int_sum * len(self._B) / len(self._C) / self._lc_ext_sum

    def compute_mllcd(self, seed: int):
        self._num_edges_in_c_each_layer: Dict[nx.Graph, int] = {
            layer: 0 for layer in self._layers}
        self._C = [seed]
        self._C_set: Set[int] = set([seed])
        self._B = set(self._C)
        self._S = set(
            [nd for layer in self._layers for nd in layer.neighbors(seed)])
        self._boundary2shells: Dict[int, set] = defaultdict(set)
        self._boundary2shells[seed] = set(
            [nd for layer in self._layers for nd in layer.neighbors(seed)])
        self._lc_int_sum = 0
        self._lc_ext_sum = 0
        self._lc_ext_sum = self._compute_lc_ext_contribution(seed)
        try:
            best_value = self._lc()
        except ZeroDivisionError:
            return [seed]

        while True:
            best_lc = 0
            best_nd = -1
            for nd in self._S:
                new_lc = self._compute_new_lc(nd)
                if new_lc > best_lc:
                    best_nd = nd
                    best_lc = new_lc
            # print(
            #     f"chosen: {best_nd}, lc: {self._lc()}, edges: {self._num_edges_in_c_each_layer}")
            if best_lc >= best_value:
                self._update_c(best_nd)
                best_value = best_lc
            else:
                # print(f"best_lc: {best_lc}")
                break
        # print(self._C)
        # print(best_value)
        # print(self._lc())
        return self._C


if __name__ == "__main__":
    # ga = read_graph("graph/aucs-facebook.gr")
    # gb = read_graph("graph/aucs-lunch.gr")
    ga = read_graph("graph/ex_a.gr")
    gb = read_graph("graph/ex_b.gr")
    # ga = read_graph("graph/0.1-0.01-3-100-normalorder.gr")
    # gb = read_graph("graph/0.1-0.01-3-100-normalorder.gr")
    gs = [ga, gb]
    mllcd = MLLCD(gs, 0)
    print(mllcd.compute_mllcd(3))
    # print(mllcd.compute_mllcd(4))
