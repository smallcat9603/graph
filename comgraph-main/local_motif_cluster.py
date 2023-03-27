from motif_type import *
import networkx as nx
import time
from collections import deque, defaultdict

INF = float('inf')


class MAPPR:
    def __init__(self, g: nx.Graph, motif, alpha=0.98, eps=0.0001):
        self._g = g
        self._motif = motif
        self._alpha = alpha
        self._eps = eps
        self._gw = g.__class__()
        self._weights = {}
        self._ksize = self._get_clique_size()
        self._total_vol = 0
        self._cnt = 0
        self._duration1 = 0
        self._duration2 = 0
        self._duration3 = 0
        self._appr_vec = defaultdict(float)
        self._node_in_order = []
        self._motif_cond_profile = []
        self._size_global_min = 0
        self._size_first_local_min = -1

        # To store the number of motif instances on each edge.
        # vec = Counts[u].GetDat(v) are the numbers of instances of several motif types on edge (u, v)
        # For undirected graph, vec[i] is the number of (i+3)-cliques on this edge, i.e., vec[0] is for triangle
        # For directed graph motif,
        #    vec[0] is for uni-directed edges,
        #    vec[1] is for bi-directed edges,
        #    vec[i] is for M_{i-1} as defined in Austin et al. (Science 2016).
        self._counts = {}

        self._gw.add_nodes_from(g.nodes())
        self._delete_self_edges()
        self._processed_graph()

    def compute_appr(self, seed_node, alpha=0.98, eps=0.0001, total_vol=None, original=True):
        self._appr_vec = defaultdict(float)
        self._node_in_order = []
        self._motif_cond_profile = []
        self._size_global_min = 0
        self._size_first_local_min = -1
        residual = defaultdict(float)
        num_pushes = 0
        appr_norm = 0
        weights = self._weights
        if weights[seed_node][seed_node] * eps >= 1:
            self._appr_vec[seed_node] = 0
            return [seed_node]
        residual[seed_node] = 1
        q = deque()
        q.append(seed_node)

        # PPR by Anderson
        while(q):
            num_pushes += 1
            nd = q.popleft()

            deg_w = weights[nd][nd]
            if (deg_w == 0):
                self._appr_vec[nd] += residual[nd]
                appr_norm += residual[nd]
                residual[nd] = 0
                continue
            push_val = residual[nd] - deg_w * eps / 2
            self._appr_vec[nd] += push_val * (1 - alpha)
            appr_norm += push_val * (1 - alpha)
            residual[nd] = deg_w * eps / 2

            push_val *= alpha/deg_w
            for nbr in self._gw.neighbors(nd):
                nbr_val_old = residual[nbr]
                nbr_val_new = nbr_val_old + push_val * weights[nd][nbr]
                residual[nbr] = nbr_val_new
                if (nbr_val_old <= eps * weights[nbr][nbr] and nbr_val_new > eps * weights[nbr][nbr]):
                    q.append(nbr)
        if total_vol:
            self._compute_profile(total_vol)
        else:
            self._compute_profile(self._total_vol)

        cluster = self._node_in_order[:self._size_global_min]

        if original:
            return cluster

        return self._single_node_cut(self._gw, cluster, seed_node)

    def get_node_in_order(self):
        return self._node_in_order

    def _single_node_cut(self, g: nx.Graph, cluster: list, s: int) -> list:
        assert g.has_node(s)
        sub = g.subgraph(cluster)
        inside = set([s] + [nd for nd in sub.neighbors(s)])
        reach = set()
        q = deque([nd for nd in sub.neighbors(s)])
        while q:
            nd = q.popleft()
            for nbr in sub.neighbors(nd):
                if nbr in inside:
                    continue
                if nbr in reach:
                    reach.remove(nbr)
                    inside.add(nbr)
                    q.append(nbr)
                    continue
                # we first meet this not
                reach.add(nbr)
        res = []
        for nd in cluster:
            if nd in inside:
                res.append(nd)
        return res

    def get_total_vol(self):
        return self._total_vol

    def get_processed_graph(self):
        return self._gw

    def compute_motif_clustering(self, nodes):
        vol, cut = 0, 0
        is_in = set()
        vol_small = 1  # = 1 if volume(IsIn) <= VolAll/2, and = -1 otherwise;
        total_vol = self._total_vol

        for nd in nodes:
            weights_here = self._weights[nd]
            self._node_in_order.append(nd)
            is_in.add(nd)
            vol += vol_small * weights_here[nd]

            if vol_small == 1 and vol >= total_vol / 2:
                vol = total_vol - vol
                vol_small = -1

            cut += weights_here[nd]
            for nbr in self._gw.neighbors(nd):
                if nbr in is_in:
                    cut -= 2 * weights_here[nbr]
        if vol:
            return cut / vol
        else:
            return 1

    def _compute_profile(self, total_vol):
        quotient = []
        weights = self._weights

        # D^-1 * p
        for nd, val in self._appr_vec.items():
            if weights[nd][nd] == 0:
                quotient.append((nd, INF))
            else:
                quotient.append((nd, val / weights[nd][nd]))
        quotient.sort(key=lambda x: x[1], reverse=True)

        vol, cut = 0, 0
        is_in = set()
        vol_small = 1  # = 1 if volume(IsIn) <= VolAll/2, and = -1 otherwise;

        # f = open("output/fb-harvard-conductance.txt", "w")

        for nd, val in quotient:
            weights_here = self._weights[nd]
            self._node_in_order.append(nd)
            is_in.add(nd)
            vol += vol_small * weights_here[nd]

            if vol_small == 1 and vol >= total_vol / 2:
                vol = total_vol - vol
                vol_small = -1

            cut += weights_here[nd]
            # print(cut, vol, cut/vol if vol else 1, sep=',')
            for nbr in self._gw.neighbors(nd):
                if nbr in is_in:
                    cut -= 2 * weights_here[nbr]
            # f.write(f'{cut},{vol},{cut/vol if vol else 1}\n')
            if vol:
                self._motif_cond_profile.append(cut / vol)
            else:
                self._motif_cond_profile.append(1)
        self._find_global_min()
        self._find_first_local_min()

    def _find_global_min(self):
        min_cond_val = 2
        for i, m in enumerate(self._motif_cond_profile):
            if (m < min_cond_val):
                self._size_global_min = i + 1
                min_cond_val = m

    def _find_first_local_min(self):
        self._size_first_local_min = 2
        while(self._size_first_local_min < len(self._motif_cond_profile)):
            if self._is_local_min(self._size_first_local_min - 1):
                break
            self._size_first_local_min += 1
        if self._size_first_local_min >= len(self._motif_cond_profile):
            if self._size_global_min == 0:
                self._find_global_min()
            self._size_first_local_min = self._size_global_min

    def _is_local_min(self, idx, thresh=1.2):
        if idx <= 0 or idx >= len(self._motif_cond_profile) - 1:
            return False
        if (self._motif_cond_profile[idx] >= self._motif_cond_profile[idx - 1]):
            return False
        idx_right = idx
        while idx_right < len(self._motif_cond_profile) - 1:
            idx_right += 1
            if self._motif_cond_profile[idx_right] > self._motif_cond_profile[idx] * thresh:
                return True
            elif self._motif_cond_profile[idx_right] <= self._motif_cond_profile[idx]:
                return False

        return False

    def _delete_self_edges(self):
        for nd in self._g.nodes:
            if self._g.has_edge(nd, nd):
                self._g.remove_edge(nd, nd)

    def _processed_graph(self):
        start3 = time.time()
        if self._g.is_directed():
            raise Exception("yet implemented")
        if self._ksize == 2:
            # Don't need to count, assign weights directly!
            for nd in self._g.nodes():
                self._weights[nd] = {}
                for nbr in self._g.neighbors(nd):
                    self._weights[nd][nbr] = 1
                self._weights[nd][nd] = self._g.degree(nd)
            self._total_vol = 2 * self._g.number_of_edges()
        else:
            for nd in self._g.nodes():
                nbr_lists = {}
                for nbr in self._g.neighbors(nd):
                    nbr_lists[nbr] = [0] * (self._ksize - 2)
                self._counts[nd] = nbr_lists
            prev_nodes = [0] * (self._ksize - 2)
            self._count_clique(self._g, prev_nodes, 0)

            # # for debugging
            # for nd, nbrs in self._counts.items():
            #     for nbr in nbrs.keys():
            #         print("id: ", nd, ", nbr: ", nbr, ", count: ",
            #               self._counts[nd][nbr][0], sep='')

            for nd in self._g.nodes():
                deg_w = 0
                self._weights[nd] = {}
                for nbr in self._g.neighbors(nd):
                    motif_cnt = self._counts[nd][nbr][self._ksize - 3]
                    if (motif_cnt):
                        self._gw.add_edge(nd, nbr, weight=motif_cnt)
                        self._weights[nd][nbr] = motif_cnt
                        deg_w += motif_cnt
                self._weights[nd][nd] = deg_w
                self._total_vol += deg_w
        end3 = time.time()
        # print("#nodes: ", self._g.number_of_nodes(),
        #       ", #edges: ", self._g.number_of_edges(), sep='')
        # print("motif count: ", self._total_vol, sep='')
        # print("duration1 time: ", self._duration1, sep='')
        # print("duration2 time: ", self._duration2, sep='')
        # print("total time: ", end3-start3, sep='')
        return

    # This function counts the undirected graph motif (clique) instances on each edge.
    # It uses recursive method for clique enumeration proposed by Chiba and Nishizeki (SIAM J. Comput. 1985).
    # Input self._ksize denotes the size of the clique we are to enumerate in the current graph self._g
    # Input {TIntV& PrevNodes} denotes a set of nodes that are directed connected to any node in the current graph G
    #    and {int level = PrevNodes.Len()} is the number of PreNodes. Therefore, any k-clique in G corresponds to
    #    a (k+level)-clique after all nodes in PrevNodes are added in the current graph G.
    def _count_clique(self, g, prev_nodes, level):
        start1 = time.time()
        self._cnt += 1
        if (level >= self._ksize - 1):
            raise Exception("exception")
        if level >= 1:
            for e in g.edges():
                src, dst = e[0], e[1]
                self._counts[src][dst][level - 1] += 1
                self._counts[dst][src][level - 1] += 1
        end1 = time.time()
        self._duration1 += end1 - start1
        for nd in g.nodes():
            start2 = time.time()
            d = g.degree(nd)
            for i in range(level):
                self._counts[prev_nodes[i]][nd][level - 1] += d
                self._counts[nd][prev_nodes[i]][level - 1] += d

            if level == self._ksize - 2:
                continue

            # Go to the next level
            prev_nodes[level] = nd
            neighbors = []
            for nbr in g.neighbors(nd):
                if self._higher_deg(nd, nbr):
                    neighbors.append(nbr)

            subgraph = g.subgraph(neighbors)
            num_edges = subgraph.number_of_edges()
            for i in range(level + 1):
                for j in range(i + 1, level + 1):
                    self._counts[prev_nodes[i]
                                 ][prev_nodes[j]][level] += num_edges
                    self._counts[prev_nodes[j]
                                 ][prev_nodes[i]][level] += num_edges

            end2 = time.time()
            self._duration2 += end2 - start2
            self._count_clique(subgraph, prev_nodes, level+1)
        # print("count:", self._cnt, "duration1:",
        #       self._duration1, "duration2:", self._duration2, "g.number_of_edges():", g.number_of_edges())
        return

    def _higher_deg(self, nd1, nd2):
        if self._g.degree(nd1) > self._g.degree(nd1):
            return True
        elif self._g.degree(nd1) == self._g.degree(nd1) and nd1 > nd2:
            return True
        return False

    def _get_clique_size(self):
        if (self._motif == UEdge):
            return 2
        elif (self._motif == clique3):
            return 3
        else:
            raise Exception("yet implemented")
