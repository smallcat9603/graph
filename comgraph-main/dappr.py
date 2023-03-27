from appr import *
from typing import Set, Dict


class DAPPR(APPR):
    def __init__(self, dg: nx.DiGraph):
        self._g: nx.DiGraph = dg
        self._out_weights: Dict[int, Dict[int, float]] = {}
        self._in_weights: Dict[int, Dict[int, float]] = {}
        self._in_out_weights: Dict[int, Dict[int, float]] = {}
        self._total_vol = 0
        self._appr_vec = defaultdict(float)
        self._max_edge_weights = defaultdict(float)
        self._node_in_order = []
        self._cond_profile = []
        self._size_global_min = 0
        self._size_first_local_min = -1

        self._delete_self_edges()
        self._create_weights()

    def _create_weights(self):
        for nd in self._g.nodes():
            # out_weights
            max_edge_weight = 0
            self._out_weights[nd] = {}
            self._in_out_weights[nd] = {}
            deg_out_w = 0
            for nbr in self._g.successors(nd):
                edge_data = self._g.get_edge_data(nd, nbr)
                w = edge_data['weight'] if 'weight' in edge_data else 1
                self._out_weights[nd][nbr] = w
                self._in_out_weights[nd][nbr] = w
                deg_out_w += w
                max_edge_weight = max(max_edge_weight, w)
            self._out_weights[nd][nd] = deg_out_w
            self._in_out_weights[nd][nd] = deg_out_w
            self._total_vol += deg_out_w
            self._max_edge_weights[nd] = max_edge_weight

            # in_weights
            self._in_weights[nd] = {}
            deg_in_w = 0
            for nbr in self._g.predecessors(nd):
                edge_data = self._g.get_edge_data(nbr, nd)
                w = edge_data['weight'] if 'weight' in edge_data else 1
                self._in_weights[nd][nbr] = w
                self._in_out_weights[nd][nbr] += w
                deg_in_w += w
            self._in_weights[nd][nd] = deg_in_w
            self._in_out_weights[nd][nd] += deg_in_w
            self._total_vol += deg_in_w

    def _delete_self_edges(self):
        for nd in self._g.nodes:
            if self._g.has_edge(nd, nd):
                self._g.remove_edge(nd, nd)

    def compute_appr(
        self,
        seed,
        degree_normalized=DEGREE_NORMALIZED,
        alpha=0.98,
        eps=0.0001,
    ):
        self._appr_vec = defaultdict(float)
        self._node_in_order = []
        self._cond_profile = []
        self._size_global_min = 0
        self._size_first_local_min = -1
        residue = defaultdict(float)

        weights = self._out_weights
        if weights[seed][seed] * eps >= 1:
            self._appr_vec[seed] = 0
            return [seed]
        residue[seed] = 1
        q = deque()
        q.append(seed)

        # PPR by Anderson
        while q:
            nd = q.popleft()

            deg_w = weights[nd][nd]
            if (deg_w == 0):
                self._appr_vec[nd] += residue[nd]
                continue
            push_val = residue[nd]
            self._appr_vec[nd] += push_val * (1 - alpha)

            push_val *= alpha / deg_w
            for nbr in self._g.successors(nd):
                nbr_val_old = residue[nbr]
                residue[nbr] = nbr_val_old + push_val * weights[nd][nbr]
                if nbr_val_old <= eps * weights[nbr][nbr] / self._max_edge_weights[nbr] < residue[nbr]:
                    q.append(nbr)

        self._compute_profile(self._total_vol, degree_normalized)

        return self._node_in_order[:self._size_global_min]

    def _compute_profile(self, total_vol, degree_normalized):
        quotient = []
        weights = self._in_out_weights

        # D^-1 * p
        for nd, val in self._appr_vec.items():
            if weights[nd][nd] == 0:
                quotient.append((nd, INF))
            else:
                if degree_normalized:
                    quotient.append((nd, val / weights[nd][nd]))
                else:
                    quotient.append((nd, val))
        quotient.sort(key=lambda x: x[1], reverse=True)
        # print(weights)
        # print(quotient)

        vol, cut = 0, 0
        is_in = set()
        vol_small = 1  # = 1 if volume(IsIn) <= VolAll/2, and = -1 otherwise;

        for nd, val in quotient:
            weights_here = weights[nd]
            self._node_in_order.append(nd)
            is_in.add(nd)
            vol += vol_small * weights_here[nd]

            if vol_small == 1 and vol >= total_vol / 2:
                vol = total_vol - vol
                vol_small = -1

            cut += weights_here[nd]
            for nbr in self._g.successors(nd):
                if nbr in is_in:
                    cut -= 2 * weights_here[nbr]
            if vol > 1 / total_vol:  # unless nd is the last node
                self._cond_profile.append(cut / vol)
            else:
                self._cond_profile.append(1)

        self._find_global_min()
        self._find_first_local_min()
