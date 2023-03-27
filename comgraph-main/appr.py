import networkx as nx
from collections import deque, defaultdict
import sys

INF = float('inf')
DEGREE_NORMALIZED = True
SUPERNODE_ID = -1002


class APPR:
    def __init__(self, g: nx.Graph, ga: nx.Graph = None, gb: nx.Graph = None):
        self._g = g
        self._ga = ga
        self._gb = gb
        self._weights = {}
        self._weights_a = {}
        self._weights_b = {}
        self._total_vol = 0
        self._total_vol_a = 0
        self._total_vol_b = 0
        self._appr_vec = defaultdict(float)
        self._max_edge_weights = defaultdict(float)
        self._max_edge_weights_a = defaultdict(float)
        self._max_edge_weights_b = defaultdict(float)
        self._node_in_order = []
        self._cond_profile = []
        self._size_global_min = 0
        self._size_first_local_min = -1

        self._delete_self_edges()
        self._create_weights()

    def top_appr(self):
        for i, v in enumerate(self._cond_profile):
            if v <= 0:
                return self._node_in_order[:i]
            return self._node_in_order

    def get_appr_vec(self):
        return self._appr_vec

    def get_node_in_order(self):
        return self._node_in_order

    def get_cond_profile(self):
        return self._cond_profile

    def compute_edge_apprs(self):
        edge_apprs = dict()
        for u, v in self._g.edges():
            if u > v:
                u, v = v, u
            edge_appr = \
                self._appr_vec[u] * self._weights[u][v] / self._weights[u][u] + \
                self._appr_vec[v] * self._weights[v][u] / self._weights[v][v]
            edge_apprs[(u, v)] = edge_appr
        return edge_apprs

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

        weights = self._weights
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
            if deg_w == 0:
                self._appr_vec[nd] += residue[nd]
                continue
            push_val = residue[nd]
            self._appr_vec[nd] += push_val * (1 - alpha)

            push_val *= alpha/deg_w
            for nbr in self._g.neighbors(nd):
                nbr_val_old = residue[nbr]
                residue[nbr] = nbr_val_old + push_val * weights[nd][nbr]
                if nbr_val_old <= eps * weights[nbr][nbr] < residue[nbr]:
                    q.append(nbr)

        self._compute_profile(self._total_vol, degree_normalized)

        return self._node_in_order[:self._size_global_min]

    def get_total_vol(self):
        return self._total_vol

    def get_graph(self):
        return self._g

    def _create_weights(self):
        for nd in self._g.nodes():
            deg_w = 0
            self._weights[nd] = {}
            max_edge_weight = 0
            for nbr in self._g.neighbors(nd):
                edge_data = self._g.get_edge_data(nd, nbr)
                if 'weight' in edge_data:
                    w = edge_data['weight']
                else:
                    w = 1
                self._weights[nd][nbr] = w
                deg_w += w
                max_edge_weight = max(max_edge_weight, w)
            self._weights[nd][nd] = deg_w
            self._total_vol += deg_w
            self._max_edge_weights[nd] = max_edge_weight
        if self._ga:
            for nd in self._ga.nodes():
                deg_w = 0
                self._weights_a[nd] = {}
                max_edge_weight = 0
                for nbr in self._ga.neighbors(nd):
                    edge_data = self._ga.get_edge_data(nd, nbr)
                    if 'weight' in edge_data:
                        w = edge_data['weight']
                    else:
                        w = 1
                    self._weights_a[nd][nbr] = w
                    deg_w += w
                    max_edge_weight = max(max_edge_weight, w)
                self._weights_a[nd][nd] = deg_w
                self._total_vol_a += deg_w
                self._max_edge_weights_a[nd] = max_edge_weight
        if self._gb:
            for nd in self._gb.nodes():
                deg_w = 0
                self._weights_b[nd] = {}
                max_edge_weight = 0
                for nbr in self._gb.neighbors(nd):
                    edge_data = self._gb.get_edge_data(nd, nbr)
                    if 'weight' in edge_data:
                        w = edge_data['weight']
                    else:
                        w = 1
                    self._weights_b[nd][nbr] = w
                    deg_w += w
                    max_edge_weight = max(max_edge_weight, w)
                self._weights_b[nd][nd] = deg_w
                self._total_vol_b += deg_w
                self._max_edge_weights_b[nd] = max_edge_weight

    def _compute_profile(self, total_vol, degree_normalized, weights=None):
        quotient = []
        if weights is None:
            weights = self._weights

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

        vol, cut = 0, 0
        is_in = set()
        vol_small = 1  # = 1 if volume(IsIn) <= VolAll/2, and = -1 otherwise;

        for nd, val in quotient:
            weights_here = self._weights[nd]
            self._node_in_order.append(nd)
            is_in.add(nd)
            vol += vol_small * weights_here[nd]

            if vol_small == 1 and vol >= total_vol / 2:
                vol = total_vol - vol
                vol_small = -1

            cut += weights_here[nd]
            for nbr in self._g.neighbors(nd):
                if nbr in is_in:
                    cut -= 2 * weights_here[nbr]
            if vol:
                self._cond_profile.append(cut / vol)
            else:
                self._cond_profile.append(1)
        self._find_global_min()
        self._find_first_local_min()

    def _find_global_min(self):
        min_cond_val = 2
        for i, m in enumerate(self._cond_profile):
            if (m < min_cond_val):
                self._size_global_min = i + 1
                min_cond_val = m

    def _find_first_local_min(self):
        self._size_first_local_min = 2
        while (self._size_first_local_min < len(self._cond_profile)):
            if self._is_local_min(self._size_first_local_min - 1):
                break
            self._size_first_local_min += 1
        if self._size_first_local_min >= len(self._cond_profile):
            if self._size_global_min == 0:
                self._find_global_min()
            self._size_first_local_min = self._size_global_min

    def _is_local_min(self, idx, thresh=1.2):
        if idx <= 0 or idx >= len(self._cond_profile) - 1:
            return False
        if (self._cond_profile[idx] >= self._cond_profile[idx - 1]):
            return False
        idx_right = idx
        while idx_right < len(self._cond_profile) - 1:
            idx_right += 1
            if self._cond_profile[idx_right] > self._cond_profile[idx] * thresh:
                return True
            elif self._cond_profile[idx_right] <= self._cond_profile[idx]:
                return False

        return False

    def _delete_self_edges(self):
        for nd in self._g.nodes:
            if self._g.has_edge(nd, nd):
                self._g.remove_edge(nd, nd)
        if self._ga:
            for nd in self._ga.nodes:
                if self._ga.has_edge(nd, nd):
                    self._ga.remove_edge(nd, nd)
        if self._gb:
            for nd in self._gb.nodes:
                if self._gb.has_edge(nd, nd):
                    self._gb.remove_edge(nd, nd)

    def compute_appr_data(
        self,
        seed,
        ga: nx.Graph,
        gb: nx.Graph,
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

        weights = self._weights
        if weights[seed][seed] * eps >= 1:
            self._appr_vec[seed] = 0
            return [seed]
        residue[seed] = 1
        q = deque()
        q.append(seed)
        rwera, rwerb = 0, 0

        # PPR by Anderson
        while q:
            nd = q.popleft()

            deg_w = weights[nd][nd]
            if deg_w == 0:
                self._appr_vec[nd] += residue[nd]

                continue
            push_val = residue[nd]
            self._appr_vec[nd] += push_val * (1 - alpha)

            push_val *= alpha/deg_w
            for nbr in self._g.neighbors(nd):
                nbr_val_old = residue[nbr]
                residue[nbr] = nbr_val_old + push_val * weights[nd][nbr]
                if ga.has_edge(nd, nbr) and gb.has_edge(nd, nbr):
                    rwera += push_val * weights[nd][nbr] / 2
                    rwerb += push_val * weights[nd][nbr] / 2
                elif ga.has_edge(nd, nbr):
                    rwera += push_val * weights[nd][nbr]
                elif gb.has_edge(nd, nbr):
                    rwerb += push_val * weights[nd][nbr]
                else:
                    print("this is impossible")
                    exit(0)

                if nbr_val_old <= eps * weights[nbr][nbr] / self._max_edge_weights[nbr] < residue[nbr]:
                    q.append(nbr)
        self._compute_profile(self._total_vol, degree_normalized)
        return self._node_in_order[:self._size_global_min], rwera, rwerb

    # def compute_dynamic_appr(
    #     self,
    #     seed,
    #     ga: nx.Graph,
    #     gb: nx.Graph,
    #     degree_normalized=DEGREE_NORMALIZED,
    #     r: float = 0.5,  # prob_ratio_b_over_a
    #     alpha=0.98,
    #     eps=0.0001,
    #     data=False,
    #     back_to_seed=False,
    # ):
    #     if r < 0 or 1 < r:
    #         print(f"r must be in [0, 1] but r  = {r}", file=sys.stderr)
    #         exit(0)
    #     self._appr_vec = defaultdict(float)
    #     self._node_in_order = []
    #     self._cond_profile = []
    #     self._size_global_min = 0
    #     self._size_first_local_min = -1
    #     residue = defaultdict(float)

    #     weights = self._weights
    #     if weights[seed][seed] * eps >= 1:
    #         self._appr_vec[seed] = 0
    #         return [seed]
    #     residue[seed] = 1
    #     q = deque([seed])
    #     flow_a, flow_b = 0, 0
    #     flow_a_at_switching, flow_b_at_switching = 0, 0
    #     while q:
    #         a, b, m = 0, 0, 0
    #         new_residue = defaultdict(float)
    #         only_to_a, only_to_b = [], []
    #         for nd in q:
    #             has_edge_in_a = ga.has_node(nd) and ga.degree(nd) != 0
    #             has_edge_in_b = gb.has_node(nd) and gb.degree(nd) != 0
    #             if has_edge_in_a and has_edge_in_b:
    #                 m += residue[nd]
    #             elif has_edge_in_a:
    #                 a += residue[nd]
    #                 only_to_a.append(nd)
    #             elif has_edge_in_b:
    #                 b += residue[nd]
    #                 only_to_b.append(nd)
    #         if back_to_seed:
    #             if c * (a + m) < b:
    #                 to_s = b - c * (a + m)
    #                 frac_hold = (b - to_s) / b
    #                 for nd in only_to_b:
    #                     self._appr_vec[nd] += residue[nd] * \
    #                         to_s / b * (1 - alpha)
    #                     residue[nd] *= frac_hold
    #                 new_residue[seed] += to_s * alpha
    #                 b = c * (a + m)
    #             elif c * a > b + m:
    #                 to_s = a - (b + m) / c
    #                 frac_hold = (a - to_s) / a
    #                 for nd in only_to_a:
    #                     self._appr_vec[nd] += residue[nd] * \
    #                         to_s / a * (1 - alpha)
    #                     residue[nd] *= frac_hold
    #                 new_residue[seed] += to_s * alpha
    #                 a = (b + m) / c
    #                 pass
    #         pa = - c * a + b + m
    #         pb = c * a - b + c * m
    #         if pa + pb == 0:
    #             pa, pb = 1, c
    #         if pa < 0:
    #             pa, pb = 0, 1
    #         if pb < 0:
    #             pa, pb = 1, 0

    #         to_a_per_step, to_b_per_step = 0, 0
    #         to_a_per_step_at_switching, to_b_per_step_at_switching = 0, 0
    #         while q:
    #             nd = q.popleft()
    #             deg_w = weights[nd][nd]
    #             if deg_w == 0:
    #                 self._appr_vec[nd] += residue[nd]
    #                 continue
    #             self._appr_vec[nd] += residue[nd] * (1 - alpha)

    #             push_val = alpha * residue[nd]
    #             sumw_a, sumw_b = 0, 0
    #             for nbr in self._g.neighbors(nd):
    #                 if ga.has_edge(nd, nbr):
    #                     sumw_a += ga[nd][nbr].get('weight', 1)
    #                 if gb.has_edge(nd, nbr):
    #                     sumw_b += gb[nd][nbr].get('weight', 1)

    #             s, to_a, to_b = 0, 0, 0
    #             to_a_at_switching, to_b_at_switching = 0, 0
    #             for nbr in self._g.neighbors(nd):
    #                 w = 0
    #                 if sumw_a != 0 and sumw_b != 0:
    #                     if ga.has_edge(nd, nbr):
    #                         ew = ga[nd][nbr].get('weight', 1)
    #                         w += ew / sumw_a * pa / (pa + pb)
    #                         to_a += push_val * ew / sumw_a * pa / (pa + pb)
    #                         to_a_at_switching += push_val * \
    #                             ew / sumw_a * pa / (pa + pb)
    #                     if gb.has_edge(nd, nbr):
    #                         ew = gb[nd][nbr].get('weight', 1)
    #                         w += ew / sumw_b * pb / (pa + pb)
    #                         to_b += push_val * ew / sumw_b * pb / (pa + pb)
    #                         to_b_at_switching += push_val * \
    #                             ew / sumw_b * pb / (pa + pb)
    #                 else:
    #                     ew = self._g[nd][nbr].get('weight', 1)
    #                     w = ew / deg_w
    #                     if ga.has_edge(nd, nbr):
    #                         to_a += push_val * w
    #                     if gb.has_edge(nd, nbr):
    #                         to_b += push_val * w
    #                 new_residue[nbr] += push_val * w
    #                 s += push_val * w
    #             to_a_per_step += to_a
    #             to_b_per_step += to_b
    #             to_a_per_step_at_switching += to_a_at_switching
    #             to_b_per_step_at_switching += to_b_at_switching
    #         flow_a += to_a_per_step
    #         flow_b += to_b_per_step
    #         flow_a_at_switching += to_a_per_step_at_switching
    #         flow_b_at_switching += to_b_per_step_at_switching
    #         residue = new_residue
    #         for nd, r in residue.items():
    #             # eps * d based on algorithm 1 of https://dl.acm.org/doi/pdf/10.1145/3448016.3457298
    #             if r > eps * weights[nd][nd]:
    #                 q.append(nd)
    #     # print("result:", flow_a, flow_b)
    #     self._compute_profile(self._total_vol, degree_normalized)

    #     if data:
    #         return self._node_in_order[:self._size_global_min], flow_a, flow_b, flow_a_at_switching, flow_b_at_switching
    #     else:
    #         return self._node_in_order[:self._size_global_min]

    def compute_allocation_appr(
        self,
        seed,
        ga: nx.Graph,
        gb: nx.Graph,
        degree_normalized=DEGREE_NORMALIZED,
        r: float = 0.5,
        alpha=0.98,
        eps=0.0001,
        data=False,
    ):
        if r < 0 or 1 < r:
            print(f"r must be in [0, 1] but r  = {r}", file=sys.stderr)
            exit(0)
        self._appr_vec = defaultdict(float)
        self._node_in_order = []
        self._cond_profile = []
        self._size_global_min = 0
        self._size_first_local_min = -1
        residue = defaultdict(float)

        weights = self._weights
        if weights[seed][seed] * eps >= 1:
            self._appr_vec[seed] = 0
            return [seed]
        residue[seed] = 1
        q = deque()
        q.append(seed)

        flow_a, flow_b = 0, 0
        flow_a_at_switching, flow_b_at_switching = 0, 0

        while q:
            nd = q.popleft()

            deg_w = weights[nd][nd]
            if deg_w == 0:
                self._appr_vec[nd] += residue[nd]
                continue
            push_val = residue[nd]
            self._appr_vec[nd] += push_val * (1 - alpha)
            push_val *= alpha
            sumw_a, sumw_b = 0, 0
            for nbr in self._g.neighbors(nd):
                if ga.has_edge(nd, nbr):
                    sumw_a += ga[nd][nbr].get('weight', 1)
                if gb.has_edge(nd, nbr):
                    sumw_b += gb[nd][nbr].get('weight', 1)

            to_a, to_b = 0, 0
            for nbr in self._g.neighbors(nd):
                nbr_val_old = residue[nbr]
                w = 0
                if sumw_a != 0 and sumw_b != 0:
                    if ga.has_edge(nd, nbr):
                        ew = ga[nd][nbr].get('weight', 1)
                        w += ew / sumw_a * r
                        to_a += push_val * \
                            ew / sumw_a * r
                        flow_a_at_switching += push_val * \
                            ew / sumw_a * r
                    if gb.has_edge(nd, nbr):
                        ew = gb[nd][nbr].get('weight', 1)
                        w += ew / sumw_b * (1 - r)
                        to_b += push_val * \
                            ew / sumw_b * (1 - r)
                        flow_b_at_switching += push_val * \
                            ew / sumw_b * (1 - r)
                else:
                    ew = self._g[nd][nbr].get('weight', 1)
                    w = ew / (sumw_a + sumw_b)
                    if ga.has_edge(nd, nbr):
                        to_a += push_val * w
                    if gb.has_edge(nd, nbr):
                        to_b += push_val * w
                residue[nbr] = nbr_val_old + push_val * w

                if nbr_val_old <= eps * weights[nbr][nbr] < residue[nbr]:
                    q.append(nbr)
            flow_a += to_a
            flow_b += to_b
        self._compute_profile(self._total_vol, degree_normalized)

        if data:
            return self._node_in_order[:self._size_global_min], flow_a, flow_b, flow_a_at_switching, flow_b_at_switching
        else:
            return self._node_in_order[:self._size_global_min]

    # def compute_dynamic_weighting_appr(
    #     self,
    #     seed,
    #     ga: nx.Graph,
    #     gb: nx.Graph,
    #     degree_normalized=DEGREE_NORMALIZED,
    #     r: float = 0.5,  # prob_ratio_b_over_a
    #     alpha=0.98,
    #     eps=0.0001,
    #     data=False,
    #     one_based=True,
    # ):
    #     if r < 0 or 1 < r:
    #         print(f"r must be in [0, 1] but r  = {r}", file=sys.stderr)
    #         exit(0)
    #     self._appr_vec = defaultdict(float)
    #     self._node_in_order = []
    #     self._cond_profile = []
    #     self._size_global_min = 0
    #     self._size_first_local_min = -1
    #     residue = defaultdict(float)

    #     weights = self._weights
    #     if weights[seed][seed] * eps >= 1:
    #         self._appr_vec[seed] = 0
    #         return [seed]
    #     residue[seed] = 1
    #     q = deque([seed])
    #     flow_a, flow_b = 0, 0
    #     flow_a_at_switching, flow_b_at_switching = 0, 0
    #     sum_tr = defaultdict(lambda: defaultdict(float))
    #     while q:
    #         a, b, m = 0, 0, 0
    #         new_residue = defaultdict(float)
    #         only_to_a, only_to_b = [], []
    #         for nd in q:
    #             has_edge_in_a = ga.has_node(nd) and ga.degree(nd) != 0
    #             has_edge_in_b = gb.has_node(nd) and gb.degree(nd) != 0
    #             if has_edge_in_a and has_edge_in_b:
    #                 m += residue[nd]
    #             elif has_edge_in_a:
    #                 a += residue[nd]
    #                 only_to_a.append(nd)
    #             elif has_edge_in_b:
    #                 b += residue[nd]
    #                 only_to_b.append(nd)
    #         pa = - c * a + b + m
    #         pb = c * a - b + c * m
    #         if pa + pb == 0:
    #             pa, pb = 1, c
    #         if pa < 0:
    #             pa, pb = 0, 1
    #         if pb < 0:
    #             pa, pb = 1, 0
    #         sum_residue = sum(residue.values())

    #         to_a_per_step, to_b_per_step = 0, 0
    #         to_a_per_step_at_switching, to_b_per_step_at_switching = 0, 0
    #         while q:
    #             nd = q.popleft()
    #             deg_w = weights[nd][nd]
    #             if deg_w == 0:
    #                 self._appr_vec[nd] += residue[nd]
    #                 continue
    #             self._appr_vec[nd] += residue[nd] * (1 - alpha)

    #             push_val = alpha * residue[nd]
    #             sumw_a, sumw_b = 0, 0
    #             for nbr in self._g.neighbors(nd):
    #                 if ga.has_edge(nd, nbr):
    #                     sumw_a += ga[nd][nbr].get('weight', 1)
    #                 if gb.has_edge(nd, nbr):
    #                     sumw_b += gb[nd][nbr].get('weight', 1)

    #             s, to_a, to_b = 0, 0, 0
    #             to_a_at_switching, to_b_at_switching = 0, 0
    #             for nbr in self._g.neighbors(nd):
    #                 w = 0
    #                 if sumw_a != 0 and sumw_b != 0:
    #                     if ga.has_edge(nd, nbr):
    #                         ew = ga[nd][nbr].get('weight', 1)
    #                         w += ew / sumw_a * pa / (pa + pb)
    #                         to_a += push_val * ew / sumw_a * pa / (pa + pb)
    #                         to_a_at_switching += push_val * \
    #                             ew / sumw_a * pa / (pa + pb)
    #                     if gb.has_edge(nd, nbr):
    #                         ew = gb[nd][nbr].get('weight', 1)
    #                         w += ew / sumw_b * pb / (pa + pb)
    #                         to_b += push_val * ew / sumw_b * pb / (pa + pb)
    #                         to_b_at_switching += push_val * \
    #                             ew / sumw_b * pb / (pa + pb)
    #                 else:
    #                     ew = self._g[nd][nbr].get('weight', 1)
    #                     w = ew / deg_w
    #                     if ga.has_edge(nd, nbr):
    #                         to_a += push_val * w
    #                     if gb.has_edge(nd, nbr):
    #                         to_b += push_val * w
    #                 new_residue[nbr] += push_val * w
    #                 sum_tr[nd][nbr] += w * sum_residue
    #                 sum_tr[nbr][nd] += w * sum_residue
    #                 sum_tr[nd][nd] += w * sum_residue
    #                 sum_tr[nbr][nbr] += w * sum_residue
    #                 s += push_val * w
    #             to_a_per_step += to_a
    #             to_b_per_step += to_b
    #             to_a_per_step_at_switching += to_a_at_switching
    #             to_b_per_step_at_switching += to_b_at_switching
    #         flow_a += to_a_per_step
    #         flow_b += to_b_per_step
    #         flow_a_at_switching += to_a_per_step_at_switching
    #         flow_b_at_switching += to_b_per_step_at_switching
    #         residue = new_residue
    #         for nd, r in residue.items():
    #             # eps * d based on algorithm 1 of https://dl.acm.org/doi/pdf/10.1145/3448016.3457298
    #             # if r > eps * weights[nd][nd]:
    #             #     q.append(nd)
    #             da = ga.degree(nd) if nd in ga else INF
    #             db = gb.degree(nd) if nd in gb else INF
    #             if nd in ga and nd in gb:
    #                 if r / (1 + c) / da > eps or r * (1 - r) / db > eps:
    #                     q.append(nd)
    #                 continue
    #             elif nd in ga:
    #                 if r / da > eps:
    #                     q.append(nd)
    #             else:
    #                 if r / db > eps:
    #                     q.append(nd)
    #     for nd in sum_tr:
    #         norm = sum_tr[nd][nd]
    #         if norm == 0:
    #             continue
    #         for nbr in sum_tr[nd]:
    #             sum_tr[nd][nbr] = sum_tr[nd][nbr] / norm
    #     ma = ga.number_of_edges()
    #     mb = gb.number_of_edges()
    #     nab = self._g.number_of_nodes()

    #     if one_based:
    #         norm = (2 * ma + 2 * c * mb + (1 + c)
    #                 * nab) / (2 * ma + 2 * mb)
    #     else:
    #         norm = (2 * ma + 2 * c * mb) / (2 * ma + 2 * mb)
    #     weights = defaultdict(lambda: defaultdict(float))

    #     for nd in sum_tr:
    #         da = ga.degree(nd) if nd in ga else 0
    #         db = gb.degree(nd) if nd in gb else 0
    #         if one_based:
    #             d = (da + c * db + 1 + c) / norm
    #         else:
    #             d = (da + c * db) / norm
    #         for nbr in self._g.neighbors(nd):
    #             weights[nd][nbr] += d * sum_tr[nd][nbr] / 2
    #             weights[nd][nd] += d * sum_tr[nd][nbr] / 2
    #             weights[nbr][nd] += d * sum_tr[nd][nbr] / 2
    #             weights[nbr][nbr] += d * sum_tr[nd][nbr] / 2
    #     self._compute_profile(self._total_vol, degree_normalized, weights)
    #     # print(weights)

    #     if data:
    #         return self._node_in_order[:self._size_global_min], flow_a, flow_b, flow_a_at_switching, flow_b_at_switching
    #     else:
    #         return self._node_in_order[:self._size_global_min]

    def compute_aclcut_c_appr(
        self,
        seed,
        omega: float = 1,
        degree_normalized=DEGREE_NORMALIZED,
        alpha=0.98,
        eps=0.0001,
    ):
        self._appr_vec = defaultdict(float)
        self._node_in_order = []
        self._cond_profile = []
        self._size_global_min = 0
        self._size_first_local_min = -1
        residues = [defaultdict(float), defaultdict(float)]

        weights_ab = [self._weights_a, self._weights_b]
        g_ab = [self._ga, self._gb]
        max_edge_weights_ab = [
            self._max_edge_weights_a, self._max_edge_weights_b]
        # if weights_ab[0][seed][seed] * eps >= 1 and weights_ab[1][seed][seed] * eps >= 1:
        #     self._appr_vec[seed] = 0
        #     return [seed]
        residues[0][seed] = 0.5  # initial RWer at seed in A
        residues[1][seed] = 0.5  # initial RWer at seed in B
        q = deque([(seed, 0), (seed, 1)])

        while q:
            nd, layer = q.popleft()
            the_other = 1 - layer

            # if weights_ab[0][nd][nd] == 0 and weights_ab[1][nd][nd] == 0:
            #     self._appr_vec[nd] += residues[layer][nd]
            #     continue

            push_val = residues[layer][nd]
            self._appr_vec[nd] += push_val * (1 - alpha)
            deg_w = weights_ab[layer][nd][nd]

            if nd in g_ab[the_other]:
                val_old = residues[the_other][nd]
                residues[the_other][nd] += push_val * omega / (omega + deg_w)
                if val_old <= eps * weights_ab[the_other][nd][nd] / max_edge_weights_ab[the_other][nd] < residues[the_other][nd]:
                    q.append((nd, the_other))
                push_val = push_val * deg_w / (omega + deg_w)

            residue = residues[layer]
            weights = weights_ab[layer]
            max_edge_weights = max_edge_weights_ab[layer]
            for nbr in g_ab[layer].neighbors(nd):
                nbr_val_old = residue[nbr]
                residue[nbr] += push_val * alpha * weights[nd][nbr] / deg_w

                if nbr_val_old <= eps * weights[nbr][nbr] / max_edge_weights[nbr] < residue[nbr]:
                    q.append((nbr, layer))
        self._compute_profile(self._total_vol, degree_normalized)

        return self._node_in_order[:self._size_global_min]

    def compute_aclcut_r_appr(
        self,
        seed,
        r: float = 1,
        degree_normalized=DEGREE_NORMALIZED,
        alpha=0.98,
        eps=0.0001,
    ):
        if r < 0 or 1 < r:
            print(f"r must be in [0, 1] but r == {r}", file=sys.stderr,)
            exit()

        self._appr_vec = defaultdict(float)
        self._node_in_order = []
        self._cond_profile = []
        self._size_global_min = 0
        self._size_first_local_min = -1
        residues = [defaultdict(float), defaultdict(float)]

        weights_ab = [self._weights_a, self._weights_b]
        g_ab = [self._ga, self._gb]
        max_edge_weights_ab = [
            self._max_edge_weights_a, self._max_edge_weights_b]
        residues[0][seed] = 0.5  # initial RWer at seed in A
        residues[1][seed] = 0.5  # initial RWer at seed in B
        q = deque([(seed, 0), (seed, 1)])

        while q:
            nd, layer = q.popleft()
            the_other = 1 - layer

            push_val = residues[layer][nd]
            self._appr_vec[nd] += push_val * (1 - alpha)

            if nd in g_ab[the_other]:
                push_val_the_other = push_val * r
                deg_w = weights_ab[the_other][nd][nd]
                g = g_ab[the_other]
                residue = residues[the_other]
                weights = weights_ab[the_other]
                max_edge_weights = max_edge_weights_ab[the_other]
                for nbr in g.neighbors(nd):
                    nbr_val_old = residue[nbr]
                    residue[nbr] += push_val_the_other * \
                        alpha * weights[nd][nbr] / deg_w
                    if nbr_val_old <= eps * weights[nbr][nbr] / max_edge_weights[nbr] < residue[nbr]:
                        q.append((nbr, the_other))
                push_val = push_val * (1 - r)

            deg_w = weights_ab[layer][nd][nd]
            g = g_ab[layer]
            residue = residues[layer]
            weights = weights_ab[layer]
            max_edge_weights = max_edge_weights_ab[layer]
            for nbr in g.neighbors(nd):
                nbr_val_old = residue[nbr]
                residue[nbr] += push_val * alpha * weights[nd][nbr] / deg_w

                if nbr_val_old <= eps * weights[nbr][nbr] / max_edge_weights[nbr] < residue[nbr]:
                    q.append((nbr, layer))
        self._compute_profile(self._total_vol, degree_normalized)

        return self._node_in_order[:self._size_global_min]

    def compute_appr_with_supernode(
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

        weights = self._weights
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
            if deg_w == 0:
                self._appr_vec[nd] += residue[nd]
                continue
            push_val = residue[nd]
            self._appr_vec[nd] += push_val * (1 - alpha)

            push_val *= alpha/deg_w
            for nbr in self._g.neighbors(nd):
                pushed_rwer = push_val * weights[nd][nbr]
                if nbr == SUPERNODE_ID:
                    nbr = seed
                nbr_val_old = residue[nbr]
                residue[nbr] = nbr_val_old + pushed_rwer
                if nbr_val_old <= eps * weights[nbr][nbr] < residue[nbr]:
                    q.append(nbr)

        self._compute_profile(self._total_vol, degree_normalized)

        return self._node_in_order[:self._size_global_min]
