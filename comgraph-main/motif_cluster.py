from motif_type import *
import networkx as nx


class MotifCluster:
    def __init__(self, dg: nx.DiGraph, motif):
        self._g = dg
        self._motif = motif
        self._num_motifs = 0
        self._gm = nx.Graph()
        self._gm.add_nodes_from(self._g.nodes())

    def get_motif_clustering(self):
        self._motif_adjacency()
        # return self._spectral_cut()

    def _motif_adjacency(self):
        if self._motif in TRIANGLES:
            return self._triangle_motif_adjacency()
        elif self._motif in WEDGES:
            return self._wedge_motif_adjacency()
        else:
            raise Exception("yet implemented")

    def _is_target_motif(self, u, v, w):
        if self._motif == M1:
            return self._is_motif_m1(u, v, w)
        elif self._motif == M2:
            return self._is_motif_m2(u, v, w)
        elif self._motif == M3:
            return self._is_motif_m3(u, v, w)
        elif self._motif == M4:
            return self._is_motif_m4(u, v, w)
        elif self._motif == M5:
            return self._is_motif_m5(u, v, w)
        elif self._motif == M6:
            return self._is_motif_m6(u, v, w)
        elif self._motif == M7:
            return self._is_motif_m7(u, v, w)
        elif self._motif == M8:
            return self._is_motif_m8(u, v, w)
        elif self._motif == M9:
            return self._is_motif_m9(u, v, w)
        elif self._motif == M10:
            return self._is_motif_m10(u, v, w)
        elif self._motif == M11:
            return self._is_motif_m11(u, v, w)
        elif self._motif == M12:
            return self._is_motif_m12(u, v, w)
        elif self._motif == M13:
            return self._is_motif_m13(u, v, w)
        else:
            raise Exception("undefined motif type")

    def _is_no_edge(self, u, v):
        return not self._g.has_edge(u, v) and not self._g.has_edge(v, u)

    def _is_unidir_edge(self, u, v):
        return self._g.has_edge(u, v) and not self._g.has_edge(v, u)

    def _is_bidir_edge(self, u, v):
        return self._g.has_edge(u, v) and self._g.has_edge(v, u)

    def _is_motif_m1(self, u, v, w):
        return (self._is_unidir_edge(u, v) and self._is_unidir_edge(v, w) and self._is_unidir_edge(w, u)) \
            or (self._is_unidir_edge(u, w) and self._is_unidir_edge(w, v) and self._is_unidir_edge(v, u))

    def _is_motif_m2(self, u, v, w):
        return ((self._is_bidir_edge(u, v) and self._is_unidir_edge(u, w) and self._is_unidir_edge(w, v)) or (self._is_bidir_edge(u, v) and self._is_unidir_edge(w, u) and self._is_unidir_edge(v, w)) or (self._is_bidir_edge(u, w) and self._is_unidir_edge(u, v) and self._is_unidir_edge(v, w)) or (self._is_bidir_edge(u, w) and self._is_unidir_edge(v, u) and self._is_unidir_edge(w, v)) or (self._is_bidir_edge(v, w) and self._is_unidir_edge(v, u) and self._is_unidir_edge(u, w)) or (self._is_bidir_edge(v, w) and self._is_unidir_edge(u, v) and self._is_unidir_edge(w, u)))

    def _is_motif_m3(self, u, v, w):
        if self._is_bidir_edge(u, v) and self._is_bidir_edge(v, w) and (self._is_unidir_edge(u, w) or self._is_unidir_edge(w, u)):
            return True
        if self._is_bidir_edge(u, w) and self._is_bidir_edge(w, v) and (self._is_unidir_edge(u, v) or self._is_unidir_edge(v, u)):
            return True
        if self._is_bidir_edge(w, u) and self._is_bidir_edge(u, v) and (self._is_unidir_edge(w, v) or self._is_unidir_edge(v, w)):
            return True
        return False

    def _is_motif_m4(self, u, v, w):
        return self._is_bidir_edge(u, v) and self._is_bidir_edge(u, w) and self._is_bidir_edge(v, w)

    def _is_motif_m5(self, u, v, w):
        if ((self._is_unidir_edge(u, v) and self._is_unidir_edge(u, w)) and (self._is_unidir_edge(v, w) or self._is_unidir_edge(w, v))):
            return True

        if ((self._is_unidir_edge(v, u) and self._is_unidir_edge(v, w)) and (self._is_unidir_edge(u, w) or self._is_unidir_edge(w, u))):
            return True

        if ((self._is_unidir_edge(w, v) and self._is_unidir_edge(w, u)) and (self._is_unidir_edge(v, u) or self._is_unidir_edge(u, v))):
            return True

        return False

    def _is_motif_m6(self, u, v, w):
        return (self._is_unidir_edge(u, v) and self._is_unidir_edge(u, w) and self._is_bidir_edge(v, w))\
            or (self._is_unidir_edge(v, u) and self._is_unidir_edge(v, w) and self._is_bidir_edge(u, w))\
            or (self._is_unidir_edge(w, u) and self._is_unidir_edge(w, v) and self._is_bidir_edge(u, v))

    def _is_motif_m7(self, u, v, w):
        return ((self._is_unidir_edge(v, u) and self._is_unidir_edge(w, u) and self._is_bidir_edge(v, w)) or (self._is_unidir_edge(u, v) and self._is_unidir_edge(w, v) and self._is_bidir_edge(u, w)) or (self._is_unidir_edge(u, w) and self._is_unidir_edge(v, w) and self._is_bidir_edge(u, v)))

    def _is_motif_m8(self, center, v, w):
        return self._is_no_edge(v, w) and self._is_unidir_edge(center, v) and self._is_unidir_edge(center, w)

    def _is_motif_m9(self, center, v, w):
        return self._is_no_edge(v, w) and ((self._is_unidir_edge(center, v) and self._is_unidir_edge(w, center)) or (self._is_unidir_edge(center, w) and self._is_unidir_edge(v, center)))

    def _is_motif_m10(self, center, v, w):
        return self._is_no_edge(v, w) and self._is_unidir_edge(v, center) and self._is_unidir_edge(w, center)

    def _is_motif_m11(self, center, v, w):
        return self._is_no_edge(v, w) and \
            ((self._is_bidir_edge(center, v) and self._is_unidir_edge(center, w))
             or (self._is_bidir_edge(center, w) and self._is_unidir_edge(center, v)))

    def _is_motif_m12(self, center, v, w):
        return self._is_no_edge(v, w) and \
            ((self._is_bidir_edge(center, v) and self._is_unidir_edge(w, center))
             or (self._is_bidir_edge(center, w) and self._is_unidir_edge(v, center)))

    def _is_motif_m13(self, center, v, w):
        return self._is_no_edge(v, w) and self._is_bidir_edge(center, v) and self._is_bidir_edge(center, w)

    def _triangle_motif_adjacency(self):
        for u in self._g.nodes():
            neighbors = set()
            neighbors = neighbors.union(self._g.predecessors(u))
            neighbors = neighbors.union(self._g.successors(u))

            for nbr1 in neighbors:
                if u >= nbr1:
                    continue
                for nbr2 in neighbors:
                    if nbr1 >= nbr2:
                        continue
                    # consider u < nbr1 < nbr2
                    if (self._is_target_motif(u, nbr1, nbr2)):
                        self._increment_weight(u, nbr1, nbr2)

    def _wedge_motif_adjacency(self):
        for center in self._g.nodes():
            neighbors = set()
            neighbors = neighbors.union(self._g.predecessors(center))
            neighbors = neighbors.union(self._g.successors(center))

            for nbr1 in neighbors:
                if center == nbr1:
                    continue
                for nbr2 in neighbors:
                    if center == nbr2 or nbr1 >= nbr2:
                        continue
                    if (self._is_target_motif(center, nbr1, nbr2)):
                        self._increment_weight(center, nbr1, nbr2)

    def _increment_weight(self, u, nbr1, nbr2):
        pairs = [(u, nbr1), (u, nbr2), (nbr1, nbr2)]
        for v, w in pairs:
            if self._gm.has_edge(v, w):
                self._gm[v][w]['weight'] += 1
            else:
                self._gm.add_edge(v, w, weight=1)
        self._num_motifs += 1

    def _sweep(self, spectral_ordering):
        vol = 0
        total_vol = self._num_motifs * 6
        vol_comp = total_vol
        cut = 0
        seen = set()
        conds = []
        for v in spectral_ordering:
            seen.add(v)
            for nbr in self._gm.neighbors(v):
                if v == nbr:
                    continue
                w = self._gm[v][nbr]['weight']
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

    def _spectral_cut(self):
        # Spectral Ordering of the nodes from the normalized motif Laplacian matrix
        import time
        start = time.time()
        spectral_ordering = nx.spectral_ordering(self._gm)
        end = time.time()
        print("spectral: ", end - start, "s", sep='')
        conds = self._sweep(spectral_ordering)
        min_index = conds.index(min(conds))
        return spectral_ordering[:min_index+1]
