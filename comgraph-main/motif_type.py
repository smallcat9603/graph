class Suite(object):
    pass


class UEdge(Suite):
    # undirected edge
    pass


class M1(Suite):
    # u  --> v, v  --> w, w  --> u
    pass


class M2(Suite):
    # u <--> v, v  --> w, w  --> u
    pass


class M3(Suite):
    # u <--> v, v <--> w, w  --> u
    pass


class M4(Suite):
    # u <--> v, v <--> w, w <--> u
    pass


class M5(Suite):
    # u  --> v, v  --> w, u  --> w
    pass


class M6(Suite):
    # u <--> v, w  --> u, w  --> v
    pass


class M7(Suite):
    # u <--> v, u  --> w, v  --> w
    pass


class M8(Suite):
    # u  --> v, u  --> w
    pass


class M9(Suite):
    # u  --> v, w  --> u
    pass


class M10(Suite):
    # v  --> u, w  --> u
    pass


class M11(Suite):
    # u <--> v, u  --> w
    pass


class M12(Suite):
    # u <--> v, w  --> u
    pass


class M13(Suite):
    # u <--> v, u <--> w
    pass


class clique3(Suite):
    # u <--> v, v <--> w, w <--> u, undirected
    pass


TRIANGLES = {M1, M2, M3, M4, M5, M6, M7}
WEDGES = {M8, M9, M10, M11, M12, M13}
