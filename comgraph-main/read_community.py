import networkx as nx


def read_community(path):
    f = open(path)
    communities = []
    s = set()
    for line in f.readlines():
        if "#" in line:
            continue
        community = set(map(int, line.split()))
        community_tuple = tuple(sorted(list(community)))
        if community_tuple in s:
            continue
        communities.append(community)
        s.add(community_tuple)

    return communities
