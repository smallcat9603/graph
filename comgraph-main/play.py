from main import *

g = read_graph("graph/ca-grqc-connected.gr")
clusters = read_clusters("cluster/ca-grqc-connected/all-appr.output")
clusters2 = read_clusters("cluster/ca-grqc-connected/all-2-hop-appr.output")
clusters3 = read_clusters("cluster/ca-grqc-connected/all-3-hop-appr.output")
clusters4 = read_clusters("cluster/ca-grqc-connected/all-4-hop-appr.output")
clusters5 = read_clusters("cluster/ca-grqc-connected/all-5-hop-appr.output")
sub2 = read_clusters("cluster/ca-grqc-connected/sub-2-hop.output")
sub3 = read_clusters("cluster/ca-grqc-connected/sub-3-hop.output")
sub4 = read_clusters("cluster/ca-grqc-connected/sub-4-hop.output")
sub5 = read_clusters("cluster/ca-grqc-connected/sub-5-hop.output")

seed = 2654
for cond in [0.3]:
    communities = read_non_seeded_clusters(
        "cluster/ca-grqc-connected/k-means-" + str(cond) + ".output")
    print("conductacne:", cond, "seed:", seed)
    print(len(communities[seed]))
    int2hop = set(clusters2[seed]).intersection(communities[seed])
    print("2-hop nodes:",
          len(sub2[seed]), "local_clusters:", len(
              clusters2[seed]), "global_clusters in 2-hop:", len(communities[seed].intersection(sub2[seed])),
          "match:", len(int2hop))

    int3hop = set(clusters3[seed]).intersection(communities[seed])
    print("3-hop nodes:",
          len(sub3[seed]), "local_clusters:", len(
              clusters3[seed]), "global_clusters in 3-hop:", len(communities[seed].intersection(sub3[seed])),
          "match:", len(int3hop))
    int4hop = set(clusters4[seed]).intersection(communities[seed])
    print("4-hop nodes:",
          len(sub4[seed]), "local_clusters:", len(
              clusters4[seed]), "global_clusters in 4-hop:", len(communities[seed].intersection(sub4[seed])),
          "match:", len(int4hop))
    int5hop = set(clusters5[seed]).intersection(communities[seed])
    print("5-hop nodes:",
          len(sub5[seed]), "local_clusters:", len(
              clusters5[seed]), "global_clusters in 5-hop:", len(communities[seed].intersection(sub5[seed])),
          "match:", len(int5hop))
    inthop = set(clusters[seed]).intersection(communities[seed])
    print("all nodes:",
          len(g), "local_clusters:", len(
              clusters[seed]), "global_clusters:", len(communities[seed]),
          "match:", len(inthop))
