# smallcat 230426

import numpy as np
import pandas as pd
import igraph as ig
import sys, getopt
import os
import time

def printUsage():
    print('Usage: python3 {0} <edgelistfile> <nsubgraphs>'.format(os.path.basename(__file__)))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h") # opts = [("-h", " ")], args = [edgelistfile, nsubgraphs]
    except getopt.GetoptError:
        printUsage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        else:
            printUsage()
            sys.exit(2)
    if len(args) != 2:
            printUsage()
            sys.exit(2)       
    edgefile = args[0]
    nsubgraphs = int(args[1])

    # read edge info from edge file
    # columns = ["s", "d"]
    # data1 = pd.read_csv(edgefile1, comment="#", sep="\s+", names=columns)
    # edges1 = []
    # for row in range(len(data1)):
    #     edges1.append([data1["s"][row], data1["d"][row]])
    # data2 = pd.read_csv(edgefile2, comment="#", sep="\s+", names=columns)
    # edges2 = []
    # for row in range(len(data2)):
    #     edges2.append([data2["s"][row], data2["d"][row]])

    # create graph based on edge file
    # DO NOT USE igraph._igraph.GraphBase, USE SUBCLASS igraph.Graph instead
    # Read_Edgelist() in igraph is 0-based !!! (number vertices from 0)
    G = ig.Graph.Read_Edgelist(edgefile, directed=False)

    # generate a random graph with a specified node number and edge number
    # nnodes = 100
    # nedges = 1000
    # G = ig.Graph.Erdos_Renyi(n=nnodes, m=nedges, directed=False, loops=False) # not guarantee connectiveness

    # check graph
    if(G.is_connected()):
        print("Graph is connected")
    else:
        print("Graph is not connected")
        nnodes = G.vcount()
        G = G.components().giant()
        print("The largest component is used instead: " + str(G.vcount()/nnodes))
        # sys.exit(1)

    # print graph info
    print(G.summary())

    # node "name" = id
    for v in G.vs:
        v["name"] = v.index

    start = time.time()
    # communities = G.community_edge_betweenness(clusters=nsubgraphs).as_clustering()
    # communities = G.community_walktrap().as_clustering(n=nsubgraphs)
    communities = G.community_fastgreedy().as_clustering(n=nsubgraphs) # 1 community --> 1 server
    membership = np.array(communities.membership) # [0, 0, 1, 1, 2, 2, 2, ...]
    subgraphs = [G.subgraph(np.where(membership == i)[0]) for i in range(nsubgraphs)]
    end = time.time()

    print("time = " + str(end - start))

    # server id is from 0
    server_route_tables = [{} for _ in range(nsubgraphs)]
    for e in G.es:
        server_source = membership[e.source]
        server_target = membership[e.target]
        if server_source != server_target:
            if e.source in server_route_tables[server_source]:
                server_route_tables[server_source][e.source].append((e.target, server_target))
            else:
                server_route_tables[server_source][e.source] = [(e.target, server_target)]   
            if e.target in server_route_tables[server_target]:
                server_route_tables[server_target][e.target].append((e.source, server_source))
            else:
                server_route_tables[server_target][e.target] = [(e.source, server_source)]

    # write to file, server id is from 0
    for n in range(nsubgraphs):
        # generate subgraph edgelist files
        sub = edgefile.split('.txt')[0] + ".sub" + str(n) + ".txt"
        g = subgraphs[n]

        # node id is from 0 in subgraph
        # df = pd.DataFrame(subgraphs[n].get_edgelist())

        # node id is maintained the same as in graph
        edgelist = [(g.vs[e.source]["name"], g.vs[e.target]["name"]) for e in g.es]
        df = pd.DataFrame(edgelist)

        df.to_csv(sub, sep=" ", index=False, header=False)

        # generate subgraph edgenode route tables
        sources = list(server_route_tables[n].keys())
        targets_servers = list(server_route_tables[n].values())
        df = pd.DataFrame({'sources':sources, 'targets_servers':targets_servers})
        rt = edgefile.split('.txt')[0] + ".rt" + str(n) + ".txt"
        df.to_csv(rt, sep=" ", index=False, header=False)

if __name__ == "__main__":
   main(sys.argv[1:])  
