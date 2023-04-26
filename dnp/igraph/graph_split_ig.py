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
    # node "name" = id
    for v in G.vs:
        v["name"] = v.index

    start = time.time()
    communities = G.community_fastgreedy().as_clustering(n=nsubgraphs)
    membership = np.array(communities.membership)
    subgraphs = [G.subgraph(np.where(membership == i)[0]) for i in range(nsubgraphs)]
    end = time.time()

    print("time = " + str(end - start))

    # write to file
    for n in range(nsubgraphs):
        sub = edgefile.split('.txt')[0] + ".sub" + str(n+1) + ".txt"
        g = subgraphs[n]

        # node id is from 0 in subgraph
        # df = pd.DataFrame(subgraphs[n].get_edgelist())

        # node id is maintained the same as in graph
        edgelist = [(g.vs[e.source]["name"], g.vs[e.target]["name"]) for e in g.es]
        df = pd.DataFrame(edgelist)

        df.to_csv(sub, sep=" ", index=False, header=False)

if __name__ == "__main__":
   main(sys.argv[1:])  
