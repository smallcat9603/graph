# smallcat 221106

import pandas as pd
import networkx as nx
import sys, getopt
import time

def printUsage():
    print('Usage: python3 nx-compose.py <inputfile1> <inputfile2>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h") # opts = [("-h", " ")], args = [inputfile1, inputfile2]
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
    edgefile1 = args[0]
    edgefile2 = args[1]

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
    G1 = nx.read_edgelist(edgefile1, comments='#', nodetype=int)
    # G1.add_edges_from(G1.edges(), weight=1)

    G2 = nx.read_edgelist(edgefile2, comments='#', nodetype=int)
    # G2.add_edges_from(G2.edges(), weight=1)

    start = time.time()
    G0 = nx.compose(G1, G2)
    end = time.time()

    print("time = " + str(end - start))

    # write to file
    edgefile0 = edgefile1.split('.edges')[0] + "." + edgefile2.split('.edges')[0] + ".edges"
    df = pd.DataFrame(G0.edges())
    df.to_csv(edgefile0, sep=" ", index=False, header=False)

if __name__ == "__main__":
   main(sys.argv[1:])  
