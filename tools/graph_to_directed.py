# smallcat 230513

import igraph as ig
import sys, getopt
import os
import time

def printUsage():
    print('Usage: python3 {0} <undirectededgelistfile>'.format(os.path.basename(__file__)))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h") # opts = [("-h", " ")], args = [undirectededgelistfile]
    except getopt.GetoptError:
        printUsage()
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        else:
            printUsage()
            sys.exit(1)
    if len(args) != 1:
            printUsage()
            sys.exit(1)       
    edgefile = "data/" + args[0]

    # create graph based on edge file
    # DO NOT USE igraph._igraph.GraphBase, USE SUBCLASS igraph.Graph instead
    # Read_Edgelist() in igraph is 0-based !!! (number vertices from 0)
    G = ig.Graph.Read_Edgelist(edgefile, directed=False)

    # check graph (connectness of a directed graph is rigid)
    if(G.is_connected()):
        print("Graph is connected")
    else:
        print("Graph is not connected")
        sys.exit(1)

    # print graph info
    print(G.summary())

    start = time.time()
    uG = G.as_directed() # 0 -- 1 to 0 --> 1 --> 0
    end = time.time()

    print("time = " + str(end - start))

    file = edgefile.split('.txt')[0] + ".directed.txt"
    uG.write_edgelist(file)
    print(file + " generated.")

if __name__ == "__main__":
   main(sys.argv[1:])  
