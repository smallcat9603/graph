# smallcat 230515

import igraph as ig
import sys, getopt
import os

def printUsage():
    print('Usage: python3 {0} <edgelistfile>'.format(os.path.basename(__file__)))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h") # opts = [("-h", " ")], args = [ededgelistfile]
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
    edgefile = args[0]

    # create graph based on edge file
    # DO NOT USE igraph._igraph.GraphBase, USE SUBCLASS igraph.Graph instead
    # Read_Edgelist() in igraph is 0-based !!! (number vertices from 0)
    G = ig.Graph.Read_Edgelist(edgefile, directed=False)

    # check graph
    if(G.is_connected()):
        print("Graph is connected")
    else:
        print("Graph is not connected")
        nnodes = G.vcount()
        G = G.components().giant()
        print("The largest component is used instead: " + str(G.vcount()/nnodes))

    # print graph info
    print(G.summary())

    # generate connected graph edgelist file
    file = edgefile[:-len(".txt")] + ".connected.txt" # use removesuffix in python3.9
    G.write_edgelist(file)
    print(file + " generated.")

if __name__ == "__main__":
   main(sys.argv[1:])  
