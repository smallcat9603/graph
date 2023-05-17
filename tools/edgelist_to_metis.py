# smallcat 230515

import igraph as ig
import sys, getopt, os

def printUsage():
    print('Usage: python3 {0} <edgelistfile>'.format(os.path.basename(__file__)))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h") # opts = [("-h", " ")], args = [edgelistfile]
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

    G = ig.Graph.Read_Edgelist(edgefile)
    vcount = G.vcount()
    ecount = G.ecount()
    metis_lines = [[] for n in range(vcount)]
    for edge in G.get_edgelist():
        u, v = edge
        metis_lines[u].append(v)
        metis_lines[v].append(u)

    output_file = edgefile.split(".txt")[0] + ".metis.txt"
    with open(output_file, 'w') as f:
        f.write(f"{vcount} {ecount}\n")
        for n in range(vcount):
            line = " ".join(str(x+1) for x in metis_lines[n]) + "\n" # metis nodes are 1-origin (igraph nodes are 0-origin)
            f.write(line)
    print(output_file + " generated.")

if __name__ == "__main__":
   main(sys.argv[1:])  
